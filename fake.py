import os
import json
import re
import cv2
import numpy as np
from pdf2image import convert_from_path

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator
from google import genai
from google.genai import types


# ---------- Pydantic models ----------

class LineItem(BaseModel):
    Description: str
    Qty: str
    Rate: str
    GSTPercent: str = Field(
        description="The combined SGST+CGST percentage, e.g., '6.0% + 6.0%'"
    )


class CustomTaxInvoice(BaseModel):
    StockiestGST: str = Field(min_length=0, max_length=15)
    InstituteGST: str = Field(min_length=0, max_length=15)
    InvoiceDate: str
    IRN64Digit: Optional[str] = Field(None, max_length=64)
    LineItems: List[LineItem] = Field(
        description="A list of all individual products and their details."
    )

    @field_validator("StockiestGST", "InstituteGST")
    def validate_gstin_length(cls, v: str) -> str:
        if len(v) not in (0, 15):
            raise ValueError("GSTIN must be exactly 15 characters or empty")
        return v


# ---------- GSTIN helpers ----------

# One-directional: from likely wrong -> likely right
# IMPORTANT: Do NOT include mappings that swap '5' and 'S'. Runtime assertion below enforces that.
AMBITUOUS_RAW = [
    ("8", "B"),   # 8 misread instead of B
    ("0", "O"),   # 0 instead of O
    ("1", "I"),   # 1 instead of I
    ("l", "1")   # lowercase L instead of 1
    # ("5", "S"),  # intentionally disabled: do NOT map 5 <-> S
]

# Uppercase-normalized list used by the fix function
AMBIGUOUS_MAPS = [(w.upper(), a.upper()) for w, a in AMBITUOUS_RAW]

# Safety check: fail fast if any mapping involves '5' or 'S' <-> '5'
for wrong, alt in AMBIGUOUS_MAPS:
    if '5' in (wrong, alt) or (wrong == 'S' and alt == '5') or (wrong == '5' and alt == 'S'):
        raise RuntimeError("Ambiguous mapping must not include '5' <-> 'S' substitutions.")
    if '2' in (wrong, alt) or (wrong == 'Z' and alt == '2') or (wrong == '2' and alt == 'Z'):
        raise RuntimeError("Ambiguous mapping must not include '2' <-> 'Z' substitutions.")


def is_valid_gstin_format(gstin: str) -> bool:
    """Basic GSTIN format validation (excluding checksum for now)"""
    if len(gstin) != 15:
        return False

    pattern = r'^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$'
    return bool(re.match(pattern, gstin.upper()))


def _gstin_position_constraints() -> List[str]:
    return [
        "digit",        # 0
        "digit",        # 1
        "letter",       # 2
        "letter",       # 3
        "letter",       # 4
        "letter",       # 5
        "letter",       # 6
        "digit",        # 7
        "digit",        # 8
        "digit",        # 9
        "digit",        # 10
        "letter",       # 11
        "alnum_no_zero",# 12
        "literal_Z",    # 13
        "alnum",        # 14
    ]


def _allowed_at_position(pos: int, ch: str) -> bool:
    constraints = _gstin_position_constraints()
    if pos < 0 or pos >= len(constraints):
        return False
    c = constraints[pos]
    if c == "digit":
        return ch.isdigit()
    if c == "letter":
        return ch.isalpha() and len(ch) == 1
    if c == "alnum":
        return ch.isalnum()
    if c == "alnum_no_zero":
        return (ch.isalpha() and len(ch) == 1) or (ch.isdigit() and ch != "0")
    if c == "literal_Z":
        return ch == "Z"
    return False


def try_fix_gstin(gstin: str) -> str:
    """Try limited OCR fixes for GSTINs. Will never touch IRN fields.

    Only applies single-character substitutions from AMBIGUOUS_MAPS,
    and only if the replacement character is valid at that GSTIN position.
    """
    if is_valid_gstin_format(gstin):
        return gstin

    original = gstin
    chars = list(gstin.upper())

    for i, ch in enumerate(chars):
        for wrong, alt in AMBIGUOUS_MAPS:
            if ch == wrong:
                if not _allowed_at_position(i, alt):
                    continue
                chars[i] = alt
                candidate = "".join(chars)
                if is_valid_gstin_format(candidate):
                    return candidate
                chars[i] = ch

    return original


# ---------- IRN helpers ----------

def is_valid_irn(irn: Optional[str]) -> bool:
    """Minimal IRN validation: ensure length == 64. No character correction."""
    if not irn:
        return False
    return len(irn) == 64


def post_process_invoice_data(data: dict) -> dict:
    """Post-process extracted data with OCR corrections.

    - Only GSTINs are subject to limited automatic correction.
    - IRN64Digit is preserved as-is. If its length is not 64, we do NOT attempt automatic fixes.
    """
    processed = data.copy()

    # Fix GSTINs only
    if processed.get("StockiestGST"):
        processed["StockiestGST"] = try_fix_gstin(processed["StockiestGST"])

    if processed.get("InstituteGST"):
        processed["InstituteGST"] = try_fix_gstin(processed["InstituteGST"])

    # Preserve IRN exactly as provided. Do not attempt substitutions (especially not 5<->S).
    irn = processed.get("IRN64Digit")
    if irn is not None:
        if not is_valid_irn(irn):
            print("Warning: IRN64Digit length is not 64 characters. Leaving it unchanged for manual review.")

    return processed

def preprocess_for_ocr(file_path: str) -> tuple[bytes, str]:
    """
    Load PDF/JPG/PNG, boost contrast & clarity, and return (processed_bytes, mime_type).
    - PDFs: first page -> high-res image
    - Images: read and enhance directly
    Output is always PNG bytes for consistency.
    """
    ext = os.path.splitext(file_path)[1].lower()

    # ---- Load as image ----
    if ext == ".pdf":
        # Convert first page of PDF to image (300 dpi for clarity)
        pages = convert_from_path(file_path, dpi=300)
        if not pages:
            raise RuntimeError("No pages found in PDF.")
        pil_img = pages[0]
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    elif ext in [".jpg", ".jpeg", ".png"]:
        img = cv2.imread(file_path)
        if img is None:
            raise RuntimeError(f"Failed to read image file: {file_path}")
    else:
        raise RuntimeError(f"Unsupported file type for preprocessing: {ext}")

    # ---- Enhance contrast & clarity ----

    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2) Apply CLAHE (adaptive histogram equalization) to boost contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 3) Optional: slight sharpening for text edges
    # kernel = np.array([[0, -1, 0],
    #                    [-1, 5, -1],
    #                    [0, -1, 0]], dtype=np.float32)
    # enhanced = cv2.filter2D(enhanced, -1, kernel)

    # 4) Encode as PNG bytes for Gemini
    success, buffer = cv2.imencode(".png", enhanced)
    if not success:
        raise RuntimeError("Failed to encode preprocessed image.")

    processed_bytes = buffer.tobytes()
    mime_type = "image/png"
    return processed_bytes, mime_type


# ---------- Main extraction function ----------

def extract_custom_invoice_data(
    file_path: str, output_filename: str = "invoice_data.json"
):
    API_KEY = os.getenv("GOOGLE_API_KEY") or "YOUR_API_KEY_HERE"

    if API_KEY == "YOUR_API_KEY_HERE":
        print("⚠ WARNING: Replace YOUR_API_KEY_HERE with a real key or set GOOGLE_API_KEY.")
    try:
        client = genai.Client(api_key=API_KEY)
    except Exception as e:
        print("Error initializing Gemini Client.")
        print(f"Details: {e}")
        return

    # 🔍 PREPROCESS: boost contrast & clarity, get PNG bytes
    try:
        processed_bytes, mime_type = preprocess_for_ocr(file_path)
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        return

    file_part = types.Part.from_bytes(
        data=processed_bytes,
        mime_type=mime_type,
    )

    # ... keep the rest of your existing code (prompt, config, model call, post_process) the same ...


    text_prompt = """You are an expert invoice parser. Extract text using character-accurate OCR.

CRITICAL RULES:
- Extract all alphanumeric fields EXACTLY as printed.
- Preserve every character with no corrections, assumptions, or normalizations.
- Do NOT alter or replace characters that appear visually similar.
- Do NOT guess missing or unclear characters.
- For GSTIN (15 chars) and IRN (64 digits), perform strict character-by-character extraction.
- GSTIN format: XXAAAAA0000A1Z1 (15 chars exactly)
- If a field is uncertain or doesn't match expected format, return empty string "".

Do not hallucinate or fabricate any data.

Return ONLY valid JSON in this schema. Empty strings for missing/uncertain fields.
"""

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=CustomTaxInvoice,  # guides the model
        temperature=0.0,
    )

    print(f"Sending document ({os.path.basename(file_path)}) to Gemini for parsing as {mime_type}...")

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[text_prompt, file_part],
            config=config,
        )

        json_output = response.text.strip()

        # Use Pydantic to validate structure/types
        model_obj = CustomTaxInvoice.model_validate(json.loads(json_output))
        raw_data = model_obj.dict()

        # Post-process GSTINs (IRN left untouched)
        processed_data = post_process_invoice_data(raw_data)

        final_json = json.dumps(processed_data, indent=2, ensure_ascii=False)

        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(final_json)

        print(f"\n✅ Success! Data extracted and saved to: {output_filename}")
        print("--- Extracted JSON Preview ---")
        print(final_json)

    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing failed: {e}")
        print("Raw response:", json_output[:500])
    except Exception as e:
        print(f"An error occurred: {e}")


# ---------- Entry point ----------

if __name__ == "__main__":
    # Now this can be PDF, JPG, JPEG, or PNG
    FILE_PATH = "pod (8).pdf"
    extract_custom_invoice_data(FILE_PATH)
