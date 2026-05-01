import os
import json
import re
import cv2
import numpy as np
from pdf2image import convert_from_bytes
from dotenv import load_dotenv
load_dotenv()
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator
from google import genai
from google.genai import types

from flask import Flask, render_template, request, session, redirect, url_for, Response


# ----------------- Flask setup -----------------

app = Flask(__name__)
app.secret_key = "change-this-secret-key"  # required for session (download JSON)


# ----------------- Pydantic models -----------------

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


# ----------------- GSTIN helpers -----------------

AMBITUOUS_RAW = [
    ("8", "B"),   # 8 misread instead of B
    ("0", "O"),   # 0 instead of O
    ("1", "I"),   # 1 instead of I
    ("l", "1"),   # lowercase L instead of 1
]

AMBIGUOUS_MAPS = [(w.upper(), a.upper()) for w, a in AMBITUOUS_RAW]

for wrong, alt in AMBIGUOUS_MAPS:
    if '5' in (wrong, alt) or (wrong == 'S' and alt == '5') or (wrong == '5' and alt == 'S'):
        raise RuntimeError("Ambiguous mapping must not include '5' <-> 'S' substitutions.")
    if '2' in (wrong, alt) or (wrong == 'Z' and alt == '2') or (wrong == '2' and alt == 'Z'):
        raise RuntimeError("Ambiguous mapping must not include '2' <-> 'Z' substitutions.")


def is_valid_gstin_format(gstin: str) -> bool:
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
    """Try limited OCR fixes for GSTINs. Will never touch IRN fields."""
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


# ----------------- IRN helpers -----------------

def is_valid_irn(irn: Optional[str]) -> bool:
    if not irn:
        return False
    return len(irn) == 64


def post_process_invoice_data(data: dict) -> dict:
    processed = data.copy()

    if processed.get("StockiestGST"):
        processed["StockiestGST"] = try_fix_gstin(processed["StockiestGST"])

    if processed.get("InstituteGST"):
        processed["InstituteGST"] = try_fix_gstin(processed["InstituteGST"])

    irn = processed.get("IRN64Digit")
    if irn is not None and not is_valid_irn(irn):
        print("Warning: IRN64Digit length is not 64 characters. Leaving it unchanged for manual review.")

    return processed
# ----------------- Prompt-based field filtering -----------------

def apply_prompt_field_filter(prompt_text: str, data: dict) -> dict:
    """
    Look at the prompt and only keep the fields the user explicitly asked for.
    For other fields in the schema, set them to empty values.
    """
    prompt_lower = prompt_text.lower()

    # All known top-level fields in your schema
    schema_fields = ["StockiestGST", "InstituteGST", "InvoiceDate", "IRN64Digit", "LineItems"]

    # Which fields are mentioned in the prompt?
    requested = set()
    for field in schema_fields:
        if field.lower() in prompt_lower:
            requested.add(field)

    # Special case: if prompt contains 'only', be strict
    strict_only = " only" in prompt_lower

    filtered = data.copy()

    for field in schema_fields:
        if field not in requested and strict_only:
            # User said "... only", so blank out everything not requested
            if field == "LineItems":
                filtered[field] = []
            else:
                filtered[field] = ""
        # else: leave as Gemini filled it

    return filtered



# ----------------- Preprocessing (contrast/clarity) -----------------

def preprocess_for_ocr(file_bytes: bytes, ext: str) -> tuple[bytes, str]:
    """
    Load PDF/JPG/PNG bytes, boost contrast & clarity, and return (processed_bytes, mime_type).
    Output is always PNG bytes for consistency.
    """
    ext = ext.lower()

    if ext == ".pdf":
        pages = convert_from_bytes(file_bytes, dpi=300)
        if not pages:
            raise RuntimeError("No pages found in PDF.")
        pil_img = pages[0]
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    elif ext in [".jpg", ".jpeg", ".png"]:
        arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("Failed to decode image bytes.")
    else:
        raise RuntimeError(f"Unsupported file type for preprocessing: {ext}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, enhanced = cv2.threshold(enhanced, 150, 255, cv2.THRESH_BINARY)


    success, buffer = cv2.imencode(".png", enhanced)
    if not success:
        raise RuntimeError("Failed to encode preprocessed image.")

    processed_bytes = buffer.tobytes()
    mime_type = "image/png"
    return processed_bytes, mime_type


# ----------------- Gemini client -----------------

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    print("⚠ WARNING: GOOGLE_API_KEY not set")
client = None
try:
    client = genai.Client(api_key=API_KEY)
except Exception as e:
    print("Error initializing Gemini Client.")
    print(f"Details: {e}")


# ----------------- Core extraction (prompt + file) -----------------

CRITICAL_RULES = """You are an expert invoice parser. Extract text using character-accurate OCR.

CRITICAL RULES:
- Extract all alphanumeric fields EXACTLY as printed.
- Preserve every character with no corrections, assumptions, or normalizations.
- Do NOT alter or replace characters that appear visually similar.
- Do NOT guess missing or unclear characters.
- For GSTIN (15 chars) and IRN (64 digits), perform strict character-by-character extraction.
- GSTIN format: XXAAAAA0000A1Z1 (15 chars exactly)
- If a field is uncertain or doesn't match expected format, return empty string "".

Do not hallucinate or fabricate any data.


Return ONLY valid JSON matching the schema provided by the tool (CustomTaxInvoice).

- If GST is given as a single percentage (e.g., 5% or 18%), split into:
  CGST + SGST (equal halves).
  Example:
  5% → 2.5% + 2.5%
  18% → 9.0% + 9.0%

  - Rate = List Price (before discount)
  - Ignore discounted final amount column

  - Always extract original List Price as Rate
  - Ignore discount and final amount columns
  - Ignore HSN/SAC column completely
  - Qty must be numeric only (ignore unit column)
"""


def extract_custom_invoice_data_with_prompt(
    prompt_text: str, file_bytes: bytes, filename: str
) -> dict:
    """
    Main pipeline used by the Flask route:
    - preprocess image/PDF bytes
    - call Gemini with custom prompt
    - validate with Pydantic schema
    - fix GSTINs / IRN
    """
    if client is None:
        raise RuntimeError("Gemini client not initialized. Check API key.")

    # Get file extension (.pdf / .jpg / .png ...)
    ext = os.path.splitext(filename)[1] or ""
    # Preprocess bytes -> high-contrast PNG bytes
    processed_bytes, processed_mime = preprocess_for_ocr(file_bytes, ext)

    file_part = types.Part.from_bytes(
        data=processed_bytes,
        mime_type=processed_mime,
    )

    if not prompt_text.strip():
        prompt_text = (
            "Extract all key invoice details into the JSON schema provided "
            "by the tool (CustomTaxInvoice)."
        )

    final_prompt = (
        prompt_text.strip()
        + "\n\n"
        + CRITICAL_RULES
        + "\n\nIMPORTANT: Respond with ONLY valid JSON. No extra commentary."
    )

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=CustomTaxInvoice,
        temperature=0.0,
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[final_prompt, file_part],
        config=config,
    )

    json_output = response.text.strip()

    # Validate & coerce using Pydantic
    model_obj = CustomTaxInvoice.model_validate(json.loads(json_output))
    raw_data = model_obj.model_dump()

    # 1) GSTIN / IRN post-processing
    processed_data = post_process_invoice_data(raw_data)

    # 2) Apply prompt-based filtering (e.g. "Fill InstituteGST only")
    final_data = apply_prompt_field_filter(prompt_text, processed_data)

    return final_data



# ----------------- Flask routes -----------------

@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    extracted_json = None

    default_prompt = (
        "You are an expert invoice parser. Extract all key invoice details into JSON.\n"
        "Fill StockiestGST, InstituteGST, InvoiceDate, IRN64Digit and LineItems with "
        "Description, Qty, Rate, GSTPercent."
    )

    prompt_text = default_prompt

    if request.method == "POST":
        prompt_text = request.form.get("prompt_text", default_prompt)
        file = request.files.get("invoice_file")

        if not file or file.filename == "":
            error = "Please upload a PDF or image file."
        else:
            try:
                file_bytes = file.read()
                data = extract_custom_invoice_data_with_prompt(
                    prompt_text, file_bytes, file.filename
                )
                extracted_json = data

                # Store for download
                session["last_json"] = json.dumps(
                    data, indent=2, ensure_ascii=False
                )

            except json.JSONDecodeError:
                error = "Model did not return valid JSON. Try refining your prompt."
            except Exception as e:
                error = f"Error while processing file: {e}"

    return render_template(
        "index.html",
        error=error,
        prompt_text=prompt_text,
        extracted_json=extracted_json,
    )


@app.route("/download-json")
def download_json():
    last_json = session.get("last_json")
    if not last_json:
        return redirect(url_for("index"))

    return Response(
        last_json,
        mimetype="application/json",
        headers={
            "Content-Disposition": "attachment; filename=extracted_invoice.json"
        },
    )


if __name__ == "__main__":
    app.run(debug=True)
