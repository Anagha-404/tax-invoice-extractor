import os
import json
import re
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
# ---------- GSTIN helpers ----------

# One-directional: from likely wrong -> likely right
AMBIGUOUS_MAPS = [
    ("8", "B"),   # 8 misread instead of B
    ("0", "O"),   # 0 instead of O
    ("1", "I"),   # 1 instead of I
    ("l", "1"),   # lowercase L instead of 1
    # ("5", "S"),  # uncomment ONLY if you really need 5 -> S
]


def is_valid_gstin_format(gstin: str) -> bool:
    """Basic GSTIN format validation (excluding checksum for now)"""
    if len(gstin) != 15:
        return False
    
    pattern = r'^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$'
    return bool(re.match(pattern, gstin.upper()))


def try_fix_gstin(gstin: str) -> str:
    """Try to fix common OCR errors in GSTIN using ambiguous character mapping"""
    if is_valid_gstin_format(gstin):
        return gstin
    
    original = gstin
    chars = list(gstin.upper())
    
    # Try single character substitutions
    for i, ch in enumerate(chars):
        for wrong, alt in AMBIGUOUS_MAPS:
            if ch == wrong:
                chars[i] = alt
                candidate = "".join(chars)
                if is_valid_gstin_format(candidate):
                    return candidate
                chars[i] = ch  # revert
    
    return original



def post_process_invoice_data(data: dict) -> dict:
    """Post-process extracted data with OCR corrections."""
    processed = data.copy()

    # Fix GSTINs
    if processed.get("StockiestGST"):
        processed["StockiestGST"] = try_fix_gstin(processed["StockiestGST"])

    if processed.get("InstituteGST"):
        processed["InstituteGST"] = try_fix_gstin(processed["InstituteGST"])

    return processed


# ---------- Main extraction function ----------

def extract_custom_invoice_data(
    pdf_file_path: str, output_filename: str = "invoice_data.json"
):
    # ⚠ For safety, move this to an env var later
    API_KEY = "AIzaSyD6ugkkJ4jzA7h4Fvdv4DIbY66i2ThuC5Y"

    try:
        client = genai.Client(api_key=API_KEY)
    except Exception as e:
        print("Error initializing Gemini Client.")
        print(f"Details: {e}")
        return

    # Read PDF
    try:
        with open(pdf_file_path, "rb") as f:
            pdf_bytes = f.read()
    except FileNotFoundError:
        print(f"Error: PDF file not found at {pdf_file_path}")
        return

    pdf_part = types.Part.from_bytes(
        data=pdf_bytes,
        mime_type="application/pdf",
    )

    text_prompt = """You are an expert invoice parser. Extract text using character-accurate OCR.

CRITICAL RULES:
- Extract all alphanumeric fields EXACTLY as printed.
- Preserve every character with no corrections, assumptions, or normalizations.
- Do NOT alter or replace characters that appear visually similar.
- Do NOT guess missing or unclear characters.
- For GSTIN (15 chars) and IRN (64 digits), perform strict character-by-character extraction.
- GSTIN format: XXAAAAA0000A1Z1 (15 chars exactly)
- If a field is uncertain or doesn't match expected format, return empty string "".

Do not hullucinate or fabricate any data.

Return ONLY valid JSON in this schema. Empty strings for missing/uncertain fields.
"""

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=CustomTaxInvoice,  # guides the model
        temperature=0.0,
    )

    print(f"Sending document ({os.path.basename(pdf_file_path)}) to Gemini for parsing...")

    try:
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=[text_prompt, pdf_part],
            config=config,
        )

        json_output = response.text.strip()

        # Use Pydantic to validate structure/types
        model_obj = CustomTaxInvoice.model_validate(json.loads(json_output))
        raw_data = model_obj.dict()

        # Post-process GSTINs
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
    PDF_FILE_PATH = "pod (8).pdf"
    extract_custom_invoice_data(PDF_FILE_PATH)
