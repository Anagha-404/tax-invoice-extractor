import os
import json
from pydantic import BaseModel, Field
from typing import List, Optional
from google import genai
from google.genai import types

# ---------- Pydantic models ----------

class LineItem(BaseModel):
    Description: str
    Qty: str
    Rate: str
    GSTPercent: str = Field(description="The combined SGST+CGST percentage, e.g., '6.0% + 6.0%'")

class CustomTaxInvoice(BaseModel):
    StockiestGST: str = Field(description="The GSTIN of the supplier (07AAFCI8134F1ZM).")
    InstituteGST: str = Field(description="The GSTIN of the buyer (07AAATL0242R2ZE).")
    InvoiceDate: str = Field(description="The date the invoice was issued (06 Oct 2023).")
    IRN64Digit: Optional[str] = Field(
        None,
        description="The 64-digit IRN number (adae2ff089948d3a94036ef818b250240ea1534043328e8e99f06c8a6481ab0f)."
    )
    LineItems: List[LineItem] = Field(description="A list of all individual products and their details.")

# ---------- Main extraction function ----------

def extract_custom_invoice_data(pdf_file_path: str, output_filename: str = 'invoice_data.json'):
    # 1) Get API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("⚠ ERROR: GOOGLE_API_KEY environment variable is not set.")
        print("Set it and rerun, or hardcode the key in the code.")
        return

    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        print("Error initializing Gemini Client.")
        print(f"Details: {e}")
        return

    # 2) Read PDF
    try:
        with open(pdf_file_path, "rb") as f:
            pdf_bytes = f.read()
    except FileNotFoundError:
        print(f"❌ Error: PDF file not found at {pdf_file_path}")
        return

    pdf_part = types.Part.from_bytes(
        data=pdf_bytes,
        mime_type='application/pdf'
    )

    # 3) Prompt + config
    text_prompt = (
        """You are an invoice extraction engine.
        Read this tax invoice and return ONLY valid JSON with this exact structure:
        {
  "StockiestGST": "",
  "InstituteGST": "",
  "InvoiceDate": "",
  "IRN64Digit": "",
  "LineItems": [
    {
      "Description": "",
      "Qty": "",
      "Rate": "",
      "GSTPercent": ""
    }
  ]
}"""
    )

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=CustomTaxInvoice,
        temperature=0.0
    )

    print(f"📤 Sending document ({os.path.basename(pdf_file_path)}) to Gemini for parsing...")

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[text_prompt, pdf_part],
        config=config
    )

    try:
        # With response_mime_type = "application/json", response.text should be pure JSON
        json_output = response.text

        # Save to file
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(json_output)

        print(f"\n✅ Success! Data extracted and saved to: {output_filename}")
        print("--- Extracted JSON Preview ---")
        print(json.dumps(json.loads(json_output), indent=2))

    except Exception as e:
        print(f"⚠ An error occurred while parsing or saving the JSON: {e}")

# ---------- Entry point ----------

if __name__ == "__main__":
    # Make sure this filename is correct and the file is in the same folder,
    # or replace with full path like r'C:\Users\ANAGHA\Desktop\pod (2).pdf'
    PDF_FILE_PATH = "pod (2).pdf"

    extract_custom_invoice_data(PDF_FILE_PATH)
