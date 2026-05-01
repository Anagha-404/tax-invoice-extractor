# Invoice JSON Extractor

An AI-powered invoice parser that extracts structured JSON from PDF/image invoices using OCR + Gemini API.

##  Features

* Extracts GSTIN, invoice date, and line items
* Handles CGST/SGST & IGST logic
* OCR preprocessing using OpenCV
* Structured validation using Pydantic
* Clean web interface using Flask

## 🛠Tech Stack

* Python (Flask)
* OpenCV (OCR preprocessing)
* Google Gemini API
* Pydantic (schema validation)

## Installation

```bash
pip install -r requirements.txt
```

## Run the app

```bash
python app.py
```

## Environment Variables

Create a `.env` file:

```
GOOGLE_API_KEY=your_api_key_here
```

## Demo

Upload invoice → Extract structured JSON output

## Future Improvements

* Confidence scoring
* Fraud detection
* Deployment (cloud)
