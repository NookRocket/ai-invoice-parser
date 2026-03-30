# 🧾 AI Invoice Processing System

An intelligent invoice extraction system powered by **FastAPI** and **Gemini AI**.

## Tech Stack
- **Backend**: Python · FastAPI · Uvicorn
- **AI**: Gemini

---

## Features
- 📤 Upload invoices (JPEG, PNG, WebP, GIF, PDF)
- 🤖 AI-powered field extraction via Gemini Generative AI
- 🧾 Extracts: Invoice #, dates, vendor, bill-to, line items, totals, tax, payment info

---

## Setup

### Prerequisites
- Python 3.10+
- Gemini API Key

### Backend

```bash
pip install -r requirements.txt

# Set your API key
export GEMINI_API_KEY="..."

uvicorn app.main:app --reload
# Runs at http://localhost:8000

# Swagger
http://127.0.0.1:8000/docs
```


---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/parse-invoice` | Extract invoice fields |

### Extract Invoice

```bash
curl -X POST http://localhost:8000/parse-invoic \
  -F "file=@invoice.jpg"
```

**Response:**
```json
{
  "success": true,
  "filename": "invoice.jpg",
  "file_size": 204800,
  "extracted_data": {
    "invoice_number": "INV-2024-001",
    "invoice_date": "2024-01-15",
    "due_date": "2024-02-15",
    "vendor": {
      "name": "Acme Corp",
      "address": "123 Main St, NY",
      "email": "billing@acme.com",
      "phone": "+1-555-0100",
      "tax_id": "12-3456789"
    },
    "bill_to": {
      "name": "Client Inc.",
      "address": "456 Oak Ave, CA",
      "email": null
    },
    "line_items": [
      {
        "description": "Web Development Services",
        "quantity": 40,
        "unit_price": 150.00,
        "total": 6000.00
      }
    ],
    "subtotal": 6000.00,
    "tax_rate": 8.5,
    "tax_amount": 510.00,
    "discount": null,
    "total_amount": 6510.00,
    "currency": "USD",
    "payment_terms": "Net 30",
    "payment_method": "Bank Transfer",
    "notes": null,
    "confidence_score": 0.95
  }
}
```

---

## Project Structure

```
invoice-parser/
├── app/
│   ├── main.py       # FastAPI app + extraction logic
│   ├── extractor.py
│   ├── llm_parser.py
│   ├── schemas.py
├── tests/            # Pytest
└── requirements.txt
└── README.md
```
