from google import genai
from google.genai.types import Part
from app.schemas import Invoice

client = genai.Client()

PROMPT = """
You are an expert invoice data extraction system. Extract structured invoice data.

Return ONLY valid JSON with this schema:

{
  "invoice_number": "string or null",
  "invoice_date": "string or null",
  "due_date": "string or null",
  "vendor": {
    "name": "string or null",
    "address": "string or null",
    "email": "string or null",
    "phone": "string or null",
    "tax_id": "string or null"
  },
  "bill_to": {
    "name": "string or null",
    "address": "string or null",
    "email": "string or null"
  },
  "line_items": [
    {
      "description": "string",
      "quantity": "number or null",
      "unit_price": "number or null",
      "total": "number or null"
    }
  ],
  "subtotal": "number or null",
  "tax_rate": "number or null",
  "tax_amount": "number or null",
  "discount": "number or null",
  "total_amount": "number or null",
  "currency": "string or null",
  "payment_terms": "string or null",
  "payment_method": "string or null",
  "notes": "string or null",
  "confidence_score": "number between 0 and 1"
}

Rules:
- Use null when value not present
- Extract line_items table carefully
- currency should be ISO code if possible
- confidence_score should reflect extraction reliability
- "payment_method" is null if there are more than one method
- "tax_id" contains 13 digits
"""


def parse_invoice_from_image(file_bytes) -> Invoice:
    prompt = f"""{PROMPT}
- Analyze this invoice image and extract ALL available information with high precision.
"""
    response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[
        prompt,
        Part.from_bytes(
            data=file_bytes,
            mime_type="image/jpeg",
        ),
    ],
            config={
        "response_mime_type": "application/json",
        "response_json_schema": Invoice.model_json_schema(),
    })


    return Invoice.model_validate_json(response.text)

def parse_invoice_from_text(text: str) -> Invoice:

    prompt = f"""{PROMPT}

Invoice text:
{text}
"""
    response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config={
        "response_mime_type": "application/json",
        "response_json_schema": Invoice.model_json_schema(),
    })


    return Invoice.model_validate_json(response.text)
