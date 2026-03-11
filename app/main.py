import os
import base64
import json
import re
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx
from typing import Optional

app = FastAPI(title="AI Invoice Processing System", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

EXTRACTION_PROMPT = """You are an expert invoice data extraction system. Analyze this invoice image and extract ALL available information with high precision.

Return ONLY a valid JSON object (no markdown, no explanation) with this exact structure:
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

Extract every field you can find. Use null for fields not present."""


@app.get("/")
async def root():
    return {"message": "AI Invoice Processing System", "status": "running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/api/extract")
async def extract_invoice(file: UploadFile = File(...)):
    # Validate file type
    allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/gif", "image/webp", "application/pdf"]
    content_type = file.content_type or ""
    
    if not any(ct in content_type for ct in ["image/", "application/pdf"]):
        raise HTTPException(status_code=400, detail="Only image files (JPEG, PNG, GIF, WebP) and PDFs are supported")
    
    # Read file content
    file_content = await file.read()
    
    if len(file_content) > 20 * 1024 * 1024:  # 20MB limit
        raise HTTPException(status_code=400, detail="File size exceeds 20MB limit")
    
    # Encode to base64
    base64_data = base64.standard_b64encode(file_content).decode("utf-8")
    
    # Determine media type for API
    if "pdf" in content_type:
        media_type = "application/pdf"
        content_block = {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": base64_data
            }
        }
    else:
        # Normalize media type
        if "jpeg" in content_type or "jpg" in content_type:
            media_type = "image/jpeg"
        elif "png" in content_type:
            media_type = "image/png"
        elif "gif" in content_type:
            media_type = "image/gif"
        elif "webp" in content_type:
            media_type = "image/webp"
        else:
            media_type = "image/jpeg"
        
        content_block = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": base64_data
            }
        }
    
    # Call Anthropic API
    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 2000,
        "messages": [
            {
                "role": "user",
                "content": [
                    content_block,
                    {
                        "type": "text",
                        "text": EXTRACTION_PROMPT
                    }
                ]
            }
        ]
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            ANTHROPIC_API_URL,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
    
    if response.status_code != 200:
        error_detail = response.json() if response.content else "Unknown error"
        raise HTTPException(
            status_code=response.status_code,
            detail=f"AI API error: {error_detail}"
        )
    
    result = response.json()
    raw_text = result["content"][0]["text"]
    
    # Parse JSON from response
    try:
        # Try direct parse first
        extracted_data = json.loads(raw_text)
    except json.JSONDecodeError:
        # Try to find JSON block
        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if json_match:
            try:
                extracted_data = json.loads(json_match.group())
            except json.JSONDecodeError:
                extracted_data = {"raw_text": raw_text, "parse_error": "Could not parse structured data"}
        else:
            extracted_data = {"raw_text": raw_text, "parse_error": "No JSON found in response"}
    
    return JSONResponse(content={
        "success": True,
        "filename": file.filename,
        "file_size": len(file_content),
        "extracted_data": extracted_data
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
