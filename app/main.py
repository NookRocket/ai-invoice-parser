import json
import base64
from fastapi import FastAPI, UploadFile, HTTPException
from app.extractor import extract_text_from_pdf
from app.llm_parser import parse_invoice_from_image, parse_invoice_from_text

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "AI Invoice Processing System", "status": "running"}

@app.post("/parse-invoice")
async def parse_invoice(file: UploadFile):

    # Validate file type
    content_type = file.content_type or ""
    
    if not any(ct in content_type for ct in ["image/", "application/pdf"]):
        raise HTTPException(status_code=400, detail="Only image files (JPEG, PNG, GIF, WebP) and PDFs are supported")

    contents = await file.read()

    if len(contents) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=400, detail="File size exceeds 10MB limit")

    if "pdf" in content_type:
        text = extract_text_from_pdf(contents)
        invoice = parse_invoice_from_text(text)
    else:
        invoice = parse_invoice_from_image(contents)   

    return invoice

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
