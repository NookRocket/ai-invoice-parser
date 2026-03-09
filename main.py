from fastapi import FastAPI, UploadFile, File
from openai import OpenAI
import fitz
import os
import json

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_text(file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""

    for page in doc:
        text += page.get_text()

    return text


@app.post("/parse-invoice")
async def parse_invoice(file: UploadFile = File(...)):

    contents = await file.read()

    text = extract_text(contents)

    prompt = f"""
Extract invoice information from the text.

Return JSON with fields:
invoice_number
vendor
date
total
currency

Text:
{text[:4000]}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"user","content":prompt}
        ]
    )

    return {
        "result": response.choices[0].message.content
    }