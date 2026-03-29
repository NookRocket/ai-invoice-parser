import io
import pdfplumber


def extract_text_from_pdf(file_bytes: bytes):
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

