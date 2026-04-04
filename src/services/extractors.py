import base64
import io

import pytesseract
from docx import Document
from PIL import Image
from pypdf import PdfReader

from src.config import get_settings


def decode_base64_payload(base64_payload: str) -> bytes:
    try:
        return base64.b64decode(base64_payload, validate=True)
    except Exception as exc:
        raise ValueError("Invalid Base64 payload") from exc


def extract_text_from_pdf(file_bytes: bytes) -> str:
    with io.BytesIO(file_bytes) as stream:
        reader = PdfReader(stream)
        pages: list[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)
    return "\n\n".join(page.strip() for page in pages if page.strip())


def extract_text_from_docx(file_bytes: bytes) -> str:
    with io.BytesIO(file_bytes) as stream:
        doc = Document(stream)
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n".join(paragraphs)


def extract_text_from_image(file_bytes: bytes) -> str:
    settings = get_settings()
    if settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

    with io.BytesIO(file_bytes) as stream:
        image = Image.open(stream)
        text = pytesseract.image_to_string(image)
    return text.strip()


def extract_text_by_type(file_bytes: bytes, file_type: str) -> str:
    normalized_type = file_type.lower().strip()
    if normalized_type == "pdf":
        return extract_text_from_pdf(file_bytes)
    if normalized_type == "docx":
        return extract_text_from_docx(file_bytes)
    if normalized_type in {"image", "png", "jpg", "jpeg"}:
        return extract_text_from_image(file_bytes)
    raise ValueError("Unsupported fileType. Use one of: pdf, docx, image, png, jpg, jpeg")
