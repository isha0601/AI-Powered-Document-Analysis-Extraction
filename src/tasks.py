from src.celery_app import celery_app
from src.services.extractors import decode_base64_payload, extract_text_by_type
from src.services.pipeline import analyze_text


@celery_app.task(name="src.tasks.analyze_document_task")
def analyze_document_task(file_name: str, file_type: str, file_base64: str) -> dict:
    file_bytes = decode_base64_payload(file_base64)
    text = extract_text_by_type(file_bytes, file_type)
    if not text.strip():
        raise ValueError("No text extracted from document")

    analysis = analyze_text(text)
    return {
        "status": "success",
        "fileName": file_name,
        "summary": analysis["summary"],
        "entities": analysis["entities"],
        "sentiment": analysis["sentiment"],
    }
