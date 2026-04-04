import os

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse

from src.config import get_settings
from src.models import DocumentAnalyzeRequest, DocumentAnalyzeResponse, Entities
from src.services.extractors import decode_base64_payload, extract_text_by_type
from src.services.pipeline import analyze_text
from src.tasks import analyze_document_task


app = FastAPI(title="Data Extraction API", version="1.0.0")


def validate_api_key(x_api_key: str = Header(default="")) -> None:
    settings = get_settings()
    if not settings.api_key:
        raise HTTPException(status_code=500, detail="Server API key is not configured")
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.post("/api/document-analyze", response_model=DocumentAnalyzeResponse)
def document_analyze(
    payload: DocumentAnalyzeRequest,
    _: None = Depends(validate_api_key),
):
    settings = get_settings()
    try:
        if settings.use_celery:
            async_result = analyze_document_task.delay(
                payload.fileName, payload.fileType, payload.fileBase64
            )
            result = async_result.get(timeout=settings.celery_task_timeout_seconds)
            return DocumentAnalyzeResponse(
                status="success",
                fileName=result["fileName"],
                summary=result["summary"],
                entities=Entities(**result["entities"]),
                sentiment=result["sentiment"],
            )

        file_bytes = decode_base64_payload(payload.fileBase64)
        text = extract_text_by_type(file_bytes, payload.fileType)
        if not text.strip():
            raise ValueError("No text extracted from document")

        analysis = analyze_text(text)

        return DocumentAnalyzeResponse(
            status="success",
            fileName=payload.fileName,
            summary=analysis["summary"],
            entities=Entities(**analysis["entities"]),
            sentiment=analysis["sentiment"],
        )
    except HTTPException:
        raise
    except Exception as exc:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "fileName": payload.fileName,
                "summary": "",
                "entities": {
                    "names": [],
                    "dates": [],
                    "organizations": [],
                    "amounts": [],
                },
                "sentiment": "Neutral",
                "error": str(exc),
            },
        )


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=os.getenv("ENV", "dev") == "dev",
    )
