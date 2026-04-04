# AI-Powered Document Analysis and Extraction API

An intelligent document processing API for the HCL hackathon problem statement.

The API accepts one Base64-encoded document at a time (`pdf`, `docx`, `image`, `png`, `jpg`, or `jpeg`), extracts text, runs AI/NLP analysis, and returns:
- concise summary,
- key entities (`names`, `dates`, `organizations`, `amounts`),
- sentiment (`Positive`, `Neutral`, `Negative`).

## Features
- Multi-format support: PDF, DOCX, image (OCR) including `png`, `jpg`, `jpeg`
- Automatic text extraction with parser/OCR pipeline
- AI-powered summarization
- Named entity extraction with rule-assisted filtering
- Sentiment analysis
- API key-based authentication
- Optional asynchronous processing with Celery + Redis

## Tech Stack
- Backend: Python, FastAPI, Uvicorn
- Async (optional): Celery, Redis
- OCR and extraction:
   - `pypdf` (PDF)
   - `python-docx` (DOCX)
   - `Pillow` + `pytesseract` (image OCR)
- AI/NLP:
   - Hugging Face Transformers (summarization)
   - spaCy + regex filters (entity extraction)
   - VADER Sentiment (sentiment classification)

## Project Structure
```text
AI-Powered-Document-Analysis-Extraction/
├── README.md
├── requirements.txt
├── .env.example
└── src/
      ├── main.py
      ├── config.py
      ├── models.py
      ├── celery_app.py
      ├── tasks.py
      └── services/
            ├── extractors.py
            ├── analyzers.py
            └── pipeline.py
```

## API Contract

### Endpoint
- `POST /api/document-analyze`

### Headers
- `x-api-key: <your-api-key>`
- `Content-Type: application/json`

### Request Body
```json
{
   "fileName": "sample.pdf",
   "fileType": "pdf",
   "fileBase64": "<base64-string>"
}
```

### Success Response
```json
{
   "status": "success",
   "fileName": "sample.pdf",
   "summary": "...",
   "entities": {
      "names": ["..."],
      "dates": ["..."],
      "organizations": ["..."],
      "amounts": ["..."]
   },
   "sentiment": "Neutral"
}
```

### Error Response
```json
{
   "status": "error",
   "fileName": "sample.pdf",
   "summary": "",
   "entities": {
      "names": [],
      "dates": [],
      "organizations": [],
      "amounts": []
   },
   "sentiment": "Neutral",
   "error": "<message>"
}
```

## Local Setup

### 1. Create virtual environment
```bash
python -m venv .venv
```

### 2. Activate environment
- Windows (PowerShell)
```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment
Copy `.env.example` to `.env` and set values:
- `API_KEY`
- `USE_CELERY` (optional)
- `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND` (if Celery enabled)
- `TESSERACT_CMD` (if Tesseract is not on PATH)

### 5. Install Tesseract OCR (required for image files)
- Windows: Install Tesseract and set `TESSERACT_CMD` in `.env`.

### 6. Install spaCy model (recommended)
```bash
python -m spacy download en_core_web_sm
```

### 7. Run API
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

### 8. Optional: run Celery worker
```bash
celery -A src.tasks worker --loglevel=info
```

## Quick Test (cURL)
```bash
curl -X POST "http://localhost:8000/api/document-analyze" \
   -H "accept: application/json" \
   -H "x-api-key: sk_track2_987654321" \
   -H "Content-Type: application/json" \
   -d '{
      "fileName": "image.png",
      "fileType": "png",
      "fileBase64": "<base64-string>"
   }'
```

## Internal Pipeline
1. Decode Base64 payload.
2. Route to extractor by `fileType`.
3. Normalize text.
4. Summarize content.
5. Extract entities (`names`, `dates`, `organizations`, `amounts`).
6. Classify sentiment.
7. Return structured JSON response.

## Submission Details

Update these values before final submission:
- Live deployed URL: `https://<your-domain>/api/document-analyze`
- API key for evaluator: `<your-api-key>`
- GitHub repository: `https://github.com/isha0601/AI-Powered-Document-Analysis-Extraction`

## Notes
- No hardcoded document-specific responses.
- Works on unseen files via parser + NLP pipeline.
- Image OCR quality depends on image clarity and Tesseract output.
