import os
from functools import lru_cache

from dotenv import load_dotenv


load_dotenv()


class Settings:
    api_key: str = os.getenv("API_KEY", "")

    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))

    use_celery: bool = os.getenv("USE_CELERY", "false").lower() == "true"
    celery_broker_url: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    celery_result_backend: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
    celery_task_timeout_seconds: int = int(os.getenv("CELERY_TASK_TIMEOUT_SECONDS", "120"))

    enable_transformers_summarization: bool = (
        os.getenv("ENABLE_TRANSFORMERS_SUMMARIZATION", "true").lower() == "true"
    )
    summarization_model: str = os.getenv("SUMMARIZATION_MODEL", "sshleifer/distilbart-cnn-12-6")
    max_summary_tokens: int = int(os.getenv("MAX_SUMMARY_TOKENS", "120"))

    tesseract_cmd: str = os.getenv("TESSERACT_CMD", "")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
