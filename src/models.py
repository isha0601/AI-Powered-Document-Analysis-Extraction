from typing import Literal

from pydantic import BaseModel, Field, field_validator


class DocumentAnalyzeRequest(BaseModel):
    fileName: str = Field(min_length=1)
    fileType: Literal["pdf", "docx", "image", "png", "jpg", "jpeg"]
    fileBase64: str = Field(min_length=1)

    @field_validator("fileType")
    @classmethod
    def normalize_file_type(cls, value: str) -> str:
        normalized = value.lower().strip()
        if normalized in {"png", "jpg", "jpeg"}:
            return "image"
        return normalized


class Entities(BaseModel):
    names: list[str]
    dates: list[str]
    organizations: list[str]
    amounts: list[str]


class DocumentAnalyzeResponse(BaseModel):
    status: Literal["success", "error"]
    fileName: str
    summary: str
    entities: Entities
    sentiment: Literal["Positive", "Neutral", "Negative"]
