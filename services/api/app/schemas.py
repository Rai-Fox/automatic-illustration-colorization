from typing import Any

from pydantic import BaseModel, Field


class ColorizeResponse(BaseModel):
    image_base64: str
    model: str
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelInfo(BaseModel):
    model_id: str
    enabled: bool
    requires_reference: bool
    supports_multiple_references: bool
    supports_cpu: bool


class JobCreateResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    model_id: str
    chat_id: int | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    input_path: str | None = None
    reference_paths: list[str] = Field(default_factory=list)
    result_path: str | None = None
    seed: int | None = None
    options: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
