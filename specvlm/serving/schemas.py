from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SamplingParamsIn(BaseModel):
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)


class GenerateIn(BaseModel):
    prompt: str
    sampling_params: SamplingParamsIn = Field(default_factory=SamplingParamsIn)
    image_base64: str | None = None
    image_url: str | None = None
    request_id: str | None = None


class MetricsOut(BaseModel):
    accepted_draft_tokens: int = 0
    proposed_draft_tokens: int = 0
    acceptance_rate: float = 0.0
    tokens_per_second: float = 0.0
    t_ms: int = 0


class WsTokenOut(BaseModel):
    type: Literal["token"] = "token"
    delta: str
    text: str
    metrics: MetricsOut


class WsErrorOut(BaseModel):
    type: Literal["error"] = "error"
    message: str
    details: Any | None = None


class WsDoneOut(BaseModel):
    type: Literal["done"] = "done"

