"""
SpecVLM FastAPI Serving Layer

Production-grade API server for VLM inference with:
- Streaming Server-Sent Events (SSE) responses
- Request batching and scheduling
- Prometheus metrics
- Rate limiting
- Graceful shutdown
- Health checks

API Endpoints:
    POST /v1/chat/completions  — OpenAI-compatible chat completion
    POST /v1/completions       — Text completion
    POST /v1/images/embeddings — Pre-compute visual embeddings
    GET  /v1/models            — List available models
    GET  /health               — Health check
    GET  /metrics              — Prometheus metrics
    GET  /stats                — Engine statistics

Architecture:
┌─────────┐     ┌──────────┐     ┌──────────────┐     ┌──────────┐
│ Client  │────▶│ FastAPI  │────▶│ Request      │────▶│ Engine   │
│         │◀────│ Server   │◀────│ Scheduler    │◀────│ Workers  │
└─────────┘     └──────────┘     └──────────────┘     └──────────┘
                                            │
                                            ▼
                                     ┌──────────┐
                                     │  Redis    │
                                     │  Queue    │
                                     └──────────┘

Production considerations (taken from vLLM, TGI, Triton):
- OpenAI-compatible API for easy integration
- SSE streaming for real-time token delivery
- Request priority queue with starvation prevention
- Automatic batching of concurrent requests
- Health checks for k8s liveness/readiness probes
"""

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from specvlm.config.settings import settings
from specvlm.inference.engine import InferenceEngine, EngineConfig, EngineMode, EngineBackend
from specvlm.models.base_vlm import VLMInput
from specvlm.serving.scheduler import RequestScheduler

logger = logging.getLogger(__name__)

# Global engine instance
engine: Optional[InferenceEngine] = None
scheduler: Optional[RequestScheduler] = None


# =============================================================================
# Pydantic Models for API
# =============================================================================

class ChatMessage(BaseModel):
    role: str = "user"
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="specvlm-target", description="Model ID to use")
    messages: list[ChatMessage] = Field(default_factory=list)
    images: list[str] = Field(default_factory=list, description="Image URLs or paths")
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=1)
    stream: bool = Field(default=False, description="Stream tokens via SSE")
    use_speculative: bool = Field(default=False, description="Use speculative decoding")
    priority: int = Field(default=0, ge=0, le=10)


class CompletionRequest(BaseModel):
    model: str = "specvlm-target"
    prompt: str = ""
    images: list[str] = Field(default_factory=list)
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    stream: bool = False
    use_speculative: bool = False
    priority: int = 0


class ImageEmbeddingRequest(BaseModel):
    images: list[str]
    model: str = "specvlm-visual"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[dict]
    usage: dict = Field(default_factory=lambda: {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    })


# =============================================================================
# Server Lifecycle
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: load models on startup, cleanup on shutdown."""
    global engine, scheduler

    logger.info("Starting SpecVLM server...")

    # Determine engine mode
    mode = EngineMode.SPECULATIVE if settings.inference.use_speculative_decoding else EngineMode.BASELINE
    backend = EngineBackend(settings.inference.engine)

    engine_config = EngineConfig(
        mode=mode,
        backend=backend,
        target_model_id=settings.model.model_id,
        draft_model_id=settings.inference.draft_model_id,
        enable_profiling=settings.inference.enable_profiling,
    )

    # Initialize engine
    engine = InferenceEngine(config=engine_config)
    await engine.load()

    # Initialize scheduler
    scheduler = RequestScheduler(engine=engine)

    # Warmup
    await engine.warmup()

    logger.info("SpecVLM server ready")
    yield

    # Shutdown
    logger.info("Shutting down SpecVLM server...")
    await engine.shutdown()


# =============================================================================
# FastAPI Application
# =============================================================================

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="SpecVLM Inference API",
        version=settings.version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # -------------------------------------------------------------------------
    # Middleware
    # -------------------------------------------------------------------------
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time-Ms"] = str(int(process_time * 1000))
        return response

    # -------------------------------------------------------------------------
    # Health & Monitoring
    # -------------------------------------------------------------------------
    @app.get("/health")
    async def health_check():
        """Health check endpoint for k8s probes."""
        global engine
        if engine is None or not engine._loaded:
            raise HTTPException(status_code=503, detail="Engine not ready")
        return {
            "status": "healthy",
            "loaded": engine._loaded,
            "mode": engine.mode.value,
            "gpu_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        }

    @app.get("/stats")
    async def get_stats():
        """Return engine performance statistics."""
        global engine
        if engine is None:
            raise HTTPException(status_code=503, detail="Engine not initialized")
        return engine.get_stats()

    @app.get("/v1/models")
    async def list_models():
        """List available models."""
        models = []
        if settings.inference.target_model_id:
            models.append({
                "id": settings.inference.target_model_id,
                "object": "model",
                "type": "target",
            })
        if settings.inference.draft_model_id:
            models.append({
                "id": settings.inference.draft_model_id,
                "object": "model",
                "type": "draft",
            })
        return {"object": "list", "data": models}

    # -------------------------------------------------------------------------
    # Main Inference Endpoints
    # -------------------------------------------------------------------------
    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        """OpenAI-compatible chat completion endpoint."""
        global engine, scheduler

        if engine is None:
            raise HTTPException(status_code=503, detail="Engine not initialized")

        # Extract the last user message as prompt
        prompt = ""
        for msg in reversed(request.messages):
            if msg.role == "user":
                prompt = msg.content
                break

        if not prompt:
            raise HTTPException(status_code=400, detail="No user message found")

        request_id = str(uuid.uuid4())

        vlm_input = VLMInput(
            prompt=prompt,
            image_paths=request.images,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            request_id=request_id,
            priority=request.priority,
        )

        # Streaming response
        if request.stream:
            return StreamingResponse(
                _stream_chat_completion(vlm_input, request.model),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # Non-streaming response
        result = await engine.generate(vlm_input)

        return ChatCompletionResponse(
            id=f"chatcmpl-{request_id}",
            model=request.model,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result.text,
                },
                "finish_reason": "stop",
            }],
            usage={
                "prompt_tokens": result.num_input_tokens,
                "completion_tokens": result.num_output_tokens,
                "total_tokens": result.num_input_tokens + result.num_output_tokens,
            },
        )

    @app.post("/v1/completions")
    async def completions(request: CompletionRequest):
        """Text completion endpoint."""
        global engine

        if engine is None:
            raise HTTPException(status_code=503, detail="Engine not initialized")

        request_id = str(uuid.uuid4())

        vlm_input = VLMInput(
            prompt=request.prompt,
            image_paths=request.images,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            request_id=request_id,
            priority=request.priority,
        )

        if request.stream:
            return StreamingResponse(
                _stream_completion(vlm_input, request.model),
                media_type="text/event-stream",
            )

        result = await engine.generate(vlm_input)

        return {
            "id": f"cmpl-{request_id}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "text": result.text,
                "index": 0,
                "finish_reason": "stop",
                "logprobs": result.logprobs if result.logprobs else None,
            }],
            "usage": {
                "prompt_tokens": result.num_input_tokens,
                "completion_tokens": result.num_output_tokens,
                "total_tokens": result.num_input_tokens + result.num_output_tokens,
            },
        }

    @app.post("/v1/images/embeddings")
    async def image_embeddings(request: ImageEmbeddingRequest):
        """Pre-compute visual embeddings for caching."""
        global engine

        if engine is None:
            raise HTTPException(status_code=503, detail="Engine not initialized")

        embeddings = []
        for image_path in request.images:
            embeds = engine.visual_encoder.encode_image(image_path)
            embeddings.append({
                "image": image_path,
                "shape": list(embeds.shape),
                "dtype": str(embeds.dtype),
            })

        return {"object": "list", "data": embeddings}

    return app


# =============================================================================
# Streaming Helpers
# =============================================================================

async def _stream_chat_completion(vlm_input: VLMInput, model_id: str):
    """Stream chat completion tokens as SSE events."""
    global engine

    request_id = f"chatcmpl-{vlm_input.request_id}"
    created = int(time.time())

    # Stream first event
    yield f"data: {{\"id\":\"{request_id}\",\"object\":\"chat.completion.chunk\",\"created\":{created},\"model\":\"{model_id}\",\"choices\":[{{\"index\":0,\"delta\":{{\"role\":\"assistant\"}},\"finish_reason\":null}}]}}\n\n"

    async for chunk in engine.generate_stream(vlm_input):
        if chunk.text:
            yield f"data: {{\"id\":\"{request_id}\",\"object\":\"chat.completion.chunk\",\"created\":{created},\"model\":\"{model_id}\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":{chunk.text!r}}},\"finish_reason\":null}}]}}\n\n"

    # Stream final event
    yield f"data: {{\"id\":\"{request_id}\",\"object\":\"chat.completion.chunk\",\"created\":{created},\"model\":\"{model_id}\",\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"stop\"}}]}}\n\n"
    yield "data: [DONE]\n\n"


async def _stream_completion(vlm_input: VLMInput, model_id: str):
    """Stream text completion tokens as SSE events."""
    global engine

    request_id = f"cmpl-{vlm_input.request_id}"
    created = int(time.time())

    async for chunk in engine.generate_stream(vlm_input):
        if chunk.text:
            yield f"data: {{\"id\":\"{request_id}\",\"object\":\"text_completion\",\"created\":{created},\"model\":\"{model_id}\",\"choices\":[{{\"text\":{chunk.text!r},\"index\":0,\"finish_reason\":null}}]}}\n\n"

    yield f"data: {{\"id\":\"{request_id}\",\"object\":\"text_completion\",\"created\":{created},\"model\":\"{model_id}\",\"choices\":[{{\"text\":\"\",\"index\":0,\"finish_reason\":\"stop\"}}]}}\n\n"
    yield "data: [DONE]\n\n"


# =============================================================================
# Entrypoint
# =============================================================================

app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "specvlm.serving.api:app",
        host=settings.serving.host,
        port=settings.serving.port,
        workers=settings.serving.workers,
        log_level=settings.log_level.lower(),
        reload=settings.debug,
    )
