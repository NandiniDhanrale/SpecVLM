from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI, WebSocket, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .engine import get_engine
from .schemas import GenerateIn, WsDoneOut, WsErrorOut, WsTokenOut


app = FastAPI(title="SpecVLM Streaming API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "service": "SpecVLM",
        "version": "0.1.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "generate_http": "/generate",
            "generate_ws": "/ws/generate",
            "generate_sse": "/api/generate/stream",
            "docs": "/docs",
        },
    }


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/generate")
async def generate_http(req: GenerateIn) -> JSONResponse:
    """HTTP streaming endpoint — returns the full generation result."""
    engine = get_engine()
    full_text = ""
    total_accepted = 0
    total_proposed = 0
    async for ev in engine.stream(req):
        full_text = ev.text
        total_accepted = ev.metrics.accepted_draft_tokens
        total_proposed = ev.metrics.proposed_draft_tokens

    return JSONResponse({
        "text": full_text,
        "metrics": {
            "accepted_draft_tokens": total_accepted,
            "proposed_draft_tokens": total_proposed,
            "acceptance_rate": total_accepted / total_proposed if total_proposed else 0.0,
        },
    })


@app.get("/api/generate/stream")
async def generate_stream_sse(prompt: str = "", max_tokens: int = 500, temperature: float = 0.8, top_p: float = 0.95):
    """SSE streaming endpoint for Vercel (Server-Sent Events)."""
    req = GenerateIn(
        prompt=prompt,
        sampling_params={"max_tokens": max_tokens, "temperature": temperature, "top_p": top_p},
    )
    engine = get_engine()

    async def event_stream():
        async for ev in engine.stream(req):
            data = json.dumps({
                "type": "token",
                "delta": ev.delta,
                "text": ev.text,
                "metrics": ev.metrics.model_dump(),
            })
            yield f"data: {data}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.websocket("/ws/generate")
async def websocket_generate(websocket: WebSocket) -> None:
    await websocket.accept()
    engine = None
    try:
        raw = await websocket.receive_text()
        payload: Any = json.loads(raw)
        req = GenerateIn.model_validate(payload)
        engine = get_engine()

        async for ev in engine.stream(req):
            msg = WsTokenOut(delta=ev.delta, text=ev.text, metrics=ev.metrics)
            await websocket.send_text(msg.model_dump_json())

        await websocket.send_text(WsDoneOut().model_dump_json())
    except Exception as e:
        await websocket.send_text(
            WsErrorOut(message=str(e), details={"engine": type(engine).__name__ if engine else None}).model_dump_json()
        )

