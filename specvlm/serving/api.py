from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

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


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


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

