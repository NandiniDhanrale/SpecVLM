# Backend (FastAPI + WebSocket)

## Run (Mock engine)

```powershell
cd D:\specvlm
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .
uvicorn specvlm.serving.api:app --reload --port 8000
```

## Run (vLLM engine)

```powershell
pip install -e ".[vllm]"
$env:SPECVLM_ENGINE="vllm"
$env:SPECVLM_MODEL="path-or-hf-model-id"
# Optional: $env:SPECVLM_SPEC_CONFIG="spec_config.json"
uvicorn specvlm.serving.api:app --reload --port 8000
```

