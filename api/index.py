"""
Vercel serverless entry point for SpecVLM FastAPI backend.
Exposes the ASGI app for Vercel's Python runtime.
"""
import sys
from pathlib import Path

# Ensure the project root is on the path so `import specvlm` works
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from specvlm.serving.api import app

# Vercel ASGI handler – Vercel picks up the `app` object automatically
