"""Run from the backend folder: python -m retrival"""
from __future__ import annotations

from retrival.config import RETRIVAL_PORT
from retrival.server import _load_env

if __name__ == "__main__":
    import uvicorn
    _load_env()
    uvicorn.run("retrival.server:app", host="0.0.0.0", port=RETRIVAL_PORT, reload=False)
