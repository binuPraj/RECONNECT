"""Question audio → transcript using the existing Whisper helper, read-only."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path


def transcribe_question_audio(data: bytes, filename: str = "question.wav") -> str:
    temp_dir = Path(tempfile.mkdtemp(prefix="retrival_q_"))
    suffix = Path(filename).suffix or ".wav"
    path = temp_dir / f"question{suffix}"
    try:
        path.write_bytes(data)
        from app.audio_process.transcribe import transcribe_wav
        return transcribe_wav(path) or ""
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
