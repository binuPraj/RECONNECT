"""Temporary empirical validity sweep for pyannote turn embeddings."""
import types
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

torchaudio.list_audio_backends = lambda: ["soundfile"]
torchaudio.io = types.SimpleNamespace(StreamReader=object)

from app.audio_process.diarization import _get_diarization_pipeline


audio, sample_rate = sf.read(
    Path("uploads/streams/session_003/recording_0001/raw.wav"),
    dtype="float32",
    always_2d=False,
)
if audio.ndim == 2:
    audio = audio.mean(axis=1)

# Begin inside a long speech turn so every tested prefix is speech, not silence.
start = int(8.367 * sample_rate)
pipeline = _get_diarization_pipeline()
for samples in (1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3200):
    clip = torch.from_numpy(audio[start:start + samples]).unsqueeze(0).unsqueeze(0)
    embedding = np.asarray(pipeline._embedding(clip)[0], dtype=np.float64).reshape(-1)
    print(
        f"samples={samples} sec={samples / sample_rate:.4f} "
        f"finite={np.isfinite(embedding).all()} "
        f"variance={np.nanvar(embedding):.10f} norm={np.linalg.norm(embedding):.10f}",
        flush=True,
    )
