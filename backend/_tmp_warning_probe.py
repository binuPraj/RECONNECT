"""Temporary warning-as-error location probe; delete after use."""
import json
import os
import types
import warnings

import numpy as np
import soundfile as sf
import torch
import torchaudio

os.environ["LOKY_MAX_CPU_COUNT"] = "4"

torchaudio.list_audio_backends = lambda: ["soundfile"]
torchaudio.io = types.SimpleNamespace(StreamReader=object)

from app.audio_process.diarization import _get_diarization_pipeline

audio, rate = sf.read("uploads/streams/session_006/recording_0003/cleaned.wav", dtype="float32")
pipeline = _get_diarization_pipeline()
records = []
embedding_type = type(pipeline._embedding)
original_call = embedding_type.__call__

def traced_call(self, waveforms, masks=None):
    records.append({
        "waveform_shape": list(waveforms.shape),
        "mask_shape": list(masks.shape) if masks is not None else None,
        "mask_active_frames": masks.sum(dim=1).cpu().tolist() if masks is not None else None,
    })
    return original_call(self, waveforms, masks=masks)

embedding_type.__call__ = traced_call
try:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pipeline({"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": rate, "uri": "cleaned.wav"}, min_speakers=2, max_speakers=6)
    result = {"completed": True, "records": records}
except Warning as exc:
    result = {"completed": False, "error_type": type(exc).__name__, "error": str(exc), "records": records}
finally:
    embedding_type.__call__ = original_call

open("_tmp_warning_probe_output.json", "w", encoding="utf-8").write(json.dumps(result, indent=2))
