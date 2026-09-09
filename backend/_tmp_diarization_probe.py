"""Temporary, non-production diarization diagnostic. Delete after use."""
import json
import os
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from scipy.cluster.hierarchy import linkage

# Compatibility shims required by the installed SpeechBrain/Torchaudio pair.
torchaudio.list_audio_backends = lambda: ["soundfile"]
torchaudio.io = types.SimpleNamespace(StreamReader=object)

from pyannote.audio.pipelines.clustering import VBxClustering
from app.audio_process.diarization import _get_diarization_pipeline

RECORDING = Path("uploads/streams/session_006/recording_0003")
OUTPUT = Path("_tmp_diarization_probe_output.json")


def cosine_distance(a, b):
    return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def inspect(name):
    audio, sample_rate = sf.read(RECORDING / name, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
    pipeline = _get_diarization_pipeline()
    trace = {}
    original_call = VBxClustering.__call__

    def traced_call(self, embeddings, segmentations=None, **kwargs):
        train, chunk_idx, speaker_idx = self.filter_embeddings(embeddings, segmentations)
        normalized = train / np.linalg.norm(train, axis=1, keepdims=True)
        trace["training_embedding_count"] = int(len(train))
        trace["training_embeddings"] = train.tolist()
        trace["training_chunk_indices"] = chunk_idx.tolist()
        trace["training_local_speaker_indices"] = speaker_idx.tolist()
        trace["ahc_merge_distances"] = linkage(normalized, method="centroid", metric="euclidean")[:, 2].tolist()
        return original_call(self, embeddings, segmentations=segmentations, **kwargs)

    VBxClustering.__call__ = traced_call
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            output = pipeline(
                {"waveform": waveform, "sample_rate": sample_rate, "uri": name},
                min_speakers=2,
                max_speakers=6,
            )
        trace["pipeline_warnings"] = [str(item.message) for item in caught]
    finally:
        VBxClustering.__call__ = original_call

    turns = [
        {"start": float(turn.start), "end": float(turn.end), "speaker_label": str(speaker)}
        for turn, speaker in output.speaker_diarization
    ]
    turn_embeddings, turn_warnings = [], []
    for index, turn in enumerate(turns):
        start, end = int(turn["start"] * sample_rate), int(turn["end"] * sample_rate)
        clip = waveform[:, start:end].unsqueeze(0)
        if clip.shape[-1] < 400:
            turn_embeddings.append(None)
            turn_warnings.append({"turn": index, "warnings": ["embedding skipped: fewer than 400 samples"]})
            continue
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            embedding = pipeline._embedding(clip)[0]
        turn_embeddings.append(np.asarray(embedding, dtype=float).tolist())
        turn_warnings.append({"turn": index, "warnings": [str(item.message) for item in caught]})

    matrix = [
        [None if a is None or b is None else cosine_distance(np.asarray(a), np.asarray(b)) for b in turn_embeddings]
        for a in turn_embeddings
    ]
    return {
        "sample_rate": sample_rate,
        "duration_sec": len(audio) / sample_rate,
        "turns": turns,
        "turn_embeddings": turn_embeddings,
        "turn_embedding_warnings": turn_warnings,
        "turn_embedding_cosine_distance_matrix": matrix,
        "clustering": trace,
    }


if __name__ == "__main__":
    result = {"cleaned.wav": inspect("cleaned.wav"), "raw.wav": inspect("raw.wav")}
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(OUTPUT.resolve(), flush=True)
