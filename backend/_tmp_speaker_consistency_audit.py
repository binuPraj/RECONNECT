"""Read-only per-segment embedding consistency audit for multi-label recordings."""
import json
import types
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

torchaudio.list_audio_backends = lambda: ["soundfile"]
torchaudio.io = types.SimpleNamespace(StreamReader=object)

from app.audio_process.diarization import _get_diarization_pipeline

ROOT = Path("uploads/streams")
MIN_SAMPLES = 1800  # Empirical finite/non-zero variance boundary, 2026-09-08.
OUT = Path("_tmp_speaker_consistency_audit.json")


def cosine_distance(left, right):
    return float(1 - np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))


pipeline = _get_diarization_pipeline()
results = []
for manifest_path in sorted(ROOT.rglob("segments.json")):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    segments = manifest.get("segments", [])
    labels = {segment.get("speaker_label", "MISSING") for segment in segments}
    measurable = sorted(label for label in labels if label != "SPEAKER_UNKNOWN")
    if len(measurable) < 2:
        continue
    groups = defaultdict(list)
    skipped = []
    for segment in segments:
        label = segment.get("speaker_label", "MISSING")
        if label not in measurable:
            continue
        audio, rate = sf.read(segment["audio_path"], dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if len(audio) < MIN_SAMPLES:
            skipped.append({"segment": segment["segment_id"], "reason": "below_minimum", "samples": len(audio)})
            continue
        embedding = np.asarray(pipeline._embedding(torch.from_numpy(audio).unsqueeze(0).unsqueeze(0))[0], dtype=np.float64)
        if not np.isfinite(embedding).all() or float(np.var(embedding)) <= 1e-8:
            skipped.append({"segment": segment["segment_id"], "reason": "invalid_embedding", "samples": len(audio)})
            continue
        groups[label].append(embedding)
    active = sorted(label for label, values in groups.items() if values)
    within, cross = [], []
    for label in active:
        values = groups[label]
        within += [cosine_distance(values[i], values[j]) for i in range(len(values)) for j in range(i + 1, len(values))]
    for index, left_label in enumerate(active):
        for right_label in active[index + 1:]:
            cross += [cosine_distance(left, right) for left in groups[left_label] for right in groups[right_label]]
    results.append({
        "recording": str(manifest_path.parent.parent),
        "valid_embeddings_per_label": {label: len(groups[label]) for label in active},
        "skipped": skipped,
        "within_mean": float(np.mean(within)) if within else None,
        "cross_mean": float(np.mean(cross)) if cross else None,
        "consistent": bool(within and cross and np.mean(within) < np.mean(cross)),
    })
OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(json.dumps(results, indent=2))
