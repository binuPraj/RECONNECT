"""Read-only audit of every persisted stream WAV and segment manifest."""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path("uploads/streams")
OUT = Path("_tmp_all_audio_audit.json")


def read(path: Path):
    try:
        data, rate = sf.read(path, dtype="float32", always_2d=False)
        return data, rate, None
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


report = {"wav_files": [], "recordings": [], "summary": {}}
for wav in sorted(ROOT.rglob("*.wav")):
    data, rate, error = read(wav)
    report["wav_files"].append({
        "path": str(wav), "error": error,
        "sample_rate": rate,
        "samples": int(len(data)) if data is not None else None,
        "duration_sec": (len(data) / rate) if data is not None else None,
        "finite": bool(np.isfinite(data).all()) if data is not None else False,
        "non_silent": bool(np.any(np.abs(data) > 1e-6)) if data is not None else False,
    })

for manifest_path in sorted(ROOT.rglob("segments.json")):
    recording_dir = manifest_path.parent.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    segments = manifest.get("segments", [])
    item = {
        "recording": str(recording_dir), "segment_count": len(segments),
        "missing_segment_files": [], "duration_mismatches": [],
        "timeline_overlaps": [], "concat": [], "labels": dict(Counter(
            s.get("speaker_label", "MISSING") for s in segments)),
    }
    existing = []
    by_label = defaultdict(list)
    for segment in segments:
        path = Path(segment.get("audio_path", ""))
        if not path.exists():
            item["missing_segment_files"].append(str(path))
            continue
        data, rate, error = read(path)
        if error:
            item["missing_segment_files"].append(f"{path}: {error}")
            continue
        actual = len(data) / rate
        expected = float(segment["end_sec"]) - float(segment["start_sec"])
        if abs(actual - expected) > 0.025:
            item["duration_mismatches"].append({"segment": segment["segment_id"], "expected": expected, "actual": actual})
        existing.append(segment)
        by_label[segment.get("speaker_label", "MISSING")].append((segment, data, rate))
    ordered = sorted(existing, key=lambda s: (s["start_sec"], s["end_sec"]))
    for left, right in zip(ordered, ordered[1:]):
        if float(right["start_sec"]) < float(left["end_sec"]) - 0.001:
            item["timeline_overlaps"].append({"left": left["segment_id"], "right": right["segment_id"], "sec": float(left["end_sec"]) - float(right["start_sec"])})
    for label, clips in by_label.items():
        concat_path = recording_dir / "segments" / f"concat_{label}.wav"
        row = {"label": label, "exists": concat_path.exists(), "source_clips": len(clips)}
        if concat_path.exists():
            concat, rate, error = read(concat_path)
            expected_rate = clips[0][2]
            expected = np.concatenate([np.asarray(c[1]).reshape(-1) for c in clips])
            row.update({"error": error, "sample_rate": rate, "expected_sample_rate": expected_rate,
                        "samples": int(len(concat)) if concat is not None else None,
                        "expected_samples": int(len(expected))})
            if concat is not None and rate == expected_rate and len(concat) == len(expected):
                row["max_abs_diff"] = float(np.max(np.abs(np.asarray(concat).reshape(-1) - expected)))
                row["matches_source_sequence"] = row["max_abs_diff"] <= 2e-4
            else:
                row["matches_source_sequence"] = False
        item["concat"].append(row)
    report["recordings"].append(item)

all_wavs = report["wav_files"]
all_concat = [c for r in report["recordings"] for c in r["concat"]]
report["summary"] = {
    "wav_count": len(all_wavs),
    "unreadable_or_nonfinite_wavs": sum(bool(x["error"]) or not x["finite"] for x in all_wavs),
    "silent_wavs": sum(not x["non_silent"] for x in all_wavs),
    "recording_manifests": len(report["recordings"]),
    "missing_segment_files": sum(len(x["missing_segment_files"]) for x in report["recordings"]),
    "duration_mismatches": sum(len(x["duration_mismatches"]) for x in report["recordings"]),
    "timeline_overlaps": sum(len(x["timeline_overlaps"]) for x in report["recordings"]),
    "concat_expected": len(all_concat),
    "concat_missing": sum(not x["exists"] for x in all_concat),
    "concat_invalid": sum(x["exists"] and not x.get("matches_source_sequence", False) for x in all_concat),
}
OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report["summary"], indent=2))
