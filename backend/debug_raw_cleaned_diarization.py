"""Temporary raw/cleaned live diarization and clustering diagnostic."""

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from scipy.cluster.hierarchy import fcluster, linkage

if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]
if not hasattr(torchaudio, "io"):
    class DummyIO:
        class StreamReader:
            pass
    torchaudio.io = DummyIO()

from pyannote.audio.pipelines.clustering import VBxClustering
from app.audio_process.diarization import _get_diarization_pipeline


RECORDING = Path("uploads/streams/session_006/recording_0003")
pipeline = _get_diarization_pipeline()
original_call = VBxClustering.__call__


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(1.0 - np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))


def run(name: str):
    audio, sample_rate = sf.read(RECORDING / name, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
    trace = {}

    def traced_call(self, embeddings, segmentations=None, **kwargs):
        train, chunk_idx, speaker_idx = self.filter_embeddings(embeddings, segmentations)
        norm = train / np.linalg.norm(train, axis=1, keepdims=True)
        trace["train"] = train.copy()
        trace["chunk_idx"] = chunk_idx.copy()
        trace["speaker_idx"] = speaker_idx.copy()
        trace["linkage"] = linkage(norm, method="centroid", metric="euclidean")
        trace["ahc"] = fcluster(trace["linkage"], self.threshold, criterion="distance") - 1
        return original_call(self, embeddings, segmentations=segmentations, **kwargs)

    VBxClustering.__call__ = traced_call
    try:
        output = pipeline(
            {"waveform": waveform, "sample_rate": sample_rate, "uri": name},
            min_speakers=2,
            max_speakers=6,
        )
    finally:
        VBxClustering.__call__ = original_call

    turns = [(float(turn.start), float(turn.end), speaker) for turn, speaker in output.speaker_diarization]
    print(f"{name.upper()}_TURNS")
    for index, (start, end, speaker) in enumerate(turns):
        print(f"turn={index} ({speaker}, {start:.3f}, {end:.3f}, duration={end-start:.3f})")

    for block_name, start, end in (("BLOCK_0_15", 0.0, 15.0), ("BLOCK_15_24", 15.0, 24.0)):
        durations = [turn_end - turn_start for turn_start, turn_end, _ in turns if turn_start < end and turn_end > start]
        print(f"{name.upper()}_{block_name}_DURATIONS", " ".join(f"{value:.3f}" for value in durations))
        print(f"{name.upper()}_{block_name}_MEAN_MEDIAN {np.mean(durations):.6f} {np.median(durations):.6f}")

    print(f"{name.upper()}_AHC_MERGE_DISTANCES")
    print(" ".join(f"{distance:.6f}" for distance in trace["linkage"][:, 2]))

    turn_embeddings = []
    for start, end, _ in turns:
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        clip = waveform[:, start_sample:end_sample].unsqueeze(0)
        if clip.shape[-1] < 400:
            turn_embeddings.append(None)
        else:
            turn_embeddings.append(pipeline._embedding(clip)[0])
    print(f"{name.upper()}_TURN_EMBEDDING_PAIR_DISTANCES")
    # Transcript-assumed A/B/A front exchange: turn 0 vs 1 and turn 1 vs 2 differ; 0 vs 2 is same.
    for left, right, relation in ((0, 1, "assumed_different"), (1, 2, "assumed_different"), (0, 2, "assumed_same")):
        print(f"turn_{left}_turn_{right} {relation} {cosine_distance(turn_embeddings[left], turn_embeddings[right]):.6f}")


run("raw.wav")
run("cleaned.wav")
