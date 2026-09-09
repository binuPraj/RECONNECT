import os
import sys
import tempfile
from pathlib import Path

import sounddevice as sd
import soundfile as sf
import numpy as np

# Add the project root to sys.path so we can import from app
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.audio_process.recognition import AudioEmbedder
from app.audio_process.audio_utils import load_audio, resample_audio, save_audio
from app.audio_process.diarization import diarize_audio
from app.audio_process.preprocessing import preprocess_audio
from app.audio_process.vad import detect_speech_regions

VOICE_DURATION_SECONDS = 10
VOICE_SAMPLE_RATE = 16000
ENROLLMENT_AUDIO_PATH = Path(__file__).resolve().parent / "enrollment" / "enrollment_audio"


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Coalesce overlapping VAD-backed turns before measuring speaker time."""
    merged: list[tuple[float, float]] = []
    for start_sec, end_sec in sorted(intervals):
        if not merged or start_sec > merged[-1][1]:
            merged.append((start_sec, end_sec))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end_sec))
    return merged


def _prepare_dominant_speaker_audio(
    audio_path: str | os.PathLike,
    output_path: Path,
    work_dir: Path,
) -> dict[str, object]:
    """Extract, clean, and save the VAD-backed dominant speaker from an enrollment clip."""
    raw_audio, original_sample_rate = load_audio(Path(audio_path))
    raw_resampled_audio, sample_rate = resample_audio(
        raw_audio, original_sample_rate
    )
    speech_regions = detect_speech_regions(raw_resampled_audio, sample_rate)
    if not speech_regions:
        raise ValueError("Enrollment audio contains no VAD-detected speech.")

    # Reuse the live diarization model on the raw resampled waveform. Unlike
    # live multi-party processing, enrollment permits one speaker so a solo
    # recording remains a single cluster rather than being forced into two.
    diarization_turns = diarize_audio(
        raw_resampled_audio,
        sample_rate,
        min_speakers=1,
        max_speakers=6,
    )
    if not diarization_turns:
        raise ValueError("Enrollment speaker diarization returned no speaker turns.")

    intervals_by_speaker: dict[str, list[tuple[float, float]]] = {}
    for turn in diarization_turns:
        for region in speech_regions:
            start_sec = max(turn.start_sec, region.start_sec)
            end_sec = min(turn.end_sec, region.end_sec)
            if end_sec > start_sec:
                intervals_by_speaker.setdefault(turn.speaker_label, []).append(
                    (start_sec, end_sec)
                )

    intervals_by_speaker = {
        label: _merge_intervals(intervals)
        for label, intervals in intervals_by_speaker.items()
        if intervals
    }
    if not intervals_by_speaker:
        raise ValueError(
            "Enrollment diarization produced no turns overlapping VAD-detected speech."
        )

    dominant_label, dominant_intervals = max(
        intervals_by_speaker.items(),
        key=lambda item: sum(end - start for start, end in item[1]),
    )

    if len(intervals_by_speaker) == 1:
        # A single diarization cluster is already the sole speaker: retain the
        # complete waveform (including natural pauses) rather than needlessly
        # cutting it into VAD fragments.
        dominant_audio = raw_resampled_audio
    else:
        clips = []
        for start_sec, end_sec in dominant_intervals:
            start_sample = max(0, int(start_sec * sample_rate))
            end_sample = min(len(raw_resampled_audio), int(end_sec * sample_rate))
            if end_sample > start_sample:
                clips.append(raw_resampled_audio[start_sample:end_sample])
        if not clips:
            raise ValueError("Dominant enrollment speaker has no usable audio samples.")
        dominant_audio = np.concatenate(clips)

    dominant_raw_path = work_dir / "dominant_speaker_raw.wav"
    save_audio(dominant_raw_path, dominant_audio, sample_rate)
    preprocessing = preprocess_audio(dominant_raw_path, output_path)
    return {
        "dominant_speaker": dominant_label,
        "dominant_speech_duration_sec": sum(
            end - start for start, end in dominant_intervals
        ),
        "speaker_count": len(intervals_by_speaker),
        "preprocessing": preprocessing,
    }


def create_voice_embedding(
    audio_path: str | os.PathLike,
    *,
    prepared_audio_path: str | os.PathLike | None = None,
    embedder: AudioEmbedder | None = None,
) -> np.ndarray:
    """Create an embedding from the diarized dominant speaker in an enrollment clip."""
    embedder = embedder or AudioEmbedder()
    with tempfile.TemporaryDirectory(prefix="reconnect_enrollment_") as temporary_dir:
        work_dir = Path(temporary_dir)
        output_path = (
            Path(prepared_audio_path)
            if prepared_audio_path is not None
            else work_dir / "dominant_speaker_cleaned.wav"
        )
        preparation = _prepare_dominant_speaker_audio(
            audio_path, output_path, work_dir
        )
        print(
            "Enrollment dominant speaker "
            f"{preparation['dominant_speaker']} "
            f"({preparation['dominant_speech_duration_sec']:.2f}s VAD speech; "
            f"{preparation['speaker_count']} cluster(s))."
        )
        return np.asarray(
            embedder.extract_embedding(str(output_path)),
            dtype=np.float32,
        )


def record_voice_embedding(
    duration_seconds: int = VOICE_DURATION_SECONDS,
    identity: str | None = None,
    sample_number: int = 1,
) -> np.ndarray:
    """Record microphone audio and create its speaker embedding."""
    device = sd.query_devices(None, "input")
    sample_rate = int(round(device["default_samplerate"]))
    frame_count = sample_rate * max(duration_seconds, VOICE_DURATION_SECONDS)

    print(
        f"Recording voice for {max(duration_seconds, VOICE_DURATION_SECONDS)} "
        f"seconds from {device['name']}..."
    )
    recording = sd.rec(
        frame_count,
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    print("Voice recording complete.")

    if identity:
        audio_directory = ENROLLMENT_AUDIO_PATH / identity
        audio_directory.mkdir(parents=True, exist_ok=True)
        audio_path = audio_directory / f"enrollment_voice_{sample_number:02d}.wav"
    else:
        temporary_file = tempfile.NamedTemporaryFile(
            suffix=".wav",
            dir=Path(__file__).resolve().parent / "data",
            delete=False,
        )
        audio_path = Path(temporary_file.name)
        temporary_file.close()

    try:
        sf.write(audio_path, recording[:, 0], sample_rate)
        if identity:
            cleaned_path = audio_path.with_name(
                f"enrollment_voice_{sample_number:02d}_cleaned.wav"
            )
            embedding = create_voice_embedding(
                audio_path, prepared_audio_path=cleaned_path
            )
        else:
            embedding = create_voice_embedding(audio_path)
        print(f"Enrollment audio saved to {audio_path}")
        return embedding
    finally:
        if not identity:
            audio_path.unlink(missing_ok=True)

def batch_enroll():
    source_dir = os.path.join("data", "enrollment")
    output_dir = os.path.join("data", "embeddings")
    
    if not os.path.exists(source_dir):
        print(f"Creating source directory '{source_dir}'. Please add person folders (e.g., data/enrollment/dad/) and run again.")
        os.makedirs(source_dir, exist_ok=True)
        return
    
    os.makedirs(output_dir, exist_ok=True)

    print("Initializing AudioEmbedder...")
    try:
        embedder = AudioEmbedder()
    except Exception as e:
        print(f"Failed to initialize AudioEmbedder: {e}")
        return

    # Iterate over folders in data/enrollment
    for person_name in os.listdir(source_dir):
        person_source_dir = os.path.join(source_dir, person_name)
        
        # We only care about directories
        if not os.path.isdir(person_source_dir):
            continue
            
        person_output_dir = os.path.join(output_dir, person_name)
        
        # Find all valid audio files in the person's directory
        audio_files = [f for f in os.listdir(person_source_dir) if f.endswith(('.wav', '.mp3', '.flac', '.m4a'))]
        if not audio_files:
            print(f"Skipping '{person_name}': No audio files found in {person_source_dir}")
            continue
            
        os.makedirs(person_output_dir, exist_ok=True)
        
        for audio_filename in sorted(audio_files):
            # Extract base name without extension and append _emb.npy
            base_name = os.path.splitext(audio_filename)[0]
            emb_filename = f"{base_name}_emb.npy"
            emb_path = os.path.join(person_output_dir, emb_filename)
            
            # Skip if already exists
            if os.path.exists(emb_path):
                print(f"Skipping '{person_name}' file '{audio_filename}': Embedding already exists at {emb_path}")
                continue
                
            audio_path = os.path.join(person_source_dir, audio_filename)
            print(f"Enrolling '{person_name}' from {audio_path}...")
            
            try:
                # Batch files can contain cross-talk. Extract the dominant
                # diarized speaker before waveform-only denoising/normalizing
                # and ECAPA embedding rather than trusting the full clip.
                embedding = create_voice_embedding(audio_path, embedder=embedder)
                np.save(emb_path, embedding)
                print(f"Successfully enrolled {person_name} -> {emb_filename}")
            except Exception as e:
                print(f"Failed to extract embedding for {person_name} file {audio_filename}: {e}")

if __name__ == "__main__":
    batch_enroll()
