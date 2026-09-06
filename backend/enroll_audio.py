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

VOICE_DURATION_SECONDS = 5
VOICE_SAMPLE_RATE = 16000


def create_voice_embedding(audio_path: str | os.PathLike) -> np.ndarray:
    """Create an embedding from an audio file using the existing pipeline."""
    embedder = AudioEmbedder()
    return np.asarray(
        embedder.extract_embedding(str(audio_path)),
        dtype=np.float32,
    )


def record_voice_embedding(
    duration_seconds: int = VOICE_DURATION_SECONDS,
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

    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        dir=Path(__file__).resolve().parent / "data",
        delete=False,
    ) as temporary_file:
        audio_path = Path(temporary_file.name)

    try:
        sf.write(audio_path, recording[:, 0], sample_rate)
        return create_voice_embedding(audio_path)
    finally:
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
                embedding = embedder.extract_embedding(audio_path)
                np.save(emb_path, embedding)
                print(f"Successfully enrolled {person_name} -> {emb_filename}")
            except Exception as e:
                print(f"Failed to extract embedding for {person_name} file {audio_filename}: {e}")

if __name__ == "__main__":
    batch_enroll()
