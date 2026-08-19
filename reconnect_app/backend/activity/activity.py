import os
import sys
import cv2
import torch
import numpy as np
import torchaudio
import argparse
import json
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from scipy.io import wavfile
from scipy.spatial.distance import cosine

from identity_registry import PersonRegistry

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

TALKNET_DIR = Path(__file__).resolve.parents[3] / "TalkNet-ASD"
sys.path.insert(0, str(TALKNET_DIR))

from talkNet import talkNet


#huggingface token for PyAnnote
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


FACE_MATCH_THRESHOLD = 0.60
VOICE_MATCH_THRESHOLD =0.70
ASD_THRESHOLD = 0.50

def extract_audio_from_video(video_path: str, output_audio: str = "temp_audio.wav") -> str:
    import subprocess
    print(f"[step 1]extracting audio from {video_path}.")

    cmd= [
        "ffmpeg", "-y",
        "-i", video_path,
        "-ac", "1",
        "-ar", "16000",
        "-vn",
        output_audio
    ]

    result = subprocess.run(cmd, capture_output = True, text= True)
    if result.returncod != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")

    print(f" Audio extracted -> {output_audio}")
    return output_audio



def extract_frames(video_path: str, fps: int =1) -> list:
    print(f"[step1] extracting frames at {fps}fps.")
    cap= cv2.VideoCapture(video_path)
    video_fps =cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int( video_fps / fps)

    frames = []
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx %frame_interval == 0:
            timestamp = frame_idx /video_fps

            frames.append((timestamp, frame))
        frame_idx+= 1

    cap.release()
    print (f" Extracted {len(frames)} frames from {video_path}")

def extract_frames_in_range(video_path:str, start: float, end:float) -> list:
    cap=cv2.VideoCapture(video_path)
    video_fps=cap.get(cv2.CAP_PROP_FPS) or 25.0
    start_frame = int(start * video_fps)
    end_frame = int(end * video_fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)   

    frames=[]
    frame_idx = start_frame
    while frame_idx <= end_frame:
        ret,frame=cap.read()
        if not ret:
            break
        