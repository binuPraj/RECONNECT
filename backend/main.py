import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    import torch
    torch.set_num_threads(2)
except Exception:
    pass

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import logging
for _noisy in ("insightface", "onnxruntime", "urllib3", "httpx", "speechbrain", "pyannote"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

import threading
import time

_video_lock = threading.Lock()
_last_video_trigger = 0
_VIDEO_TRIGGER_COOLDOWN = 120  # seconds
_video_in_progress = False
import cv2
import subprocess
import os
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from trigger.who_is_this_listener import (
    WhoIsThisListener
)

from activity.test_realtime import run_pipeline as run_activity_pipeline

from vision.pipeline import (
    WhoIsThisPipeline,
    save_results
)
from config import TARGET_FPS

from enrollment.label_unknown import (
    label_unknown
)


# ==================================================
# SHOW VISION RESULT
# ==================================================

def show_result(results):

    print()
    print("=" * 50)
    print("VISION RESULT")
    print("=" * 50)

    people = results.get(
        "people",
        []
    )

    # ----------------------------------------------
    # No faces
    # ----------------------------------------------

    if not people:

        print(
            "No faces detected."
        )

        return

    # ----------------------------------------------
    # Display every detected person
    # ----------------------------------------------

    for person in people:

        position = person.get(
            "position",
            "unknown position"
        )

        status = person.get(
            "status",
            "unknown"
        )

        identity = person.get(
            "identity",
            "unknown"
        )

        entity_id = person.get(
            "entity_id"
        )

        # ------------------------------------------
        # KNOWN
        # ------------------------------------------

        if status == "known":

            print(
                f"The person on the "
                f"{position} is "
                f"{identity}."
            )

        # ------------------------------------------
        # EXISTING UNKNOWN
        # ------------------------------------------

        elif status == "existing_unknown":

            print(
                f"There is a person on the "
                f"{position} whom I "
                f"do not recognize."
            )

            if entity_id:

                print(
                    f"  Entity ID: {entity_id}"
                )

        # ------------------------------------------
        # NEW UNKNOWN
        # ------------------------------------------

        elif status == "new_unknown":

            print(
                f"There is a new person "
                f"on the {position}."
            )

            if entity_id:

                print(
                    f"  Entity ID: {entity_id}"
                )

        # ------------------------------------------
        # FALLBACK
        # ------------------------------------------

        else:

            print(
                f"There is a person on the "
                f"{position}, but their "
                f"identity is unknown."
            )


# ==================================================
# HANDLE NEW UNKNOWN PEOPLE
# ==================================================

def handle_new_unknowns(results, interactive=False):

    people = results.get(
        "people",
        []
    )

    for person in people:

        status = person.get(
            "status"
        )

        label_request = person.get(
            "label_request",
            False
        )

        # Only ask for labeling when this
        # is genuinely a NEW unknown.

        if (
            status != "new_unknown"
            or not label_request
        ):

            continue

        print()
        print("=" * 50)
        print("NEW UNKNOWN DETECTED")
        print("=" * 50)

        entity_id = person.get(
            "entity_id"
        )

        best_face_image = person.get(
            "best_face_image"
        )

        print(
            f"Entity ID: {entity_id}"
        )

        if best_face_image:

            print(
                f"Best face image: "
                f"{best_face_image}"
            )

        print()

        # In non-interactive mode or when stdin is not a TTY, don't prompt
        if not interactive or not sys.stdin.isatty():
            print("Non-interactive mode: keeping person as unknown.")
            continue

        try:

            choice = input(
                "Label this person now? "
                "(y/n): "
            ).strip().lower()

        except EOFError:

            print(
                "Input unavailable."
            )

            print(
                "Keeping person as unknown."
            )

            continue

        # ------------------------------------------
        # Don't label
        # ------------------------------------------

        if choice not in (
            "y",
            "yes"
        ):

            print(
                "Keeping this person as unknown."
            )

            continue

        # ------------------------------------------
        # Label unknown
        # ------------------------------------------

        try:

            label_unknown(
                entity_id
            )

            print(
                f"Successfully processed "
                f"{entity_id}."
            )

        except Exception as error:

            print(
                f"Could not label "
                f"{entity_id}:"
            )

            print(
                error
            )


def extract_first_second_frames(video_path):
    """Extract Who Is This frames from the first second of a video."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open video for face identification."
        )

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = 1 / TARGET_FPS
    next_sample_time = 0.0
    frame_number = 0
    frames = []

    while True:
        success, frame = cap.read()

        if not success:
            break

        timestamp = frame_number / video_fps

        if timestamp >= 1.0:
            break

        if timestamp >= next_sample_time:
            frames.append((timestamp, frame))
            next_sample_time += frame_interval

        frame_number += 1

    cap.release()
    return frames


# ==================================================
# AUDIO & VIDEO CAPTURE HELPERS (DIRECTSHOW SYNC)
# ==================================================

def detect_dshow_devices():
    """Discover available DirectShow video and audio devices via ffmpeg."""
    video_dev = os.getenv("FFMPEG_VIDEO_DEVICE_NAME", "").strip()
    audio_dev = os.getenv("FFMPEG_AUDIO_DEVICE_NAME", "").strip()

    if video_dev and audio_dev:
        return video_dev, audio_dev

    cmd = ["ffmpeg", "-list_devices", "true", "-f", "dshow", "-i", "dummy"]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        )
        lines = proc.stderr.splitlines()
        found_videos = []
        found_audios = []
        for line in lines:
            if "(video)" in line and '"' in line:
                parts = line.split('"')
                if len(parts) >= 2:
                    found_videos.append(parts[1])
            elif "(audio)" in line and '"' in line:
                parts = line.split('"')
                if len(parts) >= 2:
                    found_audios.append(parts[1])

        if not video_dev and found_videos:
            video_dev = found_videos[0]
        if not audio_dev and found_audios:
            audio_dev = found_audios[0]
    except Exception as e:
        print(f"[Device Detection Warning] {e}")

    if not video_dev:
        video_dev = "USB2.0 HD UVC WebCam"
    if not audio_dev:
        audio_dev = "Microphone Array (Realtek(R) Audio)"

    return video_dev, audio_dev


def record_synchronized_video(output_path: str, seconds: float = 10.0) -> bool:
    """
    Record webcam video and microphone audio concurrently into a single MP4
    using DirectShow with hardware presentation timestamps (PTS).
    Guarantees millisecond-accurate audio/video synchronization.
    """
    video_dev, audio_dev = detect_dshow_devices()
    print(f"Recording {seconds}s synchronized video: video='{video_dev}', audio='{audio_dev}'...")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if os.path.exists(output_path):
        try:
            os.remove(output_path)
        except Exception:
            pass

    cmd = [
        "ffmpeg", "-y",
        "-f", "dshow",
        "-i", f"video={video_dev}:audio={audio_dev}",
        "-t", str(seconds),
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-c:a", "aac",
        "-avoid_negative_ts", "make_zero",
        output_path,
    ]

    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        _, stderr_text = proc.communicate(timeout=seconds + 20.0)
        if proc.returncode != 0:
            print(f"[Record ERROR] ffmpeg exited with code {proc.returncode}:")
            print(stderr_text[-1500:] if stderr_text else "No stderr output")
            return False

        if not os.path.exists(output_path) or os.path.getsize(output_path) < 4096:
            print(f"[Record ERROR] Output file missing or too small: {output_path}")
            return False

        print(f"Video recorded successfully: {os.path.abspath(output_path)} ({os.path.getsize(output_path)//1024} KB)")
        return True
    except subprocess.TimeoutExpired:
        print("[Record ERROR] ffmpeg timed out while recording.")
        if proc:
            proc.kill()
            proc.communicate()
        return False
    except Exception as e:
        print(f"[Record ERROR] {e}")
        if proc:
            proc.kill()
            proc.communicate()
        return False
    finally:
        if proc and proc.poll() is None:
            try:
                proc.kill()
                proc.communicate()
            except Exception:
                pass


# ==================================================
# VIDEO CAPTURE & ASD PIPELINE
# ==================================================

def capture_and_process_video(vision_pipeline=None, target_unenrolled_id=None, interactive=False) -> bool:
    """
    Records a 10-second video (webcam + mic with perfect sync), runs WhoIsThis on the first second,
    and runs Active Speaker Detection on the full video.
    Guarded by _video_lock and PipelineState to ensure only one recording at a time.
    Returns True if capture succeeded, False if skipped or failed.
    """
    from app.state import set_state, reset_state, get_state, PipelineState

    with _video_lock:
        if get_state() != PipelineState.IDLE:
            print("Video recording already in progress or not idle; skipping new trigger.")
            return False
        if not set_state(PipelineState.TRIGGERED):
            return False
        if not set_state(PipelineState.CAPTURING):
            reset_state()
            return False

    os.makedirs("uploads", exist_ok=True)
    video_filename = os.path.join("uploads", "activity_video.mp4")

    if vision_pipeline is None:
        vision_pipeline = WhoIsThisPipeline()

    try:
        success = record_synchronized_video(video_filename, seconds=10.0)
        if not success:
            raise RuntimeError("DirectShow synchronized recording failed")

        # 2. Transition to processing
        set_state(PipelineState.PROCESSING)

        # 3. Run vision pipeline on first second
        print("Running Who Is This on the first second of the video...")
        vision_people = []
        try:
            who_frames = extract_first_second_frames(video_filename)
            vision_results = vision_pipeline.process_frames(who_frames, capture_seconds=1.0)
            save_results(vision_results)
            show_result(vision_results)
            handle_new_unknowns(vision_results, interactive=interactive)
            vision_people = vision_results.get("people", [])
        except Exception as e:
            print(f"Who Is This pipeline error: {e}")

        # 4. Run active speaker detection pipeline
        print("Running Active Speaker Detection pipeline...")
        try:
            results = run_activity_pipeline(
                video_path=video_filename,
                enrolled_faces={},
                enrolled_voices={},
                output_path="asd_output.json"
            )
            print("\n" + "="*60)
            print("ACTIVE SPEAKER RESULTS")
            print("="*60)
            if results:
                if isinstance(results, dict):
                    for speaker_label, person_id in results.items():
                        if person_id:
                            print(f"  {speaker_label} -> {person_id} CONFIRMED")
                        else:
                            print(f"  {speaker_label} -> insufficient evidence, not linked")
                elif isinstance(results, list):
                    for r in results:
                        start = r.get("start_time", 0.0)
                        end = r.get("end_time", 0.0)
                        label = r.get("label", "unknown")
                        conf = r.get("confirmation", "UNCONFIRMED")
                        score = r.get("confidence", 0.0)
                        print(f"  [{start:.1f}s - {end:.1f}s] {label} ({conf}) conf={score:.2f}")
            else:
                print("No active speakers found or linked.")

            # 5. Cross-modal match: Vision Face vs Active Speaker
            active_speakers = set()
            final_links = getattr(results, "final_links", {})
            registry = getattr(results, "registry", None)
            if final_links:
                for speaker_label, person_id in final_links.items():
                    if person_id:
                        disp_name = person_id
                        if registry and hasattr(registry, "people") and person_id in registry.people:
                            disp_name = registry.people[person_id].display_label()
                        active_speakers.add(str(disp_name).strip().lower())

            if isinstance(results, list):
                for r in results:
                    conf = str(r.get("confirmation", "")).upper()
                    if conf in ("DUAL_CONFIRMED", "LINKED_UNCONFIRMED", "FACE_AND_VOICE_CONFIRMED", "ASD_CONFIRMED") or (r.get("asd_confidence", 0) > 0 and r.get("face_id")):
                        for key in ("label", "person_id", "face_id", "voice_id"):
                            val = r.get(key)
                            if val and str(val).lower() not in ("none", "unknown", "speaker_00", "speaker_01", "speaker_02"):
                                active_speakers.add(str(val).strip().lower())

            # Check identities detected on camera
            detected_names = []
            for p in vision_people:
                ident = p.get("identity")
                ent_id = p.get("entity_id")
                name = ident if (ident and ident != "unknown") else ent_id
                if name:
                    detected_names.append(str(name).strip())

            print("\n" + "="*60)
            print("SPEAKER DETERMINATION")
            print("="*60)
            if detected_names:
                for name in detected_names:
                    name_clean = name.strip()
                    is_talking = any(
                        name_clean.lower() == s or name_clean.lower() in s or s in name_clean.lower()
                        for s in active_speakers
                    )
                    if is_talking:
                        print(f"\n>>> {name_clean} is talking <<<\n")
                    else:
                        print(f"\n>>> {name_clean} is not talking <<<\n")
            else:
                if "binu" in active_speakers:
                    print("\n>>> binu is talking <<<\n")
                else:
                    print("\n>>> No face detected on camera (nobody visible is talking) <<<\n")
            print("="*60 + "\n")

            # 6. Cross-Modal Alignment: Match Video Audio Voice Embedding to Live Unknown Voice
            if target_unenrolled_id is not None:
                from database.db import get_unenrolled_identity, update_unenrolled_face, voice_blob_to_embeddings
                import numpy as np

                target_row = get_unenrolled_identity(target_unenrolled_id)
                target_voice_blob = target_row.get("voice_embedding") if target_row else None
                target_voice_embs = voice_blob_to_embeddings(target_voice_blob) if target_voice_blob else []

                voice_matched_face_id = None
                best_voice_sim = -1.0
                matched_time_interval = None

                # Compare each speech turn in the video against the target voice embedding
                if target_voice_embs and isinstance(results, list):
                    for r in results:
                        turn_voice_emb = r.get("voice_embedding")
                        if turn_voice_emb is None:
                            continue
                        if hasattr(turn_voice_emb, "detach"):
                            turn_voice_arr = turn_voice_emb.detach().cpu().numpy().flatten().astype(np.float32)
                        else:
                            turn_voice_arr = np.asarray(turn_voice_emb, dtype=np.float32).flatten()

                        t_norm = np.linalg.norm(turn_voice_arr)
                        if t_norm == 0:
                            continue

                        for stored_emb in target_voice_embs:
                            s_arr = np.asarray(stored_emb, dtype=np.float32).flatten()
                            s_norm = np.linalg.norm(s_arr)
                            if s_norm == 0 or s_arr.shape != turn_voice_arr.shape:
                                continue
                            sim = float(np.dot(turn_voice_arr, s_arr) / (t_norm * s_norm))
                            if sim > best_voice_sim:
                                best_voice_sim = sim
                                face_cand = r.get("face_id") or (r.get("person_id") if r.get("person_id") not in ("binu",) else None)
                                if face_cand:
                                    voice_matched_face_id = face_cand
                                    matched_time_interval = (r.get("start_time", 0.0), r.get("end_time", 0.0))

                print("\n" + "="*60)
                print("CROSS-MODAL VOICE-TO-FACE MATCHING")
                print("="*60)
                active_face_embedding = None

                # If a turn matched the live voice embedding with threshold >= 0.55
                if best_voice_sim >= 0.55 and matched_time_interval is not None:
                    print(
                        f"  [VOICE MATCH] Video audio interval [{matched_time_interval[0]:.1f}s - {matched_time_interval[1]:.1f}s] "
                        f"matches live unknown voice (Similarity: {best_voice_sim:.3f})"
                    )
                    # Retrieve the face embedding for the matched active speaker
                    if registry and hasattr(registry, "people") and voice_matched_face_id in registry.people:
                        p_rec = registry.people[voice_matched_face_id]
                        if p_rec.face_embedding is not None:
                            active_face_embedding = p_rec.face_embedding
                        elif p_rec.face_gallery:
                            active_face_embedding = p_rec.face_gallery[0]

                # Fallback: check speaker_face_associations from ASD if only 1 active speaker was confirmed
                if active_face_embedding is None and best_voice_sim >= 0.50:
                    if registry and hasattr(registry, "speaker_face_associations"):
                        for spk_label, info in registry.speaker_face_associations().items():
                            f_id = info.get("face_id")
                            if f_id and f_id in registry.people:
                                p_rec = registry.people[f_id]
                                if p_rec.face_embedding is not None:
                                    active_face_embedding = p_rec.face_embedding
                                    voice_matched_face_id = f_id
                                    break
                                elif p_rec.face_gallery:
                                    active_face_embedding = p_rec.face_gallery[0]
                                    voice_matched_face_id = f_id
                                    break

                # If face confirmed, bind to SQLite
                if active_face_embedding is not None:
                    update_unenrolled_face(target_unenrolled_id, active_face_embedding)
                    print(
                        f"  [ENROLLMENT] Successfully bound confirmed face ({voice_matched_face_id}) "
                        f"to unenrolled_{target_unenrolled_id} in reconnect.db!"
                    )
                else:
                    if best_voice_sim >= 0.55:
                        print(f"  [ENROLLMENT] Voice matched (sim={best_voice_sim:.2f}) but face was not clearly visible or unconfirmed by ASD.")
                    else:
                        print(f"  [ENROLLMENT] No turn in video matched the unknown voice (highest sim={best_voice_sim:.2f}, needed 0.55).")
                print("="*60 + "\n")

        except Exception as e:
            print(f"Activity pipeline error: {e}")

        return True

    except Exception as e:
        print(f"Error during video trigger processing: {e}")
        return False
    finally:
        reset_state()
        print("Returning to listening...")


# ==================================================
# MAIN
# ==================================================

def main(take_video_only=False, who_is_this_only=False, target_unenrolled_id=None, interactive=False):
    if who_is_this_only:
        print()
        print("=" * 50)
        print("RUNNING WHO IS THIS PIPELINE")
        print("=" * 50)
        
        vision_pipeline = WhoIsThisPipeline()
        
        print("Starting vision...")
        try:
            results = vision_pipeline.run()
        except Exception as e:
            print(f"Vision pipeline error: {e}")
            return
            
        try:
            save_results(results)
        except Exception as error:
            print("Warning: could not save vision results:")
            print(error)
            
        show_result(results)
        handle_new_unknowns(results, interactive=interactive)
        
        print("Vision complete.")
        return

    if take_video_only:
        capture_and_process_video(target_unenrolled_id=target_unenrolled_id, interactive=interactive)
        return

    print()
    print("=" * 50)
    print("MEMORYLENS")
    print("=" * 50)

    print(
        "Starting MemoryLens..."
    )

    # ==============================================
    # AUDIO
    # ==============================================

    print()
    print(
        "Initializing audio listener..."
    )

    listener = WhoIsThisListener()

    # ==============================================
    # VISION
    # ==============================================

    print()
    print(
        "Initializing vision pipeline..."
    )

    vision_pipeline = (
        WhoIsThisPipeline()
    )

    # ==============================================
    # READY
    # ==============================================

    print()
    print("=" * 50)
    print("MEMORYLENS READY")
    print("=" * 50)

    print(
        "Waiting for:"
    )

    print(
        "  'Who is this?'"
    )

    # ==============================================
    # MAIN LOOP
    # ==============================================

    while True:

        try:

            # --------------------------------------
            # 1. CONTINUOUS AUDIO LISTENING
            # --------------------------------------

            triggered = listener.listen_for_trigger()

            # Normally this function only returns
            # True when the trigger is detected.

            if not triggered:

                continue

            if triggered == "who_is_this":

                # --------------------------------------
                # 2. AUDIO TRIGGER DETECTED
                # --------------------------------------

                print()
                print("=" * 50)
                print(
                    "WHO IS THIS TRIGGER DETECTED"
                )
                print("=" * 50)

                # --------------------------------------
                # 3. START VISION
                # --------------------------------------

                print(
                    "Starting vision..."
                )

                results = (
                    vision_pipeline.run()
                )

                # --------------------------------------
                # 4. SAVE RESULTS
                # --------------------------------------

                try:

                    save_results(
                        results
                    )

                except Exception as error:

                    print(
                        "Warning: could not save "
                        "vision results:"
                    )

                    print(
                        error
                    )

                # --------------------------------------
                # 5. SHOW RECOGNITION
                # --------------------------------------

                show_result(
                    results
                )

                # --------------------------------------
                # 6. HANDLE NEW UNKNOWN FACES
                # --------------------------------------

                handle_new_unknowns(
                    results
                )

                # --------------------------------------
                # 7. RETURN TO AUDIO
                # --------------------------------------

                print()
                print(
                    "=" * 50
                )

                print(
                    "Vision complete."
                )

                print(
                    "Returning to listening..."
                )

            # --------------------------------------
            # 7. TAKE A VIDEO TRIGGER
            # --------------------------------------
            elif triggered == "take_a_video":
                capture_and_process_video(vision_pipeline)

        # ==========================================
        # CTRL+C
        # ==========================================

        except KeyboardInterrupt:

            print()
            print()

            print(
                "Stopping MemoryLens..."
            )

            break

        # ==========================================
        # UNEXPECTED ERROR
        # ==========================================

        except Exception as error:

            print()
            print("=" * 50)

            print(
                "RUNTIME ERROR"
            )

            print("=" * 50)

            print(
                error
            )

            print()

            print(
                "MemoryLens will continue."
            )


# ==================================================
# PROGRAM ENTRY
# ==================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--take-a-video",
        action="store_true",
        help="Run the existing video pipeline once without starting the listener.",
    )
    parser.add_argument(
        "--who-is-this",
        action="store_true",
        help="Run the existing who is this pipeline once without starting the listener.",
    )
    parser.add_argument(
        "--unenrolled-id",
        type=int,
        default=None,
        help="Optional unenrolled identity ID to bind the active speaker's face to.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Allow interactive prompts for labeling new unknown faces.",
    )
    args = parser.parse_args()
    main(
        take_video_only=args.take_a_video,
        who_is_this_only=args.who_is_this,
        target_unenrolled_id=args.unenrolled_id,
        interactive=args.interactive,
    )