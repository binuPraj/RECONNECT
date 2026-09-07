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

def handle_new_unknowns(results):

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
# AUDIO CAPTURE HELPERS
# ==================================================

def _audio_input_candidates():
    """Build candidate DirectShow audio inputs for ffmpeg."""
    candidates = []

    env_name = os.getenv("FFMPEG_AUDIO_DEVICE_NAME", "").strip()
    if env_name:
        candidates.append(f"audio={env_name}")

    candidates.extend([
        "audio=Microphone Array (Realtek(R) Audio)",
        "audio=Microphone (Realtek(R) Audio)",
        "audio=default",
    ])

    deduped = []
    seen = set()
    for item in candidates:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


def _start_audio_capture(temp_aud: str, seconds: int):
    """Try several audio device inputs and return (proc, input_name)."""
    for input_name in _audio_input_candidates():
        audio_cmd = [
            "ffmpeg", "-y", "-f", "dshow",
            "-i", input_name,
            "-t", str(seconds),
            temp_aud,
        ]

        proc = subprocess.Popen(
            audio_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Check inputs that fail immediately without delaying video start.
        if proc.poll() is not None:
            continue

        print(f"Using audio input: {input_name}")
        return proc, input_name

    return None, None


# ==================================================
# VIDEO CAPTURE & ASD PIPELINE
# ==================================================

def capture_and_process_video(vision_pipeline=None) -> bool:
    """
    Records a 5-second video (webcam + mic), runs WhoIsThis on the first second,
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

    print("Recording 5-second video (webcam + mic)...")
    os.makedirs("uploads", exist_ok=True)
    temp_vid = os.path.join("uploads", f"temp_video_{int(time.time())}.avi")
    temp_aud = os.path.join("uploads", f"temp_audio_{int(time.time())}.wav")
    video_filename = os.path.join("uploads", "activity_video.mp4")

    if vision_pipeline is None:
        vision_pipeline = WhoIsThisPipeline()

    try:
        # 1. Start audio capture
        audio_proc, audio_input_name = _start_audio_capture(
            temp_aud=temp_aud,
            seconds=5,
        )
        if audio_proc is None:
            print("Could not start microphone recording via ffmpeg.")
            print("Set FFMPEG_AUDIO_DEVICE_NAME to your exact mic device name and retry.")
            print("Tip: run 'ffmpeg -list_devices true -f dshow -i dummy' to list names.")
            raise RuntimeError("Audio capture failed")

        # 2. Capture video frames
        cap = cv2.VideoCapture(0)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        out = None
        frames_written = 0
        start_time = time.time()
        while time.time() - start_time < 5.0:
            ret, frame = cap.read()
            if ret and frame is not None and frame.shape[0] > 0:
                if out is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    out = cv2.VideoWriter(temp_vid, fourcc, fps, (w, h))
                out.write(frame)
                frames_written += 1
            else:
                time.sleep(0.05)
                continue
        cap.release()
        if out is not None:
            out.release()

        # 3. Wait for audio to finish and validate
        audio_proc.wait()
        if not os.path.exists(temp_aud) or os.path.getsize(temp_aud) < 4096:
            print("Audio track appears empty or too small.")
            print(f"Mic input attempted: {audio_input_name}")
            raise RuntimeError("Invalid audio file")
        if frames_written == 0:
            print("Failed to capture any valid frames from the webcam.")
            raise RuntimeError("No video frames captured")

        # 4. Merge audio and video with timeout to avoid hangs
        mux_cmd = [
            "ffmpeg", "-y", "-i", temp_vid, "-i", temp_aud,
            "-c:v", "libx264", "-c:a", "aac", video_filename
        ]
        subprocess.run(mux_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
        print("Video recorded successfully to activity_video.mp4!")

        # 5. Transition to processing
        set_state(PipelineState.PROCESSING)

        # 6. Run vision pipeline on first second
        print("Running Who Is This on the first second of the video...")
        vision_people = []
        try:
            who_frames = extract_first_second_frames(video_filename)
            vision_results = vision_pipeline.process_frames(who_frames, capture_seconds=1.0)
            save_results(vision_results)
            show_result(vision_results)
            handle_new_unknowns(vision_results)
            vision_people = vision_results.get("people", [])
        except Exception as e:
            print(f"Who Is This pipeline error: {e}")

        # 7. Run active speaker detection pipeline
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

            # 8. Cross-modal match: Vision Face vs Active Speaker
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

        except Exception as e:
            print(f"Activity pipeline error: {e}")

        return True

    except Exception as e:
        print(f"Error during video trigger processing: {e}")
        return False
    finally:
        # Cleanup temporary files
        for fp in (temp_vid, temp_aud):
            try:
                if os.path.exists(fp):
                    os.remove(fp)
            except Exception:
                pass
        reset_state()
        print("Returning to listening...")


# ==================================================
# MAIN
# ==================================================

def main(take_video_only=False, who_is_this_only=False):
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
        handle_new_unknowns(results)
        
        print("Vision complete.")
        return

    if take_video_only:
        capture_and_process_video()
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
    args = parser.parse_args()
    main(take_video_only=args.take_a_video, who_is_this_only=args.who_is_this)