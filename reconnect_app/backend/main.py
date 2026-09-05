import time
import cv2
import subprocess
import os
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
# MAIN
# ==================================================

def main():

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

    listener = (
        WhoIsThisListener()
    )

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

            triggered = (
                listener.listen_for_trigger()
            )

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

            elif triggered == "take_a_video":

                print()
                print("=" * 50)
                print(
                    "TAKE A VIDEO TRIGGER DETECTED"
                )
                print("=" * 50)

                video_filename = "activity_video.mp4"
                print("Recording 5-second video (webcam + mic)...")

                try:
                    # 1. Open and warm up the camera before starting audio.
                    temp_aud = "temp_audio.wav"
                    temp_vid = "temp_video.avi"

                    cap = cv2.VideoCapture(0)
                    if not cap.isOpened():
                        print("Failed to open webcam.")
                        continue

                    first_frame = None
                    while first_frame is None:
                        ret, frame = cap.read()
                        if ret and frame is not None and frame.shape[0] > 0:
                            first_frame = frame
                        else:
                            time.sleep(0.05)

                    # 2. Start audio only after the camera is producing frames.
                    audio_proc, audio_input_name = _start_audio_capture(
                        temp_aud=temp_aud,
                        seconds=5,
                    )
                    if audio_proc is None:
                        print("Could not start microphone recording via ffmpeg.")
                        print("Set FFMPEG_AUDIO_DEVICE_NAME to your exact mic device name and retry.")
                        print("Tip: run 'ffmpeg -list_devices true -f dshow -i dummy' to list names.")
                        cap.release()
                        continue

                    # The warm-up frame was captured before audio started.
                    # Discard it so the first written video frame aligns with audio.
                        
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    if fps <= 0: fps = 30.0
                    
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
                            # Camera hardware might need time to warm up
                            time.sleep(0.05)
                            continue
                            
                    cap.release()
                    if out is not None:
                        out.release()
                    
                    # 3. Wait for audio to finish
                    audio_proc.wait()

                    if not os.path.exists(temp_aud) or os.path.getsize(temp_aud) < 4096:
                        print("Audio track appears empty or too small.")
                        print(f"Mic input attempted: {audio_input_name}")
                        print("Try setting FFMPEG_AUDIO_DEVICE_NAME and retry.")
                        continue
                    
                    if frames_written == 0:
                        print("Failed to capture any valid frames from the webcam.")
                        continue
                    
                    # 4. Merge them together
                    mux_cmd = [
                        "ffmpeg", "-y", "-i", temp_vid, "-i", temp_aud, 
                        "-c:v", "libx264", "-c:a", "aac", video_filename
                    ]
                    subprocess.run(
                        mux_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                    
                    print("Video recorded successfully to activity_video.mp4!")
                except Exception as e:
                    print(f"Failed to record video/audio: {e}")
                    print("Returning to listening...")
                    continue

                print("Running Who Is This on the first second of the video...")
                try:
                    who_frames = extract_first_second_frames(
                        video_filename
                    )

                    results = vision_pipeline.process_frames(
                        who_frames,
                        capture_seconds=1.0
                    )

                    save_results(results)
                    show_result(results)
                    handle_new_unknowns(results)

                except Exception as e:
                    print(f"Who Is This pipeline error: {e}")

                print("Running Active Speaker Detection pipeline...")
                try:
                    # run_activity_pipeline takes video_path, enrolled_faces, enrolled_voices
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
                        for speaker_label, person_id in results.items():
                            if person_id:
                                print(f"  {speaker_label} → {person_id} CONFIRMED")
                            else:
                                print(f"  {speaker_label} → insufficient evidence, not linked")
                    else:
                        print("No active speakers found or linked.")

                except Exception as e:
                    print(f"Activity pipeline error: {e}")

                print()
                print("=" * 50)
                print("Activity detection complete.")
                print("Returning to listening...")

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

    main()