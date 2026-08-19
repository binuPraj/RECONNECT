import time
import cv2
import subprocess
from trigger.who_is_this_listener import (
    WhoIsThisListener
)

from activity.test_realtime import run_pipeline as run_activity_pipeline

from vision.pipeline import (
    WhoIsThisPipeline,
    save_results
)

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


# ==================================================
# MAIN
# ==================================================

def main():

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
                    # 1. Start ffmpeg to record audio in the background
                    temp_aud = "temp_audio.wav"
                    temp_vid = "temp_video.avi"
                    
                    audio_cmd = [
                        "ffmpeg", "-y", "-f", "dshow", 
                        "-i", "audio=Microphone Array (Realtek(R) Audio)",
                        "-t", "5", temp_aud
                    ]
                    audio_proc = subprocess.Popen(
                        audio_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )

                    # 2. Record video using OpenCV (Media Foundation)
                    cap = cv2.VideoCapture(0)
                    if not cap.isOpened():
                        print("Failed to open webcam.")
                        audio_proc.terminate()
                        continue
                        
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