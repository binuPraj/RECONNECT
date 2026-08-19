import speech_recognition as sr


class WhoIsThisListener:

    def __init__(self):

        self.recognizer = sr.Recognizer()

        # MacBook Air Microphone
        # Device 0 was confirmed on your system.
        self.microphone = sr.Microphone(
            device_index=0
        )

        self.trigger_phrases = [
            "who is this",
            "who's this",
            "who is that",
            "who's that"
        ]

        self.video_trigger_phrases = [
            "take a video",
            "record a video"
        ]

        # --------------------------------------
        # Calibrate microphone ONCE
        # --------------------------------------

        print("Calibrating microphone...")

        with self.microphone as source:

            self.recognizer.adjust_for_ambient_noise(
                source,
                duration=1
            )

        print(
            "Microphone ready."
        )

        print(
            f"Energy threshold: "
            f"{self.recognizer.energy_threshold:.0f}"
        )

    # ------------------------------------------
    # Continuous trigger listener
    # ------------------------------------------

    def listen_for_trigger(self):

        print()
        print("Listening...")
        print("Say: 'Who is this?' or 'Take a video'")

        while True:

            try:

                # --------------------------------
                # Wait for a speech segment
                # --------------------------------

                with self.microphone as source:

                    audio = self.recognizer.listen(
                        source,

                        # Wait indefinitely for
                        # someone to start speaking.
                        timeout=None,

                        # Maximum length of one
                        # speech segment.
                        phrase_time_limit=3
                    )

                print(
                    "Speech captured. "
                    "Recognizing..."
                )

                # --------------------------------
                # Speech -> text
                # --------------------------------

                text = self.recognizer.recognize_google(
                    audio
                )

                text = text.lower().strip()

                print(
                    f"Heard: '{text}'"
                )

                # --------------------------------
                # Check trigger
                # --------------------------------

                for trigger in self.trigger_phrases:
                    if trigger in text:
                        print()
                        print("=" * 50)
                        print("WHO IS THIS TRIGGER DETECTED")
                        print("=" * 50)
                        return "who_is_this"

                for trigger in self.video_trigger_phrases:
                    if trigger in text:
                        print()
                        print("=" * 50)
                        print("TAKE A VIDEO TRIGGER DETECTED")
                        print("=" * 50)
                        return "take_a_video"

                # --------------------------------
                # Not the trigger
                # --------------------------------

                print(
                    "Not a trigger."
                )

                print(
                    "Continuing to listen..."
                )

            # ------------------------------------
            # Speech was not understandable
            # ------------------------------------

            except sr.UnknownValueError:

                print(
                    "Could not understand speech."
                )

                print(
                    "Continuing to listen..."
                )

                continue

            # ------------------------------------
            # Google STT / network error
            # ------------------------------------

            except sr.RequestError as error:

                print(
                    f"Speech recognition error: "
                    f"{error}"
                )

                print(
                    "Continuing to listen..."
                )

                continue

            # ------------------------------------
            # Ctrl+C
            # ------------------------------------

            except KeyboardInterrupt:

                print()
                print(
                    "Stopping audio listener..."
                )

                raise