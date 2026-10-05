accepted]
INFO:     connection open
2026-09-09 05:51:56,344 | [STREAM] format=accepted session=session_001 source=44100Hz/1ch/pcm_s16le normalization=enabled
2026-09-09 05:51:59,271 | ----- [RECORDING] started segment=1 pre_roll=0.500s -----
2026-09-09 05:52:03,548 | ----- [RECORDING] finalized segment=1 reason=silence duration=5.640s -----
Initializing AudioEmbedder (SpeechBrain)...
2026-09-09 05:52:06,045 | audio_match_attempt best_score=+0.5999 best_identity=binu matched_identity=None duration_sec=3.856 gallery_identities=1 invalid_embedding=false  
2026-09-09 05:52:06,045 | audio_match_attempt best_score=+0.5999 best_identity=binu matched_identity=None duration_sec=3.856 gallery_identities=1 invalid_embedding=false  

[2026-09-09 05:52:06] [MATCH UNENROLLED]: unenrolled_49 (no face) (Highest Score: 0.600 | Speaker: SPEAKER_UNKNOWN | Duration: 3.86s)
2026-09-09 05:52:06,089 | [PIPELINE] session=session_001 recording=1 unknown=unenrolled_49 (needs face enrollment)
2026-09-09 05:52:06,090 | [PIPELINE] trigger=unknown_speaker session=session_001 target=49 action=start_subprocess
2026-09-09 05:52:06,657 | [WHISPER] loading model=base device=cpu compute_type=int8
2026-09-09 05:52:06,658 | [WHISPER] loading model=base device=cpu compute_type=int8
2026-09-09 05:52:08,709 | ----- [RECORDING] started segment=2 pre_roll=0.500s -----
2026-09-09 05:52:13,232 | [WHISPER] "Talking to you right now?" (speaker_unknown_002.wav)
2026-09-09 05:52:13,235 | [WHISPER] "Hello, this is me." (speaker_unknown_001.wav)
2026-09-09 05:52:13,258 | [TRANSCRIPT] unenrolled_49: "Hello, this is me."

>>> [TRANSCRIPT] unenrolled_49: "Hello, this is me." <<< 

2026-09-09 05:52:13,258 | [TRANSCRIPT] unenrolled_49: "Talking to you right now?"

>>> [TRANSCRIPT] unenrolled_49: "Talking to you right now?" <<<

2026-09-09 05:52:13,269 | [PIPELINE] Recording #1 processed in 9.72s (segments=2)
2026-09-09 05:52:15,027 | ----- [RECORDING] finalized segment=2 reason=silence duration=6.820s -----
[STATE] Transition PipelineState.IDLE -> PipelineState.TRIGGERED
[STATE] Transition PipelineState.TRIGGERED -> PipelineState.CAPTURING
2026-09-09 05:52:15,687 | audio_match_attempt best_score=+0.4686 best_identity=binu matched_identity=None duration_sec=3.856 gallery_identities=1 invalid_embedding=false  
2026-09-09 05:52:15,688 | audio_match_attempt best_score=+0.4686 best_identity=binu matched_identity=None duration_sec=3.856 gallery_identities=1 invalid_embedding=false  

[2026-09-09 05:52:15] [SESSION UNKNOWN]: 'unenrolled_49' (Score: 0.674 | Speaker: SPEAKER_UNKNOWN | Duration: 3.86s | Via: session_embedding)
2026-09-09 05:52:15,690 | [PIPELINE] session=session_001 recording=2 unknown=unenrolled_49 (needs face enrollment)
Applied providers: ['CPUExecutionProvider'], with options: {'CPUExecutionProvider': {}}
find model: C:\Users\ASUS/.insightface\models\buffalo_l\1k3d68.onnx landmark_3d_68 ['None', 3, 192, 192] 0.0 1.0  
Applied providers: ['CPUExecutionProvider'], with options: {'CPUExecutionProvider': {}}
find model: C:\Users\ASUS/.insightface\models\buffalo_l\2d106det.onnx landmark_2d_106 ['None', 3, 192, 192] 0.0 1.0
Applied providers: ['CPUExecutionProvider'], with options: {'CPUExecutionProvider': {}}
find model: C:\Users\ASUS/.insightface\models\buffalo_l\det_10g.onnx detection [1, 3, '?', '?'] 127.5 128.0       
Applied providers: ['CPUExecutionProvider'], with options: {'CPUExecutionProvider': {}}
find model: C:\Users\ASUS/.insightface\models\buffalo_l\genderage.onnx genderage ['None', 3, 96, 96] 0.0 1.0      
Applied providers: ['CPUExecutionProvider'], with options: {'CPUExecutionProvider': {}}
find model: C:\Users\ASUS/.insightface\models\buffalo_l\w600k_r50.onnx recognition ['None', 3, 112, 112] 127.5 127.5
set det-size: (640, 640)
2026-09-09 05:52:19,778 | [WHISPER] "Is that what you mean?" (speaker_unknown_002.wav)
2026-09-09 05:52:20,011 | [WHISPER] "Ok now you can't even detect me." (speaker_unknown_001.wav)
2026-09-09 05:52:20,011 | [TRANSCRIPT] unenrolled_49: "Ok now you can't even detect me."

>>> [TRANSCRIPT] unenrolled_49: "Ok now you can't even detect me." <<<

2026-09-09 05:52:20,012 | [TRANSCRIPT] unenrolled_49: "Is that what you mean?"

>>> [TRANSCRIPT] unenrolled_49: "Is that what you mean?" <<<

2026-09-09 05:52:20,020 | [PIPELINE] Recording #2 processed in 4.99s (segments=2)
Recording 10.0s synchronized video: video='USB2.0 HD UVC WebCam', audio='Microphone Array (Realtek(R) Audio)'...  
2026-09-09 05:52:23,429 | ----- [RECORDING] started segment=3 pre_roll=0.500s -----
2026-09-09 05:52:30,145 | ----- [RECORDING] finalized segment=3 reason=silence duration=7.200s -----
2026-09-09 05:52:31,241 | audio_match_attempt best_score=+0.5707 best_identity=binu matched_identity=None duration_sec=5.488 gallery_identities=1 invalid_embedding=false  
2026-09-09 05:52:31,243 | audio_match_attempt best_score=+0.5707 best_identity=binu matched_identity=None duration_sec=5.488 gallery_identities=1 invalid_embedding=false  

[2026-09-09 05:52:31] [SESSION UNKNOWN]: 'unenrolled_49' (Score: 0.664 | Speaker: SPEAKER_UNKNOWN | Duration: 5.49s | Via: session_embedding)
2026-09-09 05:52:31,246 | [PIPELINE] session=session_001 recording=3 unknown=unenrolled_49 (needs face enrollment)
Video recorded successfully: C:\Users\ASUS\OneDrive\Desktop\binu\reconnect\backend\uploads\activity_video.mp4 (6492 KB)
[STATE] Transition PipelineState.CAPTURING -> PipelineState.PROCESSING
Running Who Is This on the first second of the video...  
2026-09-09 05:52:31,787 | ----- [RECORDING] started segment=4 pre_roll=0.500s -----
2026-09-09 05:52:33,665 | ----- [RECORDING] finalized segment=4 reason=silence duration=2.640s -----
2026-09-09 05:52:34,090 | audio_match_attempt best_score=+0.3026 best_identity=binu matched_identity=None duration_sec=1.000 gallery_identities=1 invalid_embedding=false  
2026-09-09 05:52:34,091 | audio_match_attempt best_score=+0.3026 best_identity=binu matched_identity=None duration_sec=1.000 gallery_identities=1 invalid_embedding=false  

[2026-09-09 05:52:34] [INSUFFICIENT AUDIO]: no unenrolled persistence (Highest Score: 0.303 | Best: binu | Speaker: SPEAKER_UNKNOWN | Duration: 1.00s)
2026-09-09 05:52:38,655 | [WHISPER] "and decide who I am and who knows me as a binu." (speaker_unknown_002.wav)   
2026-09-09 05:52:39,513 | [WHISPER] "Okay, I hope the camera..." (speaker_unknown_001.wav)
2026-09-09 05:52:39,513 | [TRANSCRIPT] unenrolled_49: "Okay, I hope the camera..."

>>> [TRANSCRIPT] unenrolled_49: "Okay, I hope the camera..." <<<

2026-09-09 05:52:39,515 | [TRANSCRIPT] unenrolled_49: "and decide who I am and who knows me as a binu."

>>> [TRANSCRIPT] unenrolled_49: "and decide who I am and who knows me as a binu." <<<

2026-09-09 05:52:39,521 | [PIPELINE] Recording #3 processed in 9.38s (segments=2)
2026-09-09 05:52:40,768 | [WHISPER] "I'm talking" (speaker_unknown_001.wav)
2026-09-09 05:52:40,769 | [TRANSCRIPT] speaker_unknown: "I'm talking"

>>> [TRANSCRIPT] speaker_unknown: "I'm talking" <<<      

2026-09-09 05:52:40,777 | [PIPELINE] Recording #4 processed in 7.11s (segments=1)
Captured 8 face observations.
[unknown_045705c3] Saved new unknown face to unenrolled_identities (id=56) in SQLite.

==================================================       
VISION RESULT
==================================================       
There is a new person on the center.
  Entity ID: unknown_045705c3

==================================================       
NEW UNKNOWN DETECTED
==================================================       
Entity ID: unknown_045705c3
Best face image: C:\Users\ASUS\OneDrive\Desktop\binu\reconnect\backend\data\unknown\unknown_045705c3\best_face.jpg

Non-interactive mode: keeping person as unknown.
Running Active Speaker Detection pipeline...
Running Active Speaker Detection pipeline...
[Step 1] Extracting audio from uploads\activity_video.mp4...
  Audio extracted -> C:\Users\ASUS\AppData\Local\Temp\extracted_audio_bh7amwg1.wav

============================================================
RECONNECT — Active Speaker Detection Pipeline
============================================================

[Step 1] Extracting frames at 2fps...
  Extracted 25 frames from uploads\activity_video.mp4
Applied providers: ['CPUExecutionProvider'], with options: {'CPUExecutionProvider': {}}
find model: C:\Users\ASUS/.insightface\models\buffalo_l\1k3d68.onnx landmark_3d_68 ['None', 3, 192, 192] 0.0 1.0  
Applied providers: ['CPUExecutionProvider'], with options: {'CPUExecutionProvider': {}}
find model: C:\Users\ASUS/.insightface\models\buffalo_l\2d106det.onnx landmark_2d_106 ['None', 3, 192, 192] 0.0 1.0
Applied providers: ['CPUExecutionProvider'], with options: {'CPUExecutionProvider': {}}
find model: C:\Users\ASUS/.insightface\models\buffalo_l\det_10g.onnx detection [1, 3, '?', '?'] 127.5 128.0       
Applied providers: ['CPUExecutionProvider'], with options: {'CPUfind model: C:\Users\ASUS/.insightface\models\buffalo_l\genderage.onnx genderage ['None', 3, 96, 96] 0.0 1.0
Applied providers: ['CPUExecutionProvider'], with options: {'CPUExecutionProvider': {}}
find model: C:\Users\ASUS/.insightface\models\buffalo_l\w600k_r50.onnx recognition ['None', 3, 112, 112] 127.5 127.5
set det-size: (640, 640)
[Step 2] Reusing Who Is This face model...
[Step 2] Loading lightweight detection-only InsightFace instance...
Applied providers: ['CPUExecutionProvider'], with options: {'CPUExecutionProvider': {}}
model ignore: C:\Users\ASUS/.insightface\models\buffalo_l\1k3d68.onnx landmark_3d_68
Applied providers: ['CPUExecutionProvider'], with options: {'CPUExecutionProvider': {}}
model ignore: C:\Users\ASUS/.insightface\models\buffalo_l\2d106det.onnx landmark_2d_106
Applied providers: ['CPUExecutionProvider'], with options: {'CPUExecutionProvider': {}}
find model: C:\Users\ASUS/.insightface\models\buffalo_l\det_10g.onnx detection [1, 3, '?', '?'] 127.5 128.0
Applied providers: ['CPUExecutionProvider'], with options: {'CPUExecutionProvider': {}}
model ignore: C:\Users\ASUS/.insightface\models\buffalo_l\genderage.onnx genderage
Applied providers: ['CPUExecutionProvider'], with options: {'CPUExecutionProvider': {}}
model ignore: C:\Users\ASUS/.insightface\models\buffalo_l\w600k_r50.onnx recognition
set det-size: (320, 320)
  InsightFace loaded. Known identities so far: ['binu']
[Step 2] Running face detection on 25 frames...
  Processed 10/25 frames
  Processed 20/25 frames
  Identities found across video: {'binu'}
[Step 3] Loading PyAnnote speaker diarisation model...
  PyAnnote loaded.
[Step 3] Running speaker diarisation on C:\Users\ASUS\AppData\Local\Temp\extracted_audio_bh7amwg1.wav...
  Found 2 speaker turns
  Unique speakers: {'SPEAKER_00'}
SPEAKER_00.01: 2.60s - 4.38s
SPEAKER_00.02: 4.84s - 7.83s
Saved diarised segment: SPEAKER_00.01.wav
Saved diarised segment: SPEAKER_00.02.wav
[Step 4] Loading ECAPA-TDNN speaker recognition model...        
  ECAPA-TDNN loaded. Known identities so far: ['binu']
[Step 4] Recognising speakers in 2 turns...
  Speaker recognition complete.
[Step 5] Loading TalkNet ASD model...
TalkNet device: cpu
09-09 05:53:26 Model para number = 15.01
  TalkNet loaded.
[Step 6] Matching 2 speaker turns...
      binu: detected in 15/15 frames (100% coverage)
  TalkNet input shapes: video=(1, 15, 112, 112), audio=(1, 60, 13)
      RAW logit score for binu: -1.546939730644226
      binu: detected in 24/24 frames (100% coverage)
  TalkNet input shapes: video=(1, 24, 112, 112), audio=(1, 96, 13)
      RAW logit score for binu: 0.5983286499977112
  ✓✓ [2.6s-4.4s] binu (DUAL_CONFIRMED) conf=0.39
      |-> PyAnnote SPEAKER_00 -> binu (avg ASD=0.18, observations=1)
      Face candidates:
        binu: 0.18
  ✓✓ [4.8s-7.8s] binu (DUAL_CONFIRMED) conf=0.83
      |-> PyAnnote SPEAKER_00 -> binu (avg ASD=0.41, observations=2)
      Face candidates:
        binu: 0.65
  [finalize check] SPEAKER_00: face=binu turns=2 wins=2 win_ratio=1.00 avg_conf=0.410

Final voice=face determinations:
  SPEAKER_00 -> binu (person_id=binu) CONFIRMED
[Debug Overlay] Writing to: C:\Users\ASUS\OneDrive\Desktop\binu\reconnect\backend\debug_overlay.mp4
[Debug Overlay] Input video: 1280x720 @ 7.55fps, 74 frames      
[Debug Overlay] Frames written: 74/74
[Debug Overlay] Frames with at least one box drawn: 74
[Debug Overlay] SUCCESS (video-only): C:\Users\ASUS\OneDrive\Desktop\binu\reconnect\backend\debug_overlay.mp4 (2892.3 KB)       
[Debug Overlay] SUCCESS (with audio): C:\Users\ASUS\OneDrive\Desktop\binu\reconnect\backend\debug_overlay_with_audio.mp4        

============================================================    
RESULTS SUMMARY
============================================================    
Total speaker turns:   2
  DUAL_CONFIRMED: 2

Unique speakers identified: {'binu'}

PyAnnote speaker -> face associations:
  SPEAKER_00 -> binu (confidence=0.41, evidence=2)

Full output saved to: asd_output.json

============================================================    
ACTIVE SPEAKER RESULTS
============================================================    
  [2.6s - 4.4s] binu (DUAL_CONFIRMED) conf=0.39
  [4.8s - 7.8s] binu (DUAL_CONFIRMED) conf=0.83

============================================================    
SPEAKER DETERMINATION
============================================================    

>>> unenrolled_49 is talking <<<


>>> binu is talking <<<

============================================================    


============================================================    
CROSS-MODAL VOICE-TO-FACE MATCHING
============================================================
  [VOICE MATCH] Turn [4.8s - 7.8s]: matches UNENROLLED voice 'unenrolled_49' (sim=0.515)
  [ENROLLMENT] Voice matched unenrolled_49, but face was not clearly visible or unconfirmed by ASD.
  [ENROLLMENT] No active speaker turn was linked to an unenrolled voice profile.
============================================================

[STATE] Reset to IDLE
Returning to listening...
