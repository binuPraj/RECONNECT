"""End-to-end RECONNECT audio perception pipeline."""

import logging
import traceback
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from collections.abc import Callable
import torch
import torchaudio

# Patch torchaudio for compatibility with speechbrain (required by Pyannote/ECAPA-TDNN)
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['soundfile']
if not hasattr(torchaudio, 'io'):
    class DummyIO:
        class StreamReader:
            pass
    torchaudio.io = DummyIO()

from app.audio_process.preprocessing import preprocess_audio
from app.audio_process.diarization import (
    DIARIZATION_MODEL,
    diarize_audio,
)
from app.audio_process.segmentation import (
    fallback_vad_segments,
    merge_vad_and_diarization,
    export_segment_wavs,
)
from app.audio_process.vad import VAD_MODEL
from app.audio_process.identity_barrier import SessionIdentityResolutionBarrier

from app.schemas.audio import (
    AudioInfo,
    AudioPerceptionResult,
    AudioSegment,
    PipelineInfo,
    PreprocessingInfo,
    StreamSegmentInfo,
)
from database.db import get_identity_by_name


import unicodedata
from app.audio_process.transcribe import transcribe_wav

PipelineProgressCallback = Callable[[str, str, dict[str, object]], None]
IdentityCallback = Callable[[dict[str, object]], None]
MIN_UNENROLLED_FALLBACK_DURATION_SEC = 2.5


def validate_transcript_guard(text: str) -> tuple[bool, str]:
    """
    Validate transcript before persisting an unenrolled identity.
    Returns (True, "valid") if transcript has actual speech in expected scripts (Latin/Devanagari).
    Returns (False, "discarded_no_transcript") if empty or contains no letters.
    Returns (False, "discarded_bad_transcript") if letters belong to an unexpected script (e.g. Cyrillic mistranscriptions on noise).
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return False, "discarded_no_transcript"

    has_letters = False
    for ch in cleaned:
        if unicodedata.category(ch).startswith("L"):
            has_letters = True
            name = unicodedata.name(ch, "")
            if not ("LATIN" in name or "DEVANAGARI" in name):
                return False, "discarded_bad_transcript"

    if not has_letters:
        return False, "discarded_no_transcript"

    return True, "valid"


def _report(
    callback: PipelineProgressCallback | None,
    stage: str,
    status: str,
    **details: object,
) -> None:
    """Emit optional progress without coupling the offline pipeline to logs."""

    if callback is not None:
        callback(stage, status, details)


def _append_resolution_failure_log(
    log_path: Path,
    record: dict[str, object],
) -> None:
    """Persist a resolution failure for post-session threshold analysis."""
    with open(log_path, "a", encoding="utf-8") as failure_log:
        failure_log.write(json.dumps(record, sort_keys=True) + "\n")


def run_audio_pipeline(
    original_path: Path,
    cleaned_path: Path,
    segments_output_dir: Path,
    session_id: str,
    original_filename: str | None = None,
    stream_segment: StreamSegmentInfo | None = None,
    session_noise_profile = None,
    progress_callback: PipelineProgressCallback | None = None,
    identity_callback: IdentityCallback | None = None,
    identity_resolution_barrier: SessionIdentityResolutionBarrier | None = None,
) -> AudioPerceptionResult:
    """
    Run the complete RECONNECT audio perception pipeline.

    Pipeline:

        Original Audio
              ↓
        Load / Resample / Normalize
              ↓
        Silero VAD
              ↓
        Noise Profile
              ↓
        Noise Reduction
              ↓
        Full-Length Clean Audio
              ↓
        Pyannote Diarization
              ↓
        VAD + Diarization Merge
              ↓
        Segment WAV Export
              ↓
        AudioPerceptionResult
    """

    pipeline_start = time.perf_counter()

    # =========================================================
    # 1. PREPROCESSING
    # =========================================================

    _report(progress_callback, "preprocessing", "started")
    preprocessing = preprocess_audio(
        input_path=original_path,
        output_path=cleaned_path,
        session_noise_profile=session_noise_profile,
    )

    audio = preprocessing["audio"]

    sample_rate = preprocessing[
        "processed_sample_rate"
    ]

    duration_sec = preprocessing[
        "duration"
    ]

    vad_regions = preprocessing[
        "speech_regions"
    ]
    _report(
        progress_callback,
        "preprocessing",
        "completed",
        duration_sec=round(duration_sec, 3),
        speech_region_count=len(vad_regions),
        noise_profile_source=preprocessing["noise_profile_source"],
        noise_profile_duration_sec=round(preprocessing["noise_profile_duration_sec"], 3),
        noise_reduction_applied=preprocessing["noise_reduction_applied"],
    )

    # =========================================================
    # 2. DIARIZATION
    # =========================================================

    diarization_turns = []
    if duration_sec >= 10.0:
        _report(progress_callback, "diarization", "started")
        diarization_turns = diarize_audio(
            raw_resampled_audio=preprocessing["raw_resampled_audio"],
            sample_rate=sample_rate,
            min_speakers=2,
            max_speakers=6,
        )
        _report(progress_callback, "diarization", "completed", turn_count=len(diarization_turns))
    else:
        _report(progress_callback, "diarization", "skipped", reason="recording_under_10_seconds")

    # =========================================================
    # 3. MERGE VAD + DIARIZATION
    # =========================================================

    _report(progress_callback, "segmentation", "started")
    merged_segments = merge_vad_and_diarization(
        diarization_turns=diarization_turns, vad_regions=vad_regions,
        min_duration_sec=0.5, pad_sec=0.0, total_duration_sec=duration_sec,
    )
    fallback_reason = None
    if vad_regions and not merged_segments:
        fallback_reason = "recording_under_10_seconds" if duration_sec < 10.0 else "no_diarization_turns"
        merged_segments = fallback_vad_segments(vad_regions, min_duration_sec=0.5)
    _report(
        progress_callback,
        "segmentation",
        "completed",
        merged_segment_count=len(merged_segments),
        fallback_reason=fallback_reason,
    )

    # =========================================================
    # 4. EXPORT FINAL SEGMENT WAV FILES
    # =========================================================

    _report(progress_callback, "export", "started")
    exported_segments = export_segment_wavs(
        audio=audio,
        sample_rate=sample_rate,
        segments=merged_segments,
        output_dir=segments_output_dir,
    )
    _report(
        progress_callback,
        "export",
        "completed",
        final_segment_count=len(exported_segments),
    )

    # =========================================================
    # 5. RECOGNITION & SCHEMAS
    # =========================================================
    
    from app.audio_process.recognition import (
        AudioEmbedder,
        AudioMatcher,
        AudioAggregator,
        SessionSpeakerMemory,
        load_audio_tensor,
    )
    from database.db import (
        create_unenrolled_identity,
        update_unenrolled_voice,
        find_matching_unenrolled_voice,
    )
    global _embedder, _matcher, _aggregator, _session_memory, _triggered_unknowns
    if '_embedder' not in globals():
        _embedder = AudioEmbedder()
        _matcher = AudioMatcher()
        _aggregator = AudioAggregator()
        _triggered_unknowns = set()
    if '_session_memory' not in globals():
        _session_memory = SessionSpeakerMemory()

    segments = []
    _report(progress_callback, "recognition", "started")

    recording_id = stream_segment.main_segment_id if stream_segment is not None else None
    barrier_acquired = False
    if identity_resolution_barrier is not None and recording_id is not None:
        # Preprocessing, diarization, and export above stay concurrent. Only
        # recognition, memory mutation, and unknown persistence commit in
        # chronological order within this live session.
        identity_resolution_barrier.wait_for_turn(session_id, recording_id)
        barrier_acquired = True
    
    # 1. Group segments by speaker
    speaker_groups = {}
    for segment in exported_segments:
        if segment.speaker_label not in speaker_groups:
            speaker_groups[segment.speaker_label] = []
        speaker_groups[segment.speaker_label].append(segment)
        
    # 2. Process each speaker (concat clips, match gallery, then session continuity)
    speaker_identities = {}
    
    log_file_path = "data/match_logs.txt"
    from datetime import datetime
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    
    LOGGER = logging.getLogger(__name__)

    for speaker_label, spk_segments in speaker_groups.items():
        # Sort segments chronologically
        spk_segments = sorted(spk_segments, key=lambda s: s.start_sec)

        # Initialize per-iteration variables BEFORE the try block.
        # If these are only set inside branches and an exception fires
        # in audio-loading / embedding, we'd get a NameError caught by
        # the outer except — silently dropping the entire enrollment.
        identity: str | None = None
        status: str | None = None
        score: float = 0.0
        log_msg: str = ""

        try:
            # Concatenate all audio segments for this speaker
            wavs = []
            duration = 0.0
            for seg in spk_segments:
                signal, fs = load_audio_tensor(seg.audio_path)
                if fs != 16000:
                    resampler = torchaudio.transforms.Resample(fs, 16000)
                    signal = resampler(signal)
                wavs.append(signal)
                duration += (seg.end_sec - seg.start_sec)
                
            if not wavs:
                continue
                
            concatenated_signal = torch.cat(wavs, dim=1)
            concat_path = os.path.join(segments_output_dir, f"concat_{speaker_label}.wav")
            import soundfile as sf
            sf.write(
                concat_path,
                concatenated_signal.squeeze(0).cpu().numpy(),
                16000,
            )
            
            # Extract embedding from the glued audio
            embedding = _embedder.extract_embedding(concat_path)
            matched_id, score = _matcher.match(
                embedding, segment_duration_sec=duration
            )
            match_reason = "gallery"
            
            if matched_id:
                identity = matched_id
                status = "known"
                print(f"Known identity: {identity}")
                if identity_callback is not None:
                    identity_row = get_identity_by_name(identity)
                    identity_callback({
                        "known": True,
                        "name": identity,
                        "relation": identity_row["relation"] if identity_row else None,
                    })
                _session_memory.remember(
                    session_id, identity, embedding, confirmed=True
                )
                log_msg = (
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"[MATCH]: '{identity}' (Score: {score:.3f} | "
                    f"Speaker: {speaker_label} | Duration: {duration:.2f}s | "
                    f"Via: {match_reason})"
                )
                print(f"\n{log_msg}")
            else:
                gallery_best_id, gallery_best_score = _matcher.best_match(
                    embedding, segment_duration_sec=duration
                )
                # Only use session continuity to re-identify ENROLLED speakers
                # (e.g. binu dipping just below gallery threshold on a short clip).
                # Unknown/unenrolled speakers must NEVER be short-circuited via
                # session continuity — each diarization label could be a different
                # person, and we want them each enrolled separately.
                enrolled_continuity_id = None
                continuity_details: dict[str, object] = {
                    "gallery_best_previously_confirmed": False,
                    "max_session_embedding_score": None,
                    "max_session_embedding_identity": None,
                }
                if gallery_best_id is not None and not (gallery_best_id or "").startswith("unenrolled_"):
                    enrolled_continuity_id, continuity_score, continuity_reason, continuity_details = (
                        _session_memory.resolve_with_details(
                            session_id,
                            embedding,
                            gallery_best_id,
                            gallery_best_score,
                        )
                    )
                if enrolled_continuity_id is not None:
                    identity = enrolled_continuity_id
                    is_unenrolled = False
                    status = "known"
                    score = continuity_score
                    match_reason = continuity_reason or "session_continuity"
                    _session_memory.remember(
                        session_id, identity, embedding, confirmed=True
                    )
                    log_msg = (
                        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                        f"[MATCH]: '{identity}' (Score: {score:.3f} | "
                        f"Speaker: {speaker_label} | Duration: {duration:.2f}s | "
                        f"Via: {match_reason})"
                    )
                    print(f"\n{log_msg}")
                    if identity_callback is not None:
                        identity_row = get_identity_by_name(identity)
                        identity_callback({
                            "known": True,
                            "name": identity,
                            "relation": identity_row["relation"] if identity_row else None,
                        })
                else:
                    session_root = segments_output_dir.parent.parent
                    # Every unknown speaker turn is enrolled into the unenrolled table.
                    # Attempt to match against existing unenrolled voice embeddings first.
                    # Use a higher threshold (0.62) so different people in the same
                    # room don't collapse into the same unenrolled_ID.
                    match_row, match_score = find_matching_unenrolled_voice(
                        embedding, threshold=0.62
                    )
                    if match_row is not None:
                        un_id = match_row["id"]
                        identity = f"unenrolled_{un_id}"
                        has_face = match_row.get("face_embedding") is not None
                        status = "unenrolled_with_face" if has_face else "unenrolled"
                        update_unenrolled_voice(un_id, embedding)
                        _session_memory.remember(
                            session_id, identity, embedding, confirmed=False
                        )
                        score = gallery_best_score
                        face_note = (
                            " (face linked)"
                            if status == "unenrolled_with_face"
                            else " (no face)"
                        )
                        log_msg = (
                            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                            f"[MATCH UNENROLLED]: {identity}{face_note} "
                            f"(Highest Score: {score:.3f} | Speaker: {speaker_label} | "
                            f"Duration: {duration:.2f}s)"
                        )
                        if identity_callback is not None:
                            identity_callback({
                                "known": False,
                                "speaker": identity,
                                "unenrolled_id": un_id,
                                "has_face": status == "unenrolled_with_face",
                            })
                    else:
                        new_id = create_unenrolled_identity(voice_embedding=embedding)
                        identity = f"unenrolled_{new_id}"
                        un_id = new_id
                        status = "new_unenrolled"

                        _session_memory.remember(
                            session_id, identity, embedding, confirmed=False
                        )
                        score = gallery_best_score
                        log_msg = (
                            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                            f"[NEW UNENROLLED]: {identity} (no face) "
                            f"(Highest Score: {score:.3f} | Speaker: {speaker_label} | "
                            f"Duration: {duration:.2f}s)"
                        )
                        if identity_callback is not None:
                            identity_callback({
                                "known": False,
                                "speaker": identity,
                                "unenrolled_id": un_id,
                                "has_face": False,
                            })

                    _append_resolution_failure_log(
                        session_root / "identity_resolution_failures.jsonl",
                        {
                            "event": "session_resolution_failed",
                            "timestamp": datetime.now().astimezone().isoformat(),
                            "session_id": session_id,
                            "recording_id": recording_id,
                            "main_segment_id": recording_id,
                            "candidate_duration_sec": round(duration, 3),
                            "gallery_best_identity": gallery_best_id,
                            "gallery_best_score": gallery_best_score,
                            **continuity_details,
                            "final_outcome": (
                                "matched_existing_unenrolled"
                                if status != "new_unenrolled"
                                else "created_new_unenrolled"
                            ),
                            "final_identity": identity,
                        },
                    )
                    print(f"\n{log_msg}")

            if log_msg:
                with open(log_file_path, "a", encoding="utf-8") as f:
                    f.write(log_msg + "\n")

            speaker_identities[speaker_label] = {"identity": identity, "status": status}

        except Exception as e:
            tb = traceback.format_exc()
            LOGGER.error(
                "[PIPELINE] Recognition FAILED for speaker %s — identity will be None.\n"
                "  Error: %s\n%s",
                speaker_label, e, tb,
            )
            print(
                f"[PIPELINE ERROR] Recognition failed for {speaker_label}: {e}\n{tb}"
            )
            speaker_identities[speaker_label] = {"identity": None, "status": None}

    if barrier_acquired:
        identity_resolution_barrier.complete(session_id, recording_id)

    # 3. Apply to all segments
    for segment in exported_segments:
        identity_info = speaker_identities.get(segment.speaker_label, {"identity": None, "status": None})
        segments.append(
            AudioSegment(
                segment_id=segment.segment_id,
                start_sec=segment.start_sec,
                end_sec=segment.end_sec,
                speaker_label=segment.speaker_label,
                speaker_source=segment.speaker_source,
                audio_path=segment.audio_path,
                vad_confidence=segment.vad_confidence,
                speaker_identity=identity_info["identity"],
                match_status=identity_info["status"],
            )
        )
        
    _report(progress_callback, "recognition", "completed")

    # =========================================================
    # 6. COUNT SPEAKERS
    # =========================================================

    speaker_labels = {
        segment.speaker_label
        for segment in segments
    }

    # =========================================================
    # 7. PROCESSING TIME
    # =========================================================

    processing_time_sec = round(
        time.perf_counter() - pipeline_start,
        3,
    )

    # =========================================================
    # 8. BUILD FINAL SCHEMA
    # =========================================================

    result = AudioPerceptionResult(
        saved=True,

        session_id=session_id,

        recording_started_at=(
            datetime.now().astimezone().isoformat()
        ),

        duration_sec=round(
            duration_sec,
            3,
        ),

        original_filename=original_filename,

        stream_segment=stream_segment,

        audio=AudioInfo(
            original_path=str(
                original_path
            ),

            cleaned_path=str(
                cleaned_path
            ),

            sample_rate=sample_rate,
        ),

        preprocessing=PreprocessingInfo(
            original_sample_rate=(
                preprocessing[
                    "original_sample_rate"
                ]
            ),

            noise_profile_sec=preprocessing["noise_profile_duration_sec"],
            noise_profile_source=preprocessing["noise_profile_source"],
            noise_reduction_applied=preprocessing["noise_reduction_applied"],
            raw_rms=preprocessing["raw_rms"],
            cleaned_rms=preprocessing["cleaned_rms"],

            peak_normalized=(
                preprocessing[
                    "target_peak"
                ]
            ),

            noise_reduction_mode="vad_regions",

            preprocessing_completed=(
                preprocessing[
                    "preprocessing_completed"
                ]
            ),
        ),

        segments=segments,

        pipeline=PipelineInfo(
            vad_model="Silero VAD",

            diarization_model=(
                DIARIZATION_MODEL
            ),

            processing_time_sec=(
                processing_time_sec
            ),
        ),

        speaker_count=len(
            speaker_labels
        ),

        segment_count=len(
            segments
        ),
    )

    _report(
        progress_callback,
        "pipeline",
        "completed",
        processing_time_sec=processing_time_sec,
    )
    return result
