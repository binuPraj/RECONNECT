"""End-to-end RECONNECT audio perception pipeline."""

from datetime import datetime
import os
from pathlib import Path
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

from app.schemas.audio import (
    AudioInfo,
    AudioPerceptionResult,
    AudioSegment,
    PipelineInfo,
    PreprocessingInfo,
    StreamSegmentInfo,
)


PipelineProgressCallback = Callable[[str, str, dict[str, object]], None]


def _report(
    callback: PipelineProgressCallback | None,
    stage: str,
    status: str,
    **details: object,
) -> None:
    """Emit optional progress without coupling the offline pipeline to logs."""

    if callback is not None:
        callback(stage, status, details)


def run_audio_pipeline(
    original_path: Path,
    cleaned_path: Path,
    segments_output_dir: Path,
    session_id: str,
    original_filename: str | None = None,
    stream_segment: StreamSegmentInfo | None = None,
    session_noise_profile = None,
    progress_callback: PipelineProgressCallback | None = None,
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
        diarization_turns = diarize_audio(cleaned_audio=audio, sample_rate=sample_rate)
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
        UnknownAudioManager,
        load_audio_tensor,
    )
    global _embedder, _matcher, _unknown_manager, _aggregator, _session_memory
    if '_embedder' not in globals():
        _embedder = AudioEmbedder()
        _matcher = AudioMatcher()
        _unknown_manager = UnknownAudioManager()
        _aggregator = AudioAggregator()
    if '_session_memory' not in globals():
        _session_memory = SessionSpeakerMemory()

    segments = []
    _report(progress_callback, "recognition", "started")
    
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
    
    for speaker_label, spk_segments in speaker_groups.items():
        # Sort segments chronologically
        spk_segments = sorted(spk_segments, key=lambda s: s.start_sec)
        
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
            matched_id, score = _matcher.match(embedding)
            match_reason = "gallery"
            
            if matched_id:
                identity = matched_id
                status = "known"
                _session_memory.remember(
                    session_id, identity, embedding, confirmed=True
                )
                log_msg = (
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"✅ MATCH: '{identity}' (Score: {score:.3f} | "
                    f"Speaker: {speaker_label} | Duration: {duration:.2f}s | "
                    f"Via: {match_reason})"
                )
                print(f"\n{log_msg}")
            else:
                gallery_best_id, gallery_best_score = _matcher.best_match(embedding)
                continuity_id, continuity_score, continuity_reason = (
                    _session_memory.resolve(
                        session_id,
                        embedding,
                        gallery_best_id,
                        gallery_best_score,
                    )
                )
                if continuity_id:
                    identity = continuity_id
                    status = "known"
                    score = continuity_score
                    match_reason = continuity_reason or "session_continuity"
                    _session_memory.remember(
                        session_id, identity, embedding, confirmed=True
                    )
                    log_msg = (
                        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                        f"✅ MATCH: '{identity}' (Score: {score:.3f} | "
                        f"Speaker: {speaker_label} | Duration: {duration:.2f}s | "
                        f"Via: {match_reason})"
                    )
                    print(f"\n{log_msg}")
                else:
                    # Save concatenated audio for unknowns (longest representation)
                    identity, status = _unknown_manager.process_unknown(
                        embedding, concat_path
                    )
                    score = gallery_best_score
                    log_msg = (
                        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                        f"❌ NO MATCH: {identity} (Highest Score: {score:.3f} | "
                        f"Speaker: {speaker_label} | Duration: {duration:.2f}s)"
                    )
                    print(f"\n{log_msg}")
                    print("camera trigger")
                    
                    # Aggregation & Vision Trigger
                    is_confirmed = _aggregator.add_observation(identity)
                    if is_confirmed:
                        print(
                            f"VISION TRIGGER: Unknown speaker confirmed - "
                            f"{identity} ({status})"
                        )
                        _aggregator.reset()
                    
            with open(log_file_path, "a") as f:
                f.write(log_msg + "\n")
                
            speaker_identities[speaker_label] = {"identity": identity, "status": status}
            
        except Exception as e:
            print(f"Error during recognition for {speaker_label}: {e}")
            speaker_identities[speaker_label] = {"identity": None, "status": None}
            
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
