"""RECONNECT live audio WebSocket API."""

import asyncio
import json
import logging
import os
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from app.audio_stream.format_adapter import StreamAudioFormat, adapt_pcm16_stream_chunk
from app.audio_stream.segment_processor import StreamingSegmentProcessor
from app.audio_stream.streaming_recorder import StreamingSpeechRecorder
from app.utils.storage import allocate_stream_session_id


logging.basicConfig(
    level=getattr(logging, os.getenv("RECONNECT_LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

for _noisy in ("faster_whisper", "speechbrain", "pyannote", "urllib3", "httpx", "huggingface_hub", "onnxruntime"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

app = FastAPI(title="RECONNECT Audio Streaming API", version="1.1.0")


@app.get("/")
async def root():
    return {"service": "RECONNECT Audio Streaming API", "status": "running"}


async def _receive_stream_format(websocket: WebSocket) -> StreamAudioFormat:
    """Read and validate the required first JSON WebSocket message."""

    message = await websocket.receive()
    raw_text = message.get("text")
    if raw_text is None:
        raise ValueError("First WebSocket message must be JSON format metadata")
    try:
        metadata = json.loads(raw_text)
    except json.JSONDecodeError as error:
        raise ValueError("Format metadata must be valid JSON") from error
    if not isinstance(metadata, dict):
        raise ValueError("Format metadata must be a JSON object")
    return StreamAudioFormat.from_message(metadata)


@app.websocket("/ws/audio")
async def audio_stream(websocket: WebSocket):
    """Receive binary PCM audio and process finalized speech recordings."""

    await websocket.accept()
    session_id = allocate_stream_session_id()
    loop = asyncio.get_running_loop()
    identity_events: asyncio.Queue[dict[str, object]] = asyncio.Queue(maxsize=50)

    def on_identity_result(result: dict[str, object]) -> None:
        def enqueue_result() -> None:
            try:
                identity_events.put_nowait(result)
            except asyncio.QueueFull:
                pass

        loop.call_soon_threadsafe(enqueue_result)

    processor = StreamingSegmentProcessor(
        session_id,
        identity_callback=on_identity_result,
    )
    recorder = StreamingSpeechRecorder(
        on_finalized=processor.submit,
        on_confirmed_idle_audio=processor.add_confirmed_idle_audio,
    )
    chunk_count = source_bytes = canonical_bytes = 0
    last_progress_at = time.monotonic()

    try:
        try:
            source_format = await _receive_stream_format(websocket)
        except ValueError as error:
            LOGGER.warning(
                "[STREAM] format=rejected session=%s error=%s",
                session_id,
                error,
            )
            await websocket.send_json({"received": False, "error": str(error)})
            await websocket.close(code=1003)
            return

        LOGGER.info(
            "[STREAM] format=accepted session=%s source=%sHz/%sch/%s normalization=%s",
            session_id,
            source_format.sample_rate,
            source_format.channels,
            source_format.encoding,
            "passthrough" if source_format.is_canonical else "enabled",
        )
        await websocket.send_json({"received": True, "session_id": session_id})

        while True:
            message = await websocket.receive()
            if message.get("text") is not None:
                await websocket.send_json(
                    {"received": False, "error": "Audio format cannot change after stream start"}
                )
                await websocket.close(code=1003)
                return
            data = message.get("bytes")
            if data is None:
                raise WebSocketDisconnect(code=1000)
            try:
                canonical_data = adapt_pcm16_stream_chunk(data, source_format)
            except ValueError as error:
                LOGGER.warning(
                    "[STREAM] normalization=rejected session=%s error=%s",
                    session_id,
                    error,
                )
                await websocket.send_json({"received": False, "error": str(error)})
                await websocket.close(code=1003)
                return

            recorder.process(canonical_data)
            chunk_count += 1
            source_bytes += len(data)
            canonical_bytes += len(canonical_data)
            LOGGER.debug(
                "[STREAM] normalization=processed session=%s source_bytes=%s canonical_bytes=%s",
                session_id,
                len(data),
                len(canonical_data),
            )
            now = time.monotonic()
            if now - last_progress_at >= 15.0:
                status = recorder.status_snapshot()
                LOGGER.debug(
                    "[STREAM] progress session=%s chunks=%s source_bytes=%s "
                    "canonical_bytes=%s temp_buffer=%s/%.3fs vad=%s frames=%s "
                    "probability=%s recording=%s main_duration=%.3fs",
                    session_id,
                    chunk_count,
                    source_bytes,
                    canonical_bytes,
                    status["temp_buffer_state"],
                    status["temp_buffer_duration_sec"],
                    status["vad_state"],
                    status["vad_frames"],
                    status["last_probability"],
                    status["active_segment_id"],
                    status["main_duration_sec"],
                )
                last_progress_at = now
            try:
                identity_result = identity_events.get_nowait()
            except asyncio.QueueEmpty:
                identity_result = None

            acknowledgement = {
                "received": True,
                "chunk_number": chunk_count,
                "chunk_size": len(data),
            }
            if identity_result is not None:
                acknowledgement["identity_result"] = identity_result

            await websocket.send_json(acknowledgement)

    except WebSocketDisconnect:
        LOGGER.info("[STREAM] disconnected session=%s", session_id)
    finally:
        recorder.finish()
        await processor.drain()
        LOGGER.info(
            "[STREAM] closed session=%s chunks=%s source_bytes=%s canonical_bytes=%s",
            session_id,
            chunk_count,
            source_bytes,
            canonical_bytes,
        )
