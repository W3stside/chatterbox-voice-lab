"""Minimal HTTP TTS service mirroring supra's ``/api/tts`` contract.

    POST /tts     {text, length_scale?}   -> audio/ogg (Opus)
                  ?format=wav             -> audio/wav  (parity / size A-B)
    GET  /health                          -> {device, sample_rate, model_loaded, clone_ref}
    GET  /ui/                             -> the browser test page (static/)

Phone-side clause pipelining lives in ``static/test.html``; the server stays a
simple text->audio unit (one request = one synthesized line), exactly the shape
supra's ``/api/tts`` will take when this ports back.
"""

from __future__ import annotations

import io
import logging
import os
import wave
from typing import Optional

import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from engine import ChatterboxEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chatterbox_lab.server")

app = FastAPI(title="chatterbox-voice-lab")

_engine: Optional[ChatterboxEngine] = None


def _get_engine() -> ChatterboxEngine:
    global _engine
    if _engine is None:
        _engine = ChatterboxEngine()
    return _engine


@app.on_event("startup")
def _startup() -> None:
    # Warm the model at boot so the first /tts request isn't slow.
    try:
        _get_engine()
    except Exception as e:  # noqa: BLE001
        logger.error("engine load failed at startup: %r", e)


class TTSRequest(BaseModel):
    text: str
    length_scale: Optional[float] = None


def _collect_pcm(
    engine: ChatterboxEngine, text: str, length_scale: Optional[float]
) -> np.ndarray:
    chunks = list(engine.synthesize(text, length_scale))
    if not chunks:
        return np.zeros(0, dtype=np.int16)
    return np.concatenate(chunks)


def _encode_wav(pcm: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # int16
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def _encode_opus(pcm: np.ndarray, sr: int) -> bytes:
    """Encode int16 mono PCM to Opus in an Ogg container via PyAV (bundles ffmpeg)."""
    import av

    buf = io.BytesIO()
    container = av.open(buf, mode="w", format="ogg")
    stream = container.add_stream("libopus", rate=sr)
    stream.layout = "mono"
    frame = av.AudioFrame.from_ndarray(pcm.reshape(1, -1), format="s16", layout="mono")
    frame.sample_rate = sr
    for packet in stream.encode(frame):
        container.mux(packet)
    for packet in stream.encode(None):  # flush
        container.mux(packet)
    container.close()
    return buf.getvalue()


@app.get("/health")
def health() -> JSONResponse:
    loaded = _engine is not None
    return JSONResponse(
        {
            "model_loaded": loaded,
            "device": _engine.device if loaded else None,
            "sample_rate": _engine.sample_rate if loaded else None,
            "clone_ref": _engine.clone_ref if loaded else os.environ.get("CLONE_REF"),
        }
    )


@app.post("/tts")
def tts(req: TTSRequest, format: str = Query("opus")) -> Response:
    engine = _get_engine()
    pcm = _collect_pcm(engine, req.text, req.length_scale)
    if format == "wav":
        return Response(_encode_wav(pcm, engine.sample_rate), media_type="audio/wav")
    try:
        return Response(_encode_opus(pcm, engine.sample_rate), media_type="audio/ogg")
    except Exception as e:  # noqa: BLE001 — PyAV/ffmpeg missing -> fall back so the test still works
        logger.warning("opus encode failed (%r); returning wav", e)
        return Response(_encode_wav(pcm, engine.sample_rate), media_type="audio/wav")


# Browser test page. Mounted under /ui so it never shadows /tts or /health.
app.mount("/ui", StaticFiles(directory="static", html=True), name="ui")
