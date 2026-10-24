"""Deepgram Aura streaming text-to-speech backend (realtime).

Deepgram Aura (v1 Speak WebSocket) is a *true* low-latency streaming TTS:
  wss://api.deepgram.com/v1/speak?model=...&encoding=linear16&sample_rate=...

Client -> Server control messages:
    {"type": "Speak", "text": "..."}   # queue text
    {"type": "Flush"}                   # emit queued audio now
    {"type": "Close"}                   # end the session

Server -> Client:
    * raw binary frames = linear16 PCM audio (no JSON wrapper)
    * JSON control frames: {"type": "Flushed"}, {"type": "Metadata"},
      {"type": "Error", "message": "..."}, {"type": "Warning", ...}

Aura is *not* universal: it ships curated voices per language (English and a
growing set of others such as Spanish). For a target language Aura cannot
pronounce, callers should fall back to a multilingual backend (ElevenLabs).
"""

from __future__ import annotations

import json
import os
from typing import Iterable, Iterator, Optional

from config import DEEPGRAM_API_KEY, DEEPGRAM_TTS_MODELS, DEEPGRAM_TTS_MODEL
from logger import get_logger
from websocket import WebSocketConnectionClosedException, WebSocketTimeoutException

try:  # pragma: no cover - import guard
    import websocket  # websocket-client
except Exception:  # noqa: BLE001
    websocket = None  # type: ignore

logger = get_logger("deepgram_tts")

WS_URL = "wss://api.deepgram.com/v1/speak"


class DeepgramStreamTTS:
    """Streaming TTS via Deepgram Aura's Speak WebSocket.

    Mirrors the interface used by the streaming interpreter (``sample_rate`` and
    ``stream(text_chunks)`` / ``synthesize(text)``), so it is a drop-in
    replacement for :class:`ElevenLabsStreamTTS` on the realtime path.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        target_language: Optional[str] = None,
        encoding: str = "linear16",
        sample_rate: int = 24000,
    ):
        if websocket is None:
            raise RuntimeError("The 'websocket-client' package is required.")
        key = api_key or DEEPGRAM_API_KEY
        if not key:
            raise RuntimeError("DEEPGRAM_API_KEY is not set.")

        # Pick a voice the model actually ships for this language.
        if model:
            self.model = model
        elif target_language in DEEPGRAM_TTS_MODELS:
            self.model = DEEPGRAM_TTS_MODELS[target_language]
        else:
            self.model = DEEPGRAM_TTS_MODEL

        self.api_key = key
        self.encoding = encoding
        self.sample_rate = sample_rate
        self.output_format = f"{encoding}_{sample_rate}"

    # ------------------------------------------------------------------ #
    def stream(self, text_chunks: Iterable[str]) -> Iterator[bytes]:
        # NOTE: Deepgram's WebSocket is used single-threaded here. Concurrent
        # send/recv from separate threads starves the socket (the server never
        # answers), so we send each chunk + Flush and then drain that chunk's
        # audio inline until the "Flushed" control message arrives.
        url = (
            f"{WS_URL}?model={self.model}&encoding={self.encoding}"
            f"&sample_rate={self.sample_rate}"
        )
        ws = websocket.WebSocket()
        ws.connect(url, header=[f"Authorization: Token {self.api_key}"], timeout=30)
        logger.info("Deepgram WS connected: model=%s rate=%d", self.model, self.sample_rate)

        got_audio = False
        try:
            for chunk in text_chunks:
                if not chunk:
                    continue
                ws.send(json.dumps({"type": "Speak", "text": chunk}))
                ws.send(json.dumps({"type": "Flush"}))

                # Drain this chunk's audio until the server signals "Flushed".
                while True:
                    try:
                        msg = ws.recv()
                    except WebSocketTimeoutException:
                        raise RuntimeError(
                            "Deepgram TTS timed out waiting for audio."
                        )
                    except WebSocketConnectionClosedException:
                        break
                    if msg is None:
                        break
                    if isinstance(msg, (bytes, bytearray)):
                        yield bytes(msg)
                        got_audio = True
                        continue
                    # Text frame = JSON control message.
                    try:
                        data = json.loads(msg)
                    except json.JSONDecodeError:
                        continue
                    mtype = data.get("type")
                    if mtype == "Flushed":
                        break
                    if mtype == "Error":
                        raise RuntimeError(
                            data.get("message", "Deepgram TTS error")
                        )
                    # "Metadata" / "Warning" are ignored.
        finally:
            try:
                ws.send(json.dumps({"type": "Close"}))
            except Exception:  # noqa: BLE001
                pass
            try:
                ws.close()
            except Exception:  # noqa: BLE001
                pass

        if not got_audio:
            raise RuntimeError("Deepgram returned no audio for the request")

    # ------------------------------------------------------------------ #
    def synthesize(self, text: str) -> bytes:
        if not text.strip():
            raise ValueError("Cannot synthesize empty text.")
        return b"".join(self.stream(iter([text])))
