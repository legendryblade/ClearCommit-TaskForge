"""ElevenLabs streaming TTS over WebSocket (pre-made voice, no cloning).

Uses the ``/v1/text-to-speech/{voice_id}/stream-input`` WebSocket endpoint.
The caller feeds text incrementally (tokens/words); audio bytes are yielded as
they are generated, enabling playback to begin before the full sentence is
translated. A pre-made ``voice_id`` is used (voice cloning is intentionally
removed in this low-latency design).
"""

from __future__ import annotations

import base64
import json
import os
import queue
import threading
from typing import Iterator, Optional

import websocket  # websocket-client
from websocket import WebSocketConnectionClosedException

from logger import get_logger

logger = get_logger("elevenlabs_ws")

WS_URL = "wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input"
DEFAULT_MODEL = "eleven_multilingual_v2"


class ElevenLabsStreamTTS:
    def __init__(
        self,
        voice_id: str,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        output_format: str = "pcm_24000",
        stability: float = 0.55,
        similarity_boost: float = 0.4,
        first_audio_timeout: float = 8.0,
    ):
        self.voice_id = voice_id
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise RuntimeError("ELEVENLABS_API_KEY is not set.")
        self.model = model
        self.output_format = output_format
        self.stability = stability
        self.similarity_boost = similarity_boost
        self.sample_rate = int(output_format.split("_")[1])
        self._first_audio_timeout = first_audio_timeout

    # ------------------------------------------------------------------ #
    def _connect(self) -> websocket.WebSocket:
        url = (
            f"{WS_URL.format(voice_id=self.voice_id)}"
            f"?model_id={self.model}&output_format={self.output_format}"
            f"&xi_api_key={self.api_key}"
        )
        ws = websocket.WebSocket()
        ws.connect(url, header=[f"xi-api-key: {self.api_key}"], timeout=30)
        return ws

    def stream(self, text_chunks: Iterator[str]) -> Iterator[bytes]:
        """Feed ``text_chunks`` (tokens/words) and yield PCM audio chunks."""
        ws = self._connect()
        audio_q: "queue.Queue[Optional[bytes]]" = queue.Queue()
        err_q: "queue.Queue[object]" = queue.Queue()

        def reader() -> None:
            got_audio = False
            try:
                while True:
                    try:
                        msg = ws.recv()
                    except WebSocketConnectionClosedException:
                        break
                    except Exception as exc:  # noqa: BLE001
                        err_q.put(exc)
                        break
                    if msg is None:
                        break
                    if isinstance(msg, (bytes, bytearray)):
                        msg = msg.decode("utf-8", errors="ignore")
                    msg = msg.strip()
                    if not msg:  # skip empty/keepalive frames
                        continue
                    try:
                        data = json.loads(msg)
                    except json.JSONDecodeError:
                        logger.debug("ElevenLabs WS: non-JSON frame ignored: %.80r", msg)
                        continue
                    # Audio frames arrive as {"audio": "<base64>", ...} (no "type").
                    if "audio" in data:
                        if data.get("audio") is not None:
                            audio_q.put(base64.b64decode(data["audio"]))
                            got_audio = True
                        # audio:null with isFinal:true marks end of stream.
                        if data.get("isFinal"):
                            break
                        continue
                    mtype = data.get("type")
                    if mtype == "error":
                        err_q.put(RuntimeError(data.get("message", "ElevenLabs WS error")))
                        break
                    # Surface server errors that use a different shape
                    # (e.g. {"error": ...} / {"detail": ...} / {"message": ...}).
                    for key in ("error", "detail", "message"):
                        val = data.get(key)
                        if isinstance(val, dict):
                            val = val.get("message") or str(val)
                        if val:
                            logger.warning("ElevenLabs WS server message: %s", val)
                            if key != "message":
                                err_q.put(RuntimeError(f"ElevenLabs WS error: {val}"))
                                break
                    logger.debug("ElevenLabs WS: control frame: %.120r", msg)
                    # "flush" / "started" markers are ignored.
            except Exception as exc:  # noqa: BLE001
                err_q.put(exc)
            finally:
                if not got_audio and err_q.empty():
                    err_q.put(RuntimeError("ElevenLabs returned no audio for the request"))
                audio_q.put(None)  # sentinel

        reader_thread = threading.Thread(target=reader, daemon=True)
        reader_thread.start()

        def _drain_pending() -> Iterator[bytes]:
            while True:
                try:
                    chunk = audio_q.get_nowait()
                except queue.Empty:
                    return
                if chunk is None:
                    # stream ended early: stop yielding, let the outer drain
                    # pick up the sentinel.
                    return
                yield chunk

        try:
            # Initialization (BOS) frame: configures the session. A low
            # chunk_length_schedule makes the server generate audio even for
            # short sentences instead of buffering until 120+ chars (which
            # caused silent "no audio" failures).
            ws.send(json.dumps({
                "text": " ",
                "voice_settings": {
                    "stability": self.stability,
                    "similarity_boost": self.similarity_boost,
                    "use_speaker_boost": True,
                },
                "generation_config": {
                    "chunk_length_schedule": [50, 100, 200],
                },
            }))
            chunks = list(text_chunks)
            last_idx = len(chunks) - 1
            for idx, chunk in enumerate(chunks):
                if not chunk:
                    continue
                # Flush at sentence boundaries so consecutive text_input
                # messages stream as one continuous utterance (not staccato);
                # the FINAL chunk always flushes so buffered text is voiced
                # even if the translation has no trailing punctuation.
                stripped = chunk.rstrip()
                flush = (idx == last_idx) or (
                    bool(stripped) and stripped[-1] in ".!?…:;"
                )
                ws.send(json.dumps({"type": "text_input", "text": chunk, "flush": flush}))
                yield from _drain_pending()
            ws.send(json.dumps({"type": "input_end"}))

            # Drain remaining audio until sentinel. Bound the wait for the
            # FIRST chunk so a hung server fails the job fast (the worker
            # moves on) instead of freezing the queue for 30s+.
            got_first = False
            while True:
                timeout = None if got_first else self._first_audio_timeout
                try:
                    item = audio_q.get(timeout=timeout)
                except queue.Empty:
                    raise TimeoutError(
                        f"ElevenLabs produced no audio within "
                        f"{self._first_audio_timeout:.0f}s — failing job."
                    ) from None
                if item is None:
                    break
                got_first = True
                yield item
        finally:
            try:
                ws.close()
            except Exception:  # noqa: BLE001
                pass
        if not err_q.empty():
            raise err_q.get()

    def synthesize(self, text: str) -> bytes:
        """Non-incremental convenience: return the full audio for ``text``.

        Retries once if the server returns no audio (transient WS failures),
        to keep the live interpreter resilient.
        """
        for attempt in range(2):
            try:
                parts = [b for b in self.stream(iter([text]))]
                return b"".join(parts)
            except (RuntimeError, TimeoutError) as exc:
                if attempt == 0 and ("no audio" in str(exc).lower()
                                     or isinstance(exc, TimeoutError)):
                    logger.warning("ElevenLabs no-audio, retrying once: %s", text[:60])
                    continue
                raise
        return b""  # pragma: no cover - defensive; retry loop always raises


def resolve_voice_id(api_key: Optional[str] = None,
                     preferred: Optional[str] = None) -> str:
    """Return ``preferred`` if set, else ``ELEVENLABS_VOICE_ID`` from the env,
    else the first voice in the account."""
    api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
    if preferred:
        return preferred
    env_id = os.environ.get("ELEVENLABS_VOICE_ID")
    if env_id:
        return env_id
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY is not set (cannot resolve voice id).")
    import requests

    resp = requests.get(
        "https://api.elevenlabs.io/v1/voices",
        headers={"xi-api-key": api_key},
        timeout=20,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Could not list ElevenLabs voices ({resp.status_code}).")
    voices = resp.json().get("voices", [])
    if not voices:
        raise RuntimeError("No ElevenLabs voices found in this account.")
    return voices[0]["voice_id"]
