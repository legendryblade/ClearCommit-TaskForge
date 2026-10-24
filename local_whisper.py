"""Local offline Whisper STT (TaskForge).

Drop-in replacement for the cloud Transcriber: same
``transcribe(audio_path, language=None) -> TranscriptionResult`` interface.
Runs OpenAI Whisper locally via faster-whisper (CTranslate2), so speech
recognition works with no API key, no network, and full privacy.

The model is loaded lazily on first use; missing dependencies degrade to a
clear error instead of a crash.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

try:
    from logger import get_logger

    logger = get_logger("local_whisper")
except Exception:  # noqa: BLE001
    import logging

    logger = logging.getLogger("local_whisper")


@runtime_checkable
class _TranscriptionResultLike(Protocol):
    text: str
    language: str


class LocalWhisperTranscriber:
    """Offline speech recognition with OpenAI Whisper.

    Matches the cloud transcriber's surface: ``audio_path`` (a WAV file
    written by the pipeline) and an optional ISO-639-1 language code. When
    the language is omitted, Whisper multilingual auto-detects it.
    """

    def __init__(self, model_size: str = "base", device: str = "auto",
                 compute_type: str = "auto"):
        # Imported defensively: a fresh machine may not have faster-whisper.
        try:
            from faster_whisper import WhisperModel
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Local Whisper needs the 'faster-whisper' package "
                "(pip install faster-whisper). The Cloud recognizer is the "
                "fallback when you cannot install it."
            ) from exc
        self._model_size = model_size
        self._device = None if device == "auto" else device
        self._compute_type = None if compute_type == "auto" else compute_type
        logger.info("Loading local Whisper model '%s' (first run downloads "
                    "the model)…", model_size)
        self._model = WhisperModel(model_size,
                                   device=self._device or "cpu",
                                   compute_type=self._compute_type or "int8")

    def transcribe(self, audio_path, language: Optional[str] = None):
        from transcriber import TranscriptionResult

        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        lang = language if language else None
        segments, info = self._model.transcribe(
            str(path), language=lang, vad_filter=True,
            beam_size=1, best_of=1)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        detected = getattr(info, "language", None) or lang or "unknown"
        if not text:
            raise RuntimeError("Local transcription returned empty text.")
        logger.info("Local Whisper (%s): %s", detected, text[:80])
        return TranscriptionResult(text=text, language=detected,
                                   duration=None)
