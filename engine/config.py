"""Configuration and shared constants for the voice-to-voice translation system.

All secrets are pulled from environment variables (optionally loaded from a
local ``.env`` file). Nothing sensitive is ever hard-coded.
"""

from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# App data home for frozen builds: %APPDATA%\LiveInterpreter
APP_DIR_NAME = "LiveInterpreter"


def app_data_dir() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home())
    return Path(base) / APP_DIR_NAME

# ---------------------------------------------------------------------------
# Optional .env support
# ---------------------------------------------------------------------------
# python-dotenv is *optional*. If it is installed we load a local .env file so
# developers can keep secrets out of their shell environment. If it is missing
# the system still works as long as the variables are exported in the shell.
#
# Frozen builds (PyInstaller) can't rely on a .env next to the exe — the
# user-writable home for keys is %APPDATA%\LiveInterpreter\.env instead.
try:  # pragma: no cover - environment dependent
    import os as _os
    from pathlib import Path as _Path

    from dotenv import load_dotenv

    load_dotenv()  # developer flow: repo/.env
    if getattr(sys, "frozen", False):  # installed flow
        _appdata_env = app_data_dir() / ".env"
        if _appdata_env.exists():
            load_dotenv(_appdata_env, override=True)
except Exception:  # noqa: BLE001 - never fail on a missing optional dep
    pass


# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------
GROQ_TRANSCRIBE_MODEL = "whisper-large-v3"
GROQ_TRANSCRIBE_MODEL_FAST = "whisper-large-v3-turbo"
GROQ_TRANSLATE_MODEL = "openai/gpt-oss-120b"
GROQ_TRANSLATE_MODEL_FAST = "openai/gpt-oss-20b"
GROQ_TRANSLATE_MODELS = {
    "accurate": GROQ_TRANSLATE_MODEL,   # gpt-oss-120b
    "fast": GROQ_TRANSLATE_MODEL_FAST,  # gpt-oss-20b
}
ELEVENLABS_MODEL = "eleven_multilingual_v2"

# Groq transcription hard limit for uploaded files (bytes).
GROQ_MAX_FILE_BYTES = 25 * 1024 * 1024  # 25 MB

# ElevenLabs voice-cloning guidance.
ELEVENLABS_MIN_CLONE_SECONDS = 10  # recommended minimum for a usable clone

# ---------------------------------------------------------------------------
# TTS backend selection
# ---------------------------------------------------------------------------
# "groq"      -> Groq Orpheus TTS (preset voices; English/Arabic only; no cloning)
# "elevenlabs"-> ElevenLabs eleven_multilingual_v2 (clones the user's voice)
TTS_BACKEND = os.environ.get("TTS_BACKEND", "groq").lower()

# ---------------------------------------------------------------------------
# Realtime TTS backend (streaming interpreter)
# ---------------------------------------------------------------------------
# Groq Orpheus TTS is request/response only (no streaming WebSocket, 200-char
# cap), so it is NOT usable for realtime. The realtime path therefore uses
# Deepgram Aura (true streaming WS) by default, falling back to ElevenLabs for
# target languages Aura has no voice for.
REALTIME_TTS_BACKEND = os.environ.get("REALTIME_TTS_BACKEND", "deepgram").lower()

DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")
DEEPGRAM_TTS_MODEL = os.environ.get("DEEPGRAM_TTS_MODEL", "aura-2-luna-en")
# Languages Aura ships a curated voice for. Anything else routes to ElevenLabs.
DEEPGRAM_TTS_MODELS = {
    "en": "aura-2-luna-en",
    "es": "aura-2-sirio-es",
}


def groq_api_keys() -> list[str]:
    """All configured Groq keys (GROQ_API_KEY, GROQ_API_KEY_2, ...).

    The pipeline spreads STT/translate requests round-robin across these to
    avoid per-key rate limits.
    """
    keys, idx = [], 2
    first = os.environ.get("GROQ_API_KEY")
    if first:
        keys.append(first)
    while True:
        key = os.environ.get(f"GROQ_API_KEY_{idx}")
        if not key:
            break
        keys.append(key)
        idx += 1
    seen, out = set(), []
    for k in keys:
        if k and k not in seen:
            out.append(k)
            seen.add(k)
    return out

# ---------------------------------------------------------------------------
# Translation backend selection
# ---------------------------------------------------------------------------
# "nvidia" -> NVIDIA NIM riva-translate-4b-instruct-v2
# "groq"   -> Groq llama-3.3-70b-versatile (default; far more accurate)
TRANSLATE_BACKEND = os.environ.get("TRANSLATE_BACKEND", "groq").lower()

# Groq Orpheus TTS model ids + voice persona ids.
GROQ_TTS_MODEL_EN = "canopylabs/orpheus-v1-english"
GROQ_TTS_MODEL_AR = "canopylabs/orpheus-arabic-saudi"
GROQ_TTS_VOICES_EN = ["autumn", "diana", "hannah", "austin", "daniel", "troy"]
GROQ_TTS_VOICES_AR = ["abdullah", "fahad", "sultan", "lulwa", "noura", "aisha"]
# Orpheus enforces a hard 200-character input limit per request.
GROQ_TTS_MAX_CHARS = 200

# Languages Groq Orpheus TTS can actually pronounce. Anything else must be
# routed to a multilingual backend (e.g. ElevenLabs) for intelligible speech.
GROQ_TTS_LANGS = {"en", "ar"}


def select_tts_backend(target_language: str, preferred: Optional[str] = None) -> str:
    """Pick the TTS backend for a given target language.

    ``preferred`` may be ``"groq"``, ``"elevenlabs"``, ``"auto"``/``None``.
    ``auto`` uses Groq only for en/ar, otherwise ElevenLabs (multilingual).
    """
    if preferred in ("groq", "elevenlabs"):
        return preferred
    return "groq" if target_language in GROQ_TTS_LANGS else "elevenlabs"


def select_realtime_tts_backend(
    target_language: str, preferred: Optional[str] = None
) -> str:
    """Pick the *streaming* TTS engine for the live interpreter.

    Groq TTS is not realtime-capable, so it is never selected here. The default
    is Deepgram Aura; if the target language has no Aura voice, we fall back to
    ElevenLabs (multilingual). ``preferred`` overrides the default but is
    downgraded to ElevenLabs when Deepgram cannot pronounce the language.
    """
    want = (preferred or REALTIME_TTS_BACKEND).lower()
    if want == "deepgram" and target_language not in DEEPGRAM_TTS_MODELS:
        return "elevenlabs"
    if want == "elevenlabs":
        return "elevenlabs"
    return "deepgram" if target_language in DEEPGRAM_TTS_MODELS else "elevenlabs"


def select_translate_backend(preferred: Optional[str] = None) -> str:
    """Pick the translation backend for the live interpreter.

    Accepts ``"nvidia"`` (riva-translate-4b) or ``"groq"`` (Llama-3.3-70B).
    Defaults to ``TRANSLATE_BACKEND`` (Groq). Unknown values fall back to Groq.
    """
    want = (preferred or TRANSLATE_BACKEND).lower()
    if want in ("nvidia", "groq"):
        return want
    return "groq"


# ---------------------------------------------------------------------------
# Language table (ISO-639-1 -> human readable name)
# ---------------------------------------------------------------------------
# Used for validation and to build high-quality translation prompts.
SUPPORTED_LANGUAGES: dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "ru": "Russian",
    "pl": "Polish",
    "tr": "Turkish",
    "ar": "Arabic",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "hi": "Hindi",
    "id": "Indonesian",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "no": "Norwegian",
    "cs": "Czech",
    "el": "Greek",
    "he": "Hebrew",
    "hu": "Hungarian",
    "ro": "Romanian",
    "sk": "Slovak",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
    "th": "Thai",
    "bg": "Bulgarian",
    "hr": "Croatian",
    "sr": "Serbian",
    "ta": "Tamil",
    "te": "Telugu",
    "fil": "Filipino",
}

# Whisper/Groq report the detected language as an English name ("Greek");
# map those back to ISO codes for translation prompts and captions.
_LANGUAGE_NAME_TO_CODE = {
    name.lower(): code for code, name in SUPPORTED_LANGUAGES.items()
}
# Extra aliases Whisper uses that differ from our display names.
_LANGUAGE_NAME_TO_CODE.update({
    "telugu": "te",
    "greek": "el",
    "filipino": "fil",
    "tagalog": "fil",
})


def language_name_to_code(name: Optional[str]) -> Optional[str]:
    """'Greek' -> 'el'; None/unknown -> None."""
    if not name:
        return None
    return _LANGUAGE_NAME_TO_CODE.get(str(name).strip().lower())


@dataclass
class Settings:
    """Resolved runtime configuration.

    ``from_env`` validates that the mandatory credentials are present and
    raises a clear error early rather than failing mid-pipeline.
    """

    groq_api_key: str
    elevenlabs_api_key: str
    temp_dir: Path = Path(tempfile.gettempdir())

    @classmethod
    def from_env(
        cls,
        *,
        groq_api_key: Optional[str] = None,
        elevenlabs_api_key: Optional[str] = None,
        backend: Optional[str] = None,
    ) -> "Settings":
        groq = groq_api_key or os.environ.get("GROQ_API_KEY")
        eleven = elevenlabs_api_key or os.environ.get("ELEVENLABS_API_KEY")
        backend = (backend or TTS_BACKEND).lower()

        missing: list[str] = []
        if not groq:
            missing.append("GROQ_API_KEY")
        # ElevenLabs key is only required when using the cloning backend.
        if backend in ("elevenlabs", "clone") and not eleven:
            missing.append("ELEVENLABS_API_KEY")

        if missing:
            raise RuntimeError(
                "Missing required environment variable(s): "
                + ", ".join(missing)
                + ". Set them in your shell or a .env file (see .env.example)."
            )

        return cls(groq_api_key=groq, elevenlabs_api_key=eleven)

    @staticmethod
    def validate_language(code: str) -> str:
        code = code.lower().strip()
        if code not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language code '{code}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_LANGUAGES))}"
            )
        return code
