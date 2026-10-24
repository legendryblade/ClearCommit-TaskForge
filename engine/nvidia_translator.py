"""Translation via NVIDIA NIM: ``nvidia/riva-translate-4b-instruct-v2``.

NVIDIA hosts this as an OpenAI-compatible chat-completions endpoint, so we call
it directly over ``requests`` (no extra SDK). The key is read from the
``NVIDIA_KEY_1`` environment variable. The model is an instruction-tuned
translator, so we steer it with a strict "translate only" prompt.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import requests

from config import SUPPORTED_LANGUAGES
from logger import get_logger
from translator import TranslationResult
from utils import with_retry

logger = get_logger("nvidia_translator")

NIM_BASE_URL = os.environ.get(
    "NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"
)
DEFAULT_MODEL = "nvidia/riva-translate-4b-instruct-v2"

SYSTEM_PROMPT = (
    "You are a precise translation engine. Translate the user's text from the "
    "source language into the target language. Output ONLY the translated text "
    "with no quotation marks, explanations, notes, or extra commentary. Preserve "
    "names, tone, and formatting."
)


def _lang_name(code: Optional[str]) -> str:
    """Map an ISO-639-1 code to a full name; the model follows names far better
    than bare codes (e.g. 'el' was being read as Hungarian)."""
    if not code or code == "auto":
        return "the detected source language"
    return SUPPORTED_LANGUAGES.get(code, code)


@dataclass
class NvidiaTranslator:
    """NVIDIA NIM-backed translator."""

    api_key: str
    model: str = DEFAULT_MODEL
    base_url: str = NIM_BASE_URL

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL,
                 base_url: str = NIM_BASE_URL):
        key = api_key or os.environ.get("NVIDIA_KEY_1")
        if not key:
            raise RuntimeError("NVIDIA_KEY_1 is not set.")
        self.api_key = key
        self.model = model
        self.base_url = base_url

    def translate(
        self,
        text: str,
        target_language: str,
        source_language: Optional[str] = None,
        target_language_name: Optional[str] = None,
    ) -> TranslationResult:
        if not text.strip():
            raise ValueError("Cannot translate empty text.")

        src = _lang_name(source_language)
        tgt = target_language_name or _lang_name(target_language)
        user_prompt = (
            f"Translate the following text from {src} into {tgt}.\n\n"
            f"TEXT:\n{text}"
        )

        logger.info("Translating into %s via NVIDIA %s ...", tgt, self.model)

        def _call() -> requests.Response:
            return requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.2,
                    "max_tokens": 2048,
                    "top_p": 1.0,
                    "stream": False,
                },
                timeout=60,
            )

        try:
            resp = with_retry(_call, label="nvidia.translate")
        except requests.exceptions.RequestException as exc:
            logger.error("NVIDIA translation request failed: %s", exc)
            raise RuntimeError(f"NVIDIA translation failed: {exc}") from exc

        if resp.status_code != 200:
            raise RuntimeError(
                f"NVIDIA translation failed ({resp.status_code}): {resp.text}"
            )

        data = resp.json()
        translated = data["choices"][0]["message"]["content"].strip()
        if len(translated) >= 2 and translated[0] in "\"'`" and translated[-1] == translated[0]:
            translated = translated[1:-1].strip()

        if not translated:
            raise RuntimeError("NVIDIA translation returned empty text.")

        logger.info("Translated: %s", translated[:80] + ("..." if len(translated) > 80 else ""))
        return TranslationResult(
            translated_text=translated,
            source_language=source_language or "auto",
            target_language=target_language,
        )
