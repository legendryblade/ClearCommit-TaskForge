"""Streaming translation via NVIDIA NIM (Server-Sent Events).

Uses the same ``riva-translate-4b-instruct-v2`` model as the batch translator
but with ``stream=True``, so translated tokens are yielded as they are
generated — the core of the low-latency pipe.
"""

from __future__ import annotations

import json
import os
from typing import Iterator, Optional

import requests

from config import SUPPORTED_LANGUAGES
from logger import get_logger
from nvidia_translator import _lang_name

logger = get_logger("nvidia_stream")

NIM_BASE_URL = os.environ.get(
    "NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"
)
DEFAULT_MODEL = "nvidia/riva-translate-4b-instruct-v2"

SYSTEM_PROMPT = (
    "You are a precise translation engine. Translate the user's text from the "
    "source language into the target language. Output ONLY the translated text "
    "with no quotation marks, explanations, notes, or extra commentary."
)


class NvidiaStreamTranslator:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        base_url: str = NIM_BASE_URL,
    ):
        key = api_key or os.environ.get("NVIDIA_KEY_1")
        if not key:
            raise RuntimeError("NVIDIA_KEY_1 is not set.")
        self.api_key = key
        self.model = model
        self.base_url = base_url

    def translate_stream(
        self,
        text: str,
        target_language: str,
        source_language: Optional[str] = None,
        target_language_name: Optional[str] = None,
    ) -> Iterator[str]:
        """Yield translated tokens as they stream from NVIDIA NIM."""
        if not text.strip():
            return

        src = _lang_name(source_language)
        tgt = target_language_name or _lang_name(target_language)
        user_prompt = (
            f"Translate the following text from {src} into {tgt}.\n\n"
            f"TEXT:\n{text}"
        )

        logger.info("Streaming translation into %s via NVIDIA %s ...", tgt, self.model)
        resp = requests.post(
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
                "stream": True,
            },
            timeout=60,
            stream=True,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"NVIDIA streaming translation failed ({resp.status_code}): {resp.text}"
            )

        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
            if delta:
                yield delta
