"""Machine translation via Groq's Llama-3.3-70b.

The model is instructed to act as a strict translator so we get clean target
text without explanatory chatter. We keep the prompt tightly constrained and
return only the translated string.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator, Optional

from logger import get_logger

try:  # pragma: no cover - import guard
    from groq import Groq, GroqError
except Exception:  # noqa: BLE001
    Groq = None  # type: ignore
    GroqError = Exception  # type: ignore

from config import SUPPORTED_LANGUAGES
from logger import get_logger
from utils import KeyRotator

logger = get_logger("translator")

SYSTEM_PROMPT = (
    "You are an elite, native-level polyglot translator capable of translating "
    "between 80+ languages. Your primary goal is to deliver translations that "
    "sound indistinguishable from text written by a native speaker.\n\n"
    "Strictly enforce these translation rules:\n"
    "1. IDIOMATIC FIDELITY: Never translate literally or word-for-word if it "
    "damages natural phrasing. Adapt idioms, cultural references, and syntax to "
    "match native speech patterns of the target language.\n"
    "2. GRAMMATICAL PRECISION: Adhere flawlessly to the target language's "
    "grammatical rules, including correct case endings, verb conjugations, "
    "noun-adjective agreements, and sentence structures.\n"
    "3. CONTEXT & TONE MATCHING: Analyze the source text's tone (casual, "
    "formal, technical, literary) and mirror it perfectly in the target "
    "language. Default to standard polite speech unless casual cues are "
    "present.\n"
    "4. ZERO CONVERSATIONAL FILLER: Do not explain your choices, apologize, or "
    "provide commentary. Output ONLY the final translated text.\n"
    "5. THANKS PHRASES: When the text contains a thanks variant — 'appreciate "
    "it', 'many thanks', 'thanks a lot', 'much appreciated' — render it as the "
    "standard natural 'thank you' equivalent in the target language, never as "
    "a literal translation of the variant."
)


@dataclass
class TranslationResult:
    translated_text: str
    source_language: str
    target_language: str
    detected_source: Optional[str] = None


class Translator:
    """Wrap Groq chat-completions translation with key rotation."""

    def __init__(self, api_key: Optional[str] = None,
                 api_keys: Optional[list[str]] = None,
                 model: str = "openai/gpt-oss-120b"):
        if Groq is None:
            raise RuntimeError(
                "The 'groq' package is required. Install it with "
                "`pip install groq`."
            )
        keys = list(api_keys or [])
        if not keys and api_key:
            keys.append(api_key)
        if not keys:
            from config import groq_api_keys

            keys = groq_api_keys()
        if not keys:
            raise RuntimeError("GROQ_API_KEY is not set.")
        self._rotator = KeyRotator(keys)
        self._client = None  # created per request (key rotates)
        self.model = model

    def translate(
        self,
        text: str,
        target_language: str,
        source_language: Optional[str] = None,
        target_language_name: Optional[str] = None,
    ) -> TranslationResult:
        """Translate ``text`` into ``target_language`` (ISO-639-1 code)."""
        if not text.strip():
            raise ValueError("Cannot translate empty text.")

        tgt = target_language_name or target_language
        src_hint = (
            f" The source language is {source_language}." if source_language
            else " The source language should be auto-detected."
        )
        user_prompt = (
            f"Translate the following text into {tgt}.{src_hint}\n\n"
            f"TEXT:\n{text}"
        )

        logger.info("Translating into %s with %s ...", tgt, self.model)
        try:
            client = Groq(api_key=self._rotator.next())
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,          # deterministic, faithful output
                max_tokens=2048,
                stop=None,
            )
        except GroqError as exc:
            logger.error("Groq translation failed: %s", exc)
            raise RuntimeError(f"Translation failed: {exc}") from exc

        translated = (response.choices[0].message.content or "").strip()
        # Strip stray surrounding quotes the model sometimes adds.
        if len(translated) >= 2 and translated[0] in "\"'`" and translated[-1] == translated[0]:
            translated = translated[1:-1].strip()

        if not translated:
            raise RuntimeError("Translation returned empty text.")

        logger.info("Translated: %s", translated[:80] + ("..." if len(translated) > 80 else ""))
        return TranslationResult(
            translated_text=translated,
            source_language=source_language or "auto",
            target_language=target_language,
        )

    def translate_stream(
        self,
        text: str,
        target_language: str,
        source_language: Optional[str] = None,
        target_language_name: Optional[str] = None,
    ) -> Iterator[str]:
        """Yield translated tokens as they stream from Groq (Llama-3.3-70B)."""
        if not text.strip():
            return

        # Use full language names for reliability (Llama reads "Greek" better
        # than the ISO code "el").
        tgt = target_language_name or SUPPORTED_LANGUAGES.get(target_language, target_language)
        src = (
            SUPPORTED_LANGUAGES.get(source_language, source_language)
            if source_language
            else None
        )
        src_hint = (
            f" The source language is {src}." if src
            else " The source language should be auto-detected."
        )
        user_prompt = (
            f"Translate the following text into {tgt}.{src_hint}\n\n"
            f"TEXT:\n{text}"
        )

        logger.info("Streaming translation into %s with %s ...", tgt, self.model)
        try:
            client = Groq(api_key=self._rotator.next())
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=2048,
                stream=True,
            )
        except GroqError as exc:  # type: ignore[name-defined]
            logger.error("Groq streaming translation failed: %s", exc)
            raise RuntimeError(f"Translation failed: {exc}") from exc

        for chunk in resp:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
