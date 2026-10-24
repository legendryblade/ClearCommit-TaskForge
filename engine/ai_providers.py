"""Real AI provider backends for ClearCommit.

Implements the existing protocols (CoachingProvider, AssistantProvider,
MeetingIntelligenceProvider) against any OpenAI-compatible chat-completions
endpoint. Provider presets (OpenRouter, Groq, custom) reduce configuration
to: pick a preset, paste an API key, optionally override the model.

Safety:
    * API keys are read from environment variables (the existing .env
      mechanism) - never hard-coded, never logged.
    * Synchronous HTTP with a bounded timeout; the calling service (AIService)
      already enforces per-request timeouts and circuit breaking.
    * A failed call raises - the calling service records it as FAILED and
      continues.
    * No raw audio is ever transmitted - text only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import os
import time

import requests

#: Transient HTTP statuses worth one bounded retry (free-tier rate limits and
#: gateway hiccups fail fast, so the retry adds ~2s, well inside AIService
#: request timeouts).
_RETRY_STATUS = {429, 502, 503, 504}
_RETRY_BACKOFF_S = 1.5


@dataclass(frozen=True)
class ProviderPreset:
    key: str
    label: str
    base_url: str
    key_env_var: str
    default_model: str
    models: tuple[str, ...] = ()


PROVIDER_PRESETS: dict = {
    "openrouter": ProviderPreset(
        key="openrouter", label="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
        key_env_var="OPENROUTER_API_KEY",
        default_model="openai/gpt-oss-120b",
        models=(
            "openai/gpt-oss-120b", "openai/gpt-oss-20b",
            "anthropic/claude-sonnet-4", "google/gemini-2.0-flash-001",
            "meta-llama/llama-4-maverick", "qwen/qwen3-235b-a22b",
        ),
    ),
    "groq": ProviderPreset(
        key="groq", label="Groq",
        base_url="https://api.groq.com/openai/v1",
        key_env_var="GROQ_API_KEY",
        default_model="openai/gpt-oss-120b",
        models=(
            "openai/gpt-oss-120b", "openai/gpt-oss-20b",
            "llama-3.3-70b-versatile", "llama-3.1-8b-instant",
        ),
    ),
    "custom": ProviderPreset(
        key="custom", label="Custom (OpenAI-compatible)",
        base_url="", key_env_var="CUSTOM_AI_API_KEY",
        default_model="", models=(),
    ),
}


@dataclass
class ProviderConfig:
    preset: str = "groq"
    base_url: str = ""
    model: str = ""
    temperature: float = 0.3
    max_tokens: int = 1024
    timeout_s: float = 15.0
    #: Per-preset keys entered in the Models tab. Stored in the app settings
    #: file (plaintext, same trust level as .env). Empty/missing falls back
    #: to the preset's env var. UI key takes precedence over the env var.
    api_keys: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            return cls()
        preset = data.get("preset", "groq")
        if preset not in PROVIDER_PRESETS:
            preset = "groq"
        try:
            temperature = max(0.0, min(1.5, float(data.get("temperature", 0.3))))
        except (TypeError, ValueError):
            temperature = 0.3
        try:
            max_tokens = max(64, min(4096, int(data.get("max_tokens", 1024))))
        except (TypeError, ValueError):
            max_tokens = 1024
        try:
            timeout_s = max(3.0, min(60.0, float(data.get("timeout_s", 15.0))))
        except (TypeError, ValueError):
            timeout_s = 15.0
        base_url = data.get("base_url", "")
        model = data.get("model", "")
        raw_keys = data.get("api_keys", {})
        api_keys = {}
        if isinstance(raw_keys, dict):
            for k, v in raw_keys.items():
                if k in PROVIDER_PRESETS and isinstance(v, str) and v.strip():
                    api_keys[k] = v.strip()
        return cls(
            preset=preset,
            base_url=str(base_url).strip() if isinstance(base_url, str) else "",
            model=str(model).strip() if isinstance(model, str) else "",
            temperature=temperature, max_tokens=max_tokens, timeout_s=timeout_s,
            api_keys=api_keys,
        )

    def to_dict(self):
        return {"preset": self.preset, "base_url": self.base_url,
                "model": self.model, "temperature": self.temperature,
                "max_tokens": self.max_tokens, "timeout_s": self.timeout_s,
                "api_keys": dict(self.api_keys)}

    def resolve_base_url(self):
        if self.base_url:
            return self.base_url.rstrip("/")
        p = PROVIDER_PRESETS.get(self.preset)
        return p.base_url if p else ""

    def resolve_model(self):
        if self.model:
            return self.model
        p = PROVIDER_PRESETS.get(self.preset)
        return p.default_model if p else ""

    def resolve_api_key(self):
        typed = ""
        if isinstance(self.api_keys, dict):
            typed = str(self.api_keys.get(self.preset, "") or "").strip()
        if typed:
            return typed
        p = PROVIDER_PRESETS.get(self.preset)
        return os.environ.get(p.key_env_var if p else "GROQ_API_KEY", "")

    def api_key_source(self):
        """'custom' when a UI-entered key is active, 'env' when the env var
        is used, '' when no key is available for the current preset."""
        if isinstance(self.api_keys, dict) and \
                str(self.api_keys.get(self.preset, "") or "").strip():
            return "custom"
        p = PROVIDER_PRESETS.get(self.preset)
        if p and os.environ.get(p.key_env_var, ""):
            return "env"
        return ""

    def is_configured(self):
        return bool(self.resolve_base_url() and self.resolve_api_key()
                    and self.resolve_model())


class OpenAICompatibleProvider:
    """Universal provider for any OpenAI-compatible chat-completions endpoint."""

    def __init__(self, config: ProviderConfig):
        self._config = config
        self._session = requests.Session()

    def _chat(self, system: str, user_text: str,
              history: Optional[list] = None) -> str:
        cfg = self._config
        url = cfg.resolve_base_url() + "/chat/completions"
        api_key = cfg.resolve_api_key()
        if not api_key:
            env_var = PROVIDER_PRESETS[cfg.preset].key_env_var
            raise RuntimeError("api_key_not_set:" + env_var)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_text})
        headers = {"Authorization": "Bearer " + api_key,
                   "Content-Type": "application/json"}
        payload = {"model": cfg.resolve_model(), "messages": messages,
                   "temperature": cfg.temperature, "max_tokens": cfg.max_tokens}
        resp = None
        for attempt in range(2):
            resp = self._session.post(url, json=payload, headers=headers,
                                      timeout=cfg.timeout_s)
            if resp.status_code in _RETRY_STATUS and attempt == 0:
                time.sleep(_RETRY_BACKOFF_S)
                continue
            break
        if resp.status_code == 429:
            raise RuntimeError("rate_limited")
        if resp.status_code == 401:
            raise RuntimeError("invalid_api_key")
        if resp.status_code != 200:
            raise RuntimeError("provider_error_" + str(resp.status_code))
        data = resp.json()
        content = (data.get("choices", [{}])[0].get("message", {})
                   .get("content", ""))
        return (content or "").strip()

    def analyze(self, text: str, context=None):
        """CoachingProvider: text is the full prompt; returns raw response.

        Also handles MeetingIntelligenceProvider when the context includes
        the meeting schema (set by _MeetingAnalysisProvider).
        """
        system = ""
        if context and isinstance(context, dict):
            system = context.get("system_prompt_override", "")
            if context.get("meeting_schema"):
                # MeetingIntelligenceProvider: use the schema as part of
                # the user text and the meeting system prompt.
                system = context.get("system_prompt",
                                     "You are a precise meeting-intelligence extractor.")
                return self._chat(system, text + "\n\n" + context["meeting_schema"]) or None
        return self._chat(system, text) or None

    def answer(self, question: str, context: str, history=None):
        """AssistantProvider: context is the grounded brief (which already
        contains the grounding contract and any custom instructions).

        The system message is kept minimal; the grounding contract and
        custom instructions are in the context (user message) where they
        have the most influence on the model's output.
        """
        system = ("You are ClearCommit's meeting assistant.")
        return self._chat(system, context + "\n\nQuestion: " + question,
                          history=history) or ""

    def analyze_transcript(self, transcript_text: str, context=None):
        """MeetingIntelligenceProvider: extract structured intelligence."""
        system = ("You are a precise meeting-intelligence extractor. From "
                  "the transcript, output only the requested JSON.")
        if context and isinstance(context, dict):
            system = context.get("system_prompt", system)
        schema = (
            'Respond ONLY with valid JSON, no other text:\n'
            '{\n'
            '  "topics": ["..."],\n'
            '  "decisions": [{"topic": "...", "proposed_value": "...", '
            '"decided_value": null, "status": "proposed|decided|rejected", '
            '"evidence_ids": [int], "speaker": "p1|you"}],\n'
            '  "action_items": [{"description": "...", "owner": "p1|you|null", '
            '"deadline": "YYYY-MM-DD|null", "status": "assigned|unassigned", '
            '"evidence_ids": [int]}],\n'
            '  "open_questions": [{"question": "...", "answered": bool, '
            '"answer": "...|null", "evidence_ids": [int]}],\n'
            '  "candidate_commitments": [{"topic": "...", "proposed_value": '
            '"...", "status": "inferred|proposed", "evidence_ids": [int], '
            '"speaker": "p1|you"}],\n'
            '  "events": [{"event_type": "meeting|appointment|deadline|'
            'milestone|follow_up|risk|blocker", "title": "...", '
            '"excerpt": "...", "speaker": "p1|you", "spoken_date": "...|null", '
            '"normalized_date": "YYYY-MM-DD|null", "review_status": '
            '"explicit|inferred|needs_review", "missing_details": "...", '
            '"evidence_ids": [int]}]\n'
            '}\n'
        )
        return self._chat(system, transcript_text + "\n\n" + schema) or None


def get_provider_for_config(config: ProviderConfig):
    if not config.is_configured():
        p = PROVIDER_PRESETS.get(config.preset)
        env_var = p.key_env_var if p else "API_KEY"
        raise RuntimeError(
            "Provider '" + config.preset + "' not configured. Set " + env_var + ".")
    return OpenAICompatibleProvider(config)
