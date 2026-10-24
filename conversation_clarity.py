"""Conversation Clarity (TaskForge): bounded ambiguity detection.

A finalized utterance is submitted to the configured LLM provider, which is
asked to identify ONE materially plausible ambiguity. Output is validated
deterministically before it ever reaches the UI:

    * the model must quote text that actually appears in the utterance
      (case/punctuation-insensitive) -- a fabricated quote is rejected;
    * confidence is clamped to [0, 1];
    * strings are bounded; interpretations are capped;
    * "no meaningful ambiguity" produces no alert at all.

The finding is a SUGGESTION only. The user can edit the proposed
clarification question and approve it. Nothing is ever auto-sent,
auto-spoken, or treated as a fact about a speaker's intentions.
Provider calls run through the proven bounded AIService (queue, timeout,
circuit breaker) -- never inline on the audio pipeline thread.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from threading import RLock
from typing import Optional, Protocol
import json
import logging
import re
import time
import uuid

logger = logging.getLogger(__name__)

_MAX_INTERPRETATIONS = 4
_MAX_TEXT_CHARS = 300
_MIN_UTTERANCE_WORDS = 4
_MAX_FINDINGS = 20
_DEDUP_MAX = 60
_MIN_WORDS_FOR_CLARIFICATION = 4

_SYSTEM_PROMPT = (
    "You are a careful communication reviewer. You receive one finalized "
    "utterance from a live interpreted conversation (the original text and, "
    "when available, its translation). Identify at most ONE materially "
    "plausible ambiguity: wording a listener could reasonably misread, or "
    "whose translation may not carry the speaker's intent. Ignore trivial "
    "vagueness and small talk. If the utterance is clear, say so.\n"
    "Respond ONLY with valid JSON, no other text:\n"
    "{\n"
    '  "ambiguous": true|false,\n'
    '  "quote": "exact phrase copied from the original utterance",\n'
    '  "why": "one or two sentences on why it may be misread",\n'
    '  "interpretations": ["reading A", "reading B"],\n'
    '  "confidence": 0.0,\n'
    '  "clarification": "one concise question the speaker could ask to '
    'remove the ambiguity"\n'
    "}\n"
    "The quote MUST be copied verbatim from the original utterance. If "
    "ambiguous is false, omit the other fields."
)


@dataclass(frozen=True)
class ClaritySource:
    """Trusted request data. Provenance is bound here, never taken from
    model output."""

    uid: int
    speaker: str  # "p1" | "you"
    original: str
    translation: str = ""


@dataclass
class ClarityFinding:
    finding_id: str
    source_uid: int
    speaker: str
    quote: str
    why_ambiguous: str
    interpretations: tuple
    confidence: float  # model-estimated, clamped to [0, 1]
    suggested_clarification: str
    status: str = "pending"  # pending | approved | dismissed
    approved_text: str = ""
    created_ms: float = 0.0

    def to_dict(self):
        return {
            "finding_id": self.finding_id,
            "source_uid": self.source_uid,
            "speaker": self.speaker,
            "quote": self.quote,
            "why": self.why_ambiguous,
            "interpretations": list(self.interpretations),
            "confidence": round(self.confidence, 2),
            "suggested_clarification": self.suggested_clarification,
            "status": self.status,
            "approved_text": self.approved_text,
        }


class ClarityProvider(Protocol):
    """Same shape the AIService expects: analyze(request) -> text."""

    def analyze(self, request) -> Optional[str]:  # pragma: no cover - protocol
        ...


def _norm(text: str) -> str:
    """Lowercase, strip punctuation/whitespace for quote containment checks."""
    return re.sub(r"[\W_]+", "", (text or "").lower())


def _extract_json(raw: str):
    """Parse the first JSON object found in the model output, or None."""
    if not raw:
        return None
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except (ValueError, TypeError):
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else None
    except (ValueError, TypeError):
        return None


def validate_output(raw: str, source: ClaritySource):
    """Deterministically validate model output against the trusted source.

    Returns a ClarityFinding, or None when there is no meaningful
    ambiguity or the output fails validation.
    """
    data = _extract_json(raw)
    if data is None:
        logger.warning("Clarity: unparseable model output discarded")
        return None
    if data.get("ambiguous") is not True:
        return None  # explicit "no ambiguity" -> no alert, by design

    original = source.original or ""
    quote = str(data.get("quote", "") or "").strip()
    if not quote or _norm(quote) not in _norm(original):
        logger.warning("Clarity: fabricated or missing quote rejected (uid=%s)",
                       source.uid)
        return None

    why = str(data.get("why", "") or "").strip()
    if len(why) < 10:
        return None
    why = why[:_MAX_TEXT_CHARS]

    raw_interp = data.get("interpretations", [])
    interpretations = []
    if isinstance(raw_interp, list):
        for item in raw_interp:
            text = str(item or "").strip()
            if text:
                interpretations.append(text[:_MAX_TEXT_CHARS])
            if len(interpretations) >= _MAX_INTERPRETATIONS:
                break
    if not interpretations:
        return None

    try:
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0

    clarification = str(data.get("clarification", "") or "").strip()
    if len(clarification.split()) < _MIN_WORDS_FOR_CLARIFICATION:
        return None
    clarification = clarification[:_MAX_TEXT_CHARS]
    if not clarification.endswith("?"):
        clarification += "?"

    return ClarityFinding(
        finding_id=uuid.uuid4().hex[:12],
        source_uid=source.uid,
        speaker=source.speaker,
        quote=quote[:_MAX_TEXT_CHARS],
        why_ambiguous=why,
        interpretations=tuple(interpretations),
        confidence=confidence,
        suggested_clarification=clarification,
        created_ms=time.time() * 1000.0,
    )


class ClarityEngine:
    """Bounded ambiguity analysis over a live transcript.

    Submissions are pre-filtered (short/small-talk utterances and duplicates
    are skipped), deduplicated by uid and normalized quote, and executed on
    the bounded AIService. ``poll()`` drains finished requests, validates
    output, and delivers findings to the callback.
    """

    def __init__(self, ai_service, *, on_finding=None,
                 min_confidence: float = 0.0):
        self._service = ai_service
        self._on_finding = on_finding
        self._min_confidence = float(min_confidence)
        self._lock = RLock()
        self._seen_uids = set()
        self._seen_quotes = deque(maxlen=_DEDUP_MAX)
        self._inflight = {}  # request_id -> ClaritySource
        self._findings = {}  # finding_id -> ClarityFinding
        self._order = deque(maxlen=_MAX_FINDINGS)
        self._closed = False

    # -- submission ----------------------------------------------------- #
    def submit(self, source: ClaritySource) -> Optional[str]:
        """Submit one finalized utterance. Non-blocking; returns the request
        id or None when pre-filtered/deduplicated/closed."""
        with self._lock:
            if self._closed or source.uid in self._seen_uids:
                return None
            norm_quote = _norm(source.original)
            if not norm_quote or norm_quote in self._seen_quotes:
                return None
            if len((source.original or "").split()) < _MIN_UTTERANCE_WORDS:
                return None
            self._seen_uids.add(source.uid)
            self._seen_quotes.append(norm_quote)
        request = self._build_request(source)
        try:
            result = self._service.submit(request, timeout_s=10.0)
        except Exception as exc:  # noqa: BLE001 - clarity must never break audio
            logger.warning("Clarity submit failed: %s", exc)
            return None
        if result is None:
            return None
        with self._lock:
            self._inflight[result.request_id] = source
        return result.request_id

    def _build_request(self, source: ClaritySource):
        from ai_service import AIRequest  # engine module; imported lazily

        parts = [f'Original: "{source.original}"']
        if source.translation:
            parts.append(f'Translation: "{source.translation}"')
        return AIRequest(
            id=uuid.uuid4().hex[:16],
            text="\n".join(parts),
            side=source.speaker,
            uid=source.uid,
            mode="clarity",
            timestamp=time.time(),
        )

    # -- polling -------------------------------------------------------- #
    def poll(self) -> Optional[ClarityFinding]:
        """Drain at most one finished request. Called from the UI poller."""
        with self._lock:
            if not self._inflight:
                return None
            request_id, source = next(iter(self._inflight.items()))
        result = self._service.get_result(request_id)
        if result is None or result.status.value in ("pending", "running"):
            return None
        with self._lock:
            self._inflight.pop(request_id, None)
        if result.status.value != "completed" or not result.output:
            logger.info("Clarity: request %s ended as %s", request_id,
                        result.status.value)
            return None
        finding = validate_output(result.output, source)
        if finding is None:
            return None
        if finding.confidence < self._min_confidence:
            return None
        self._store(finding)
        if self._on_finding is not None:
            try:
                self._on_finding(finding)
            except Exception:  # noqa: BLE001 - UI must not break analysis
                logger.exception("Clarity finding callback failed")
        return finding

    def _store(self, finding: ClarityFinding) -> None:
        with self._lock:
            self._findings[finding.finding_id] = finding
            self._order.append(finding.finding_id)
            while len(self._order) > _MAX_FINDINGS:
                dropped = self._order.popleft()
                self._findings.pop(dropped, None)

    # -- user actions ---------------------------------------------------- #
    def approve(self, finding_id: str, edited_text: str) -> Optional[ClarityFinding]:
        """Record the user's (possibly edited) clarification. Never auto-sends."""
        with self._lock:
            finding = self._findings.get(finding_id)
            if finding is None or finding.status != "pending":
                return None
            text = (edited_text or "").strip()[:_MAX_TEXT_CHARS]
            if not text:
                return None
            finding.approved_text = text
            finding.status = "approved"
            return finding

    def dismiss(self, finding_id: str) -> bool:
        with self._lock:
            finding = self._findings.get(finding_id)
            if finding is None or finding.status != "pending":
                return False
            finding.status = "dismissed"
            return True

    def get(self, finding_id: str) -> Optional[ClarityFinding]:
        with self._lock:
            return self._findings.get(finding_id)

    def pending_count(self) -> int:
        with self._lock:
            return sum(1 for f in self._findings.values()
                       if f.status == "pending")

    def close(self) -> None:
        with self._lock:
            self._closed = True
        try:
            self._service.close()
        except Exception:  # noqa: BLE001
            pass
