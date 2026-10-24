"""Tests for Conversation Clarity (TaskForge)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from conversation_clarity import (
    ClarityEngine,
    ClaritySource,
    validate_output,
)


SRC = ClaritySource(
    uid=7, speaker="p1",
    original="We should touch base about the deliverable sometime next week.",
    translation="Deberíamos ponernos en contacto sobre la entrega la próxima semana.",
)

GOOD = (
    '{"ambiguous": true, "quote": "touch base", '
    '"why": "The idiom may translate poorly and sound like vague small talk.", '
    '"interpretations": ["schedule a short meeting", "just stay in contact"], '
    '"confidence": 0.7, '
    '"clarification": "Would you like to schedule a specific time to review it?"}'
)


# ---- validate_output ------------------------------------------------------ #
def test_valid_output_produces_finding():
    f = validate_output(GOOD, SRC)
    assert f is not None
    assert f.quote == "touch base"
    assert f.source_uid == 7
    assert len(f.interpretations) == 2
    assert 0.0 <= f.confidence <= 1.0
    assert f.suggested_clarification.endswith("?")
    assert f.status == "pending"


def test_no_ambiguity_means_no_alert():
    out = '{"ambiguous": false}'
    assert validate_output(out, SRC) is None


def test_unparseable_output_rejected():
    assert validate_output("not json at all", SRC) is None
    assert validate_output("", SRC) is None


def test_fabricated_quote_rejected():
    out = GOOD.replace('"touch base"', '"ship the budget by Friday"')
    assert validate_output(out, SRC) is None


def test_quote_with_different_punctuation_still_matches():
    out = GOOD.replace('"touch base"', '"Touch, base!"')
    assert validate_output(out, SRC) is not None


def test_confidence_clamped():
    out = GOOD.replace('"confidence": 0.7', '"confidence": 9.5')
    f = validate_output(out, SRC)
    assert f is not None and f.confidence == 1.0
    out = GOOD.replace('"confidence": 0.7', '"confidence": "high"')
    f = validate_output(out, SRC)
    assert f is not None and f.confidence == 0.0


def test_interpretations_bounded_and_cleaned():
    many = ", ".join(f'"read {i}"' for i in range(10))
    out = GOOD.replace(
        '["reading a short meeting", "just stay in contact"]', "[" + many + "]")
    f = validate_output(out, SRC)
    assert f is not None and len(f.interpretations) <= 4


def test_question_mark_appended():
    out = GOOD.replace(
        "Would you like to schedule a specific time to review it?",
        "Could you clarify what review you mean")
    f = validate_output(out, SRC)
    assert f.suggested_clarification.endswith("?")


def test_useless_fields_rejected():
    no_why = GOOD.replace(
        '"why": "The idiom may translate poorly and sound like vague small talk.", ',
        '"why": "short", ')
    assert validate_output(no_why, SRC) is None
    no_clar = GOOD.replace(
        '"clarification": "Would you like to schedule a specific time to review it?"',
        '"clarification": "hmm?"')
    assert validate_output(no_clar, SRC) is None


# ---- ClarityEngine -------------------------------------------------------- #
class FakeService:
    """Minimal AIService stand-in: submit -> handle, get_result by uid."""

    def __init__(self):
        self.next_status = "completed"
        self.output = GOOD
        self.requests = []

    def submit(self, request, timeout_s=None):
        class H:
            request_id = f"r{len(self.requests)}"
        self.requests.append(request)
        return H()

    def get_result(self, request_id):
        class R:
            status = type("S", (), {"value": self.next_status})()
            output = self.output
        return R()

    def close(self):
        pass


def _engine(**kw):
    return ClarityEngine(FakeService(), **kw)


def test_submit_prefilters_short_utterances():
    e = _engine()
    assert e.submit(ClaritySource(1, "you", "hi there ok")) is None
    assert e.submit(ClaritySource(2, "you", "x" * 200)) is None  # one word


def test_submit_dedups_uid_and_quote():
    e = _engine()
    rid = e.submit(SRC)
    assert rid is not None
    assert e.submit(SRC) is None            # same uid
    assert e.submit(ClaritySource(8, "p1", SRC.original)) is None  # same text


def test_poll_delivers_validated_finding():
    seen = []
    e = _engine(on_finding=seen.append)
    assert e.submit(SRC) is not None
    finding = e.poll()               # fake service completes immediately
    assert finding is not None
    assert len(seen) == 1
    assert e.pending_count() == 1
    # duplicate poll must not re-deliver
    assert e.poll() is None
    assert len(seen) == 1


def test_failed_request_delivers_nothing():
    svc = FakeService()
    svc.next_status = "timed_out"
    e = ClarityEngine(svc)
    e.submit(SRC)
    assert e.poll() is None
    assert e.pending_count() == 0


def test_min_confidence_gate():
    out = GOOD.replace('"confidence": 0.7', '"confidence": 0.2')
    svc = FakeService()
    svc.output = out
    e = ClarityEngine(svc, min_confidence=0.5)
    e.submit(SRC)
    assert e.poll() is None


def test_approve_records_edited_text_and_never_sends():
    seen = []
    e = _engine(on_finding=seen.append)
    e.submit(SRC)
    e.poll()
    f = seen[0]
    assert e.approve(f.finding_id, "  Did you mean a concrete meeting?  ")
    stored = e.get(f.finding_id)
    assert stored.status == "approved"
    assert stored.approved_text == "Did you mean a concrete meeting?"
    assert e.approve(f.finding_id, "again") is None  # terminal
    assert e.approve("missing", "x") is None


def test_approve_rejects_empty_text():
    e = _engine()
    e.submit(SRC)
    e.poll()
    f = e.get(list(e._findings)[0])
    assert e.approve(f.finding_id, "   ") is None
    assert e.get(f.finding_id).status == "pending"


def test_dismiss():
    e = _engine()
    e.submit(SRC)
    e.poll()
    fid = list(e._findings)[0]
    assert e.dismiss(fid) is True
    assert e.dismiss(fid) is False


def test_findings_store_bounded():
    e = _engine()
    for i in range(30):
        e.submit(ClaritySource(100 + i, "p1", f"Ambiguity number {i} about deadlines and owners"))
        e.poll()
    assert len(e._findings) <= 20


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
