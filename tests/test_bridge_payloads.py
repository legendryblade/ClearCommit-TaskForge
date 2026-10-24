"""Bridge payload schema tests (TaskForge)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import bridge_payloads as bp


class FakeFinding:
    def __init__(self):
        self.finding_id = "abc123"
        self.source_uid = 4
        self.speaker = "p1"
        self.quote = "touch base"
        self.why = "idiom"
        self.interpretations = ("a", "b")
        self.confidence = 0.7
        self.suggested_clarification = "Schedule a time?"
        self.status = "pending"
        self.approved_text = ""

    def to_dict(self):
        return {"finding_id": self.finding_id, "source_uid": self.source_uid,
                "speaker": self.speaker, "quote": self.quote, "why": self.why,
                "interpretations": list(self.interpretations),
                "confidence": self.confidence,
                "suggested_clarification": self.suggested_clarification,
                "status": self.status, "approved_text": self.approved_text}


def test_caption_event_schema():
    ev = bp.caption_event("orig", "p1", "system / Discord", "hello there", 12)
    assert ev["type"] == "caption" and ev["kind"] == "orig"
    assert ev["side"] == "p1" and ev["uid"] == 12 and ev["text"] == "hello there"
    ev = bp.caption_event("trans", "you", "en", "hola", 13)
    assert ev["kind"] == "trans" and ev["side"] == "you"


def test_caption_event_rejects_bad_values():
    with pytest.raises(ValueError):
        bp.caption_event("bogus", "p1", "m", "t", 1)
    with pytest.raises(ValueError):
        bp.caption_event("orig", "p3", "m", "t", 1)


def test_status_event_schema():
    for state in ("idle", "starting", "running", "paused", "stopped", "error"):
        ev = bp.status_event(state, "msg")
        assert ev == {"type": "status", "state": state, "message": "msg"}
    with pytest.raises(ValueError):
        bp.status_event("warp")


def test_clarity_and_error_events():
    ev = bp.clarity_event(FakeFinding())
    assert ev["type"] == "clarity" and ev["finding"]["quote"] == "touch base"
    ev = bp.clarity_resolved_event("abc123", "approved", "final question?")
    assert ev["approved_text"] == "final question?"
    ev = bp.error_event("boom")
    assert ev == {"type": "error", "message": "boom"}


def test_hello_payload():
    ev = bp.hello_payload(["en", "es"], {"groq": True}, False)
    assert ev["type"] == "hello"
    assert ev["languages"] == ["en", "es"]
    assert ev["credentials"] == {"groq": True}
    assert ev["clarity_provider"] is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
