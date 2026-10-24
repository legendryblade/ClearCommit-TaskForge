"""Pure bridge payload builders (TaskForge).

Every message that crosses the Python -> JavaScript boundary is produced
here as a plain dict with a fixed schema. Keeping them in a webview-free
module makes the contracts unit-testable.
"""

from __future__ import annotations


def caption_event(kind: str, side: str, meta: str, text: str, uid: int):
    """One caption line from the pipeline queue.

    kind: "orig" (recognized speech) or "trans" (translation).
    side: "p1" (other participant) or "you" (the user).
    meta: source label for orig ("system / Discord"), language for trans.
    """
    if kind not in ("orig", "trans"):
        raise ValueError("kind")
    if side not in ("p1", "you"):
        raise ValueError("side")
    return {
        "type": "caption",
        "kind": kind,
        "side": side,
        "meta": str(meta or ""),
        "text": str(text or ""),
        "uid": int(uid),
        "ts": None,  # wall-clock time attached by the JS layer on arrival
    }


def status_event(state: str, message: str = ""):
    if state not in ("idle", "starting", "running", "paused", "stopped", "error"):
        raise ValueError("state")
    return {"type": "status", "state": state, "message": str(message or "")}


def clarity_event(finding):
    return {"type": "clarity", "finding": finding.to_dict()}


def clarity_resolved_event(finding_id: str, status: str,
                           approved_text: str = ""):
    return {
        "type": "clarity_resolved",
        "finding_id": str(finding_id),
        "status": status,
        "approved_text": str(approved_text or ""),
    }


def error_event(message: str):
    return {"type": "error", "message": str(message or "")}


def hello_payload(languages, credentials: dict, has_clarity_provider: bool):
    return {
        "type": "hello",
        "app": "ClearCommit TaskForge",
        "languages": list(languages),
        "credentials": dict(credentials or {}),
        "clarity_provider": bool(has_clarity_provider),
    }
