"""ClearCommit TaskForge — native desktop window over the proven pipeline.

Python owns everything real: audio capture, device routing, STT
(Groq Whisper / Deepgram), LLM translation, TTS (Deepgram Aura /
ElevenLabs), and the Conversation Clarity analyzer. The HTML/CSS/JS UI
renders inside a pywebview native window and talks to Python through a
narrow whitelisted bridge (window.pywebview.api.<method>). Python pushes
schema-validated events to the frontend via window.__push(...).

No API key, transcript dump, or executable payload ever crosses into the
frontend; no arbitrary JS from the frontend is ever executed by Python.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "engine"))

import webview  # noqa: E402

import bridge_payloads as bp  # noqa: E402
from conversation_clarity import ClarityEngine, ClaritySource  # noqa: E402

logger = logging.getLogger("taskforge")

UI_DIR = ROOT / "ui"


def _list_languages():
    try:
        from config import SUPPORTED_LANGUAGES
        if isinstance(SUPPORTED_LANGUAGES, dict):
            return sorted(SUPPORTED_LANGUAGES.keys())
        return sorted(SUPPORTED_LANGUAGES)
    except Exception:  # noqa: BLE001 - config should never break the window
        return ["en", "es"]


KEY_ENV = {
    "groq": "GROQ_API_KEY",
    "deepgram": "DEEPGRAM_API_KEY",
    "elevenlabs": "ELEVENLABS_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}
KEYS_FILE = (Path(os.environ.get("APPDATA") or str(ROOT))
             / "ClearCommitTaskForge" / "keys.json")


def _load_saved_keys():
    """Keys entered in the UI are stored locally and applied at startup."""
    try:
        data = json.loads(KEYS_FILE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - missing/corrupt file is not fatal
        return
    for name, value in (data or {}).items():
        env = KEY_ENV.get(name)
        if env and isinstance(value, str) and value.strip():
            os.environ[env] = value.strip()


def _credentials():
    return {
        "groq": bool(os.environ.get("GROQ_API_KEY")),
        "deepgram": bool(os.environ.get("DEEPGRAM_API_KEY")),
        "elevenlabs": bool(os.environ.get("ELEVENLABS_API_KEY")),
        "openrouter": bool(os.environ.get("OPENROUTER_API_KEY")),
        "custom": bool(os.environ.get("CUSTOM_AI_API_KEY")),
    }


def _resolve_provider_preset() -> str:
    """TASKFORGE_AI_PRESET: explicit preset, or 'auto' = first preset that
    actually has a key available."""
    wanted = (os.environ.get("TASKFORGE_AI_PRESET") or "auto").strip().lower()
    if wanted != "auto":
        return wanted
    for preset in ("groq", "openrouter", "custom"):
        if _credentials().get(preset):
            return preset
    return "groq"


def _audio_devices():
    """Real device enumeration via sounddevice. Never fabricated."""
    try:
        import sounddevice as sd

        raw = sd.query_devices()
        inputs, outputs = [], []
        for idx, dev in enumerate(raw):
            entry = {"index": idx, "name": str(dev.get("name", ""))}
            if int(dev.get("max_input_channels", 0)) > 0:
                inputs.append(entry)
            if int(dev.get("max_output_channels", 0)) > 0:
                outputs.append(entry)
        return {"inputs": inputs, "outputs": outputs,
                "default_input": int(sd.default.device[0]),
                "default_output": int(sd.default.device[1])}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Device enumeration failed: %s", exc)
        return {"inputs": [], "outputs": [], "default_input": -1,
                "default_output": -1, "error": str(exc)}


class Backend:
    """Owns the pipeline thread, the clarity engine, and the event pump."""

    def __init__(self):
        self.interp = None
        self._worker = None
        self._state = "idle"
        self._paused = False
        self._pending_view = []  # caption events buffered while paused
        self._window = None
        self._stop_poll = threading.Event()
        self.clarity = None
        self._lock = threading.Lock()

    # -- frontend push ---------------------------------------------------- #
    def push(self, payload: dict) -> None:
        if self._window is None:
            return
        try:
            self._window.evaluate_js(
                "window.__push(" + json.dumps(payload, ensure_ascii=False) + ")")
        except Exception:  # noqa: BLE001 - window may be closing
            pass

    def _set_state(self, state: str, message: str = "") -> None:
        with self._lock:
            self._state = state
        self.push(bp.status_event(state, message))

    # -- on_final_utterance hook (pipeline worker thread) ------------------ #
    def _on_final_utterance(self, *, side="p1", original_text="",
                            translated_text="", uid=0, **_ignored) -> None:
        if self.clarity is None:
            return
        try:
            self.clarity.submit(ClaritySource(
                uid=int(uid),
                speaker=str(side),
                original=str(original_text or ""),
                translation=str(translated_text or ""),
            ))
        except Exception:  # noqa: BLE001 - clarity must never break audio
            logger.exception("Clarity submit raised")

    # -- pipeline lifecycle ------------------------------------------------ #
    def start_session(self, cfg: dict) -> str:
        with self._lock:
            if self._state in ("running", "starting"):
                return json.dumps({"ok": False,
                                   "error": "A session is already running."})
            self._state = "starting"
        self._start_cfg = dict(cfg or {})
        self._set_state("starting", "Opening audio devices…")
        # Construction opens devices and can take seconds; keep the bridge
        # responsive by doing it off the caller's thread.
        threading.Thread(target=self._start_worker, name="session-start",
                         daemon=True).start()
        return json.dumps({"ok": True})

    def _start_worker(self) -> None:
        cfg = getattr(self, "_start_cfg", {})
        try:
            from stream_pipeline import StreamingInterpreter

            auto = str(cfg.get("person1_lang") or "es").lower() == "auto"
            person1_lang = "en" if auto else str(cfg.get("person1_lang") or "es")
            mic = cfg.get("input_device")
            out = cfg.get("output_device")
            self.interp = StreamingInterpreter(
                person1_lang,
                user_language=str(cfg.get("user_lang") or "en"),
                mic_device=(int(mic) if mic not in (None, "", -1) else None),
                tts_output_device=(int(out) if out not in (None, "", -1)
                                   else None),
                person1_loopback=True,
                force_language=not auto,
                tts_mode=str(cfg.get("tts_mode") or "utterance"),
                monitor_tts=bool(cfg.get("monitor_tts")),
                groq_api_key=os.environ.get("GROQ_API_KEY"),
                deepgram_api_key=os.environ.get("DEEPGRAM_API_KEY"),
                on_final_utterance=self._on_final_utterance,
            )
            stt = str(cfg.get("stt_backend") or "cloud").lower()
            if stt == "local":
                from local_whisper import LocalWhisperTranscriber
                self.interp.transcriber_p1 = LocalWhisperTranscriber()
                self.interp.transcriber_user = LocalWhisperTranscriber()
                self.push(bp.status_event("starting",
                                          "Local Whisper model loading…"))
            elif str(cfg.get("stt_quality") or "accurate").lower() == "fast":
                # Groq: whisper-large-v3-turbo instead of whisper-large-v3
                self.interp.set_stt_model(True, side="p1")
                self.interp.set_stt_model(True, side="user")
                self.push(bp.status_event("starting",
                                          "STT model: fast (turbo)"))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Pipeline construction failed")
            with self._lock:
                self._state = "idle"
            self._set_state("error",
                            f"Could not start audio pipeline: {exc}")
            return

        self._make_clarity_engine()
        self._stop_poll.clear()
        self._worker = threading.Thread(
            target=self._run_pipeline, name="pipeline", daemon=True)
        self._worker.start()
        threading.Thread(target=self._poll_loop, name="event-pump",
                         daemon=True).start()
        self._set_state("running")

    def _run_pipeline(self) -> None:
        try:
            self.interp.run()
            self._set_state("stopped")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Pipeline crashed")
            self._set_state("error", f"Audio pipeline failed: {exc}")

    def _make_clarity_engine(self) -> None:
        """Connect the clarity engine when a provider is configured."""
        try:
            from ai_service import AIService
            from ai_providers import ProviderConfig, get_provider_for_config

            preset = (os.environ.get("TASKFORGE_AI_PRESET") or "groq").lower()
            pconf = ProviderConfig(preset=preset)
            if not pconf.is_configured():
                self.clarity = None
                self.push(bp.hello_payload(
                    _list_languages(), _credentials(), False))
                return
            service = AIService(max_concurrency=1, max_queue=2,
                                default_timeout_s=10.0)
            service.set_provider(get_provider_for_config(pconf))
            self.clarity = ClarityEngine(
                service,
                on_finding=lambda f: self.push(bp.clarity_event(f)),
            )
            self.push(bp.hello_payload(
                _list_languages(), _credentials(), True))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Clarity engine unavailable: %s", exc)
            self.clarity = None

    # -- event pump -------------------------------------------------------- #
    def _poll_loop(self) -> None:
        while not self._stop_poll.is_set() and self._window is not None:
            try:
                if self.interp is not None and not self._paused:
                    for tup in self.interp.pop_captions():
                        try:
                            event = bp.caption_event(*tup)
                        except (TypeError, ValueError):
                            continue
                        self.push(event)
                if self.clarity is not None:
                    self.clarity.poll()
            except Exception:  # noqa: BLE001 - pump must survive anything
                logger.exception("Event pump iteration failed")
            time.sleep(0.08)

    # -- bridge (whitelisted js_api) ---------------------------------------- #
    def hello(self) -> str:
        return json.dumps(bp.hello_payload(
            _list_languages(), _credentials(), self.clarity is not None))

    def list_audio_devices(self) -> str:
        return json.dumps(_audio_devices())

    def save_keys(self, keys: dict) -> str:
        """Persist UI-entered API keys locally (plaintext, user profile) and
        apply them to the running process. Values are never returned."""
        saved = {}
        for name, value in (keys or {}).items():
            env = KEY_ENV.get(name)
            if env and isinstance(value, str) and value.strip():
                os.environ[env] = value.strip()
                saved[name] = value.strip()
        if saved:
            try:
                KEYS_FILE.parent.mkdir(parents=True, exist_ok=True)
                existing = {}
                try:
                    existing = json.loads(KEYS_FILE.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001
                    existing = {}
                existing.update(saved)
                KEYS_FILE.write_text(json.dumps(existing), encoding="utf-8")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not persist keys: %s", exc)
                return json.dumps({"ok": False,
                                   "error": f"Could not save keys: {exc}"})
        if self.clarity is None:
            self._make_clarity_engine()
        creds = _credentials()
        self.push(bp.hello_payload(_list_languages(), creds,
                                   self.clarity is not None))
        restart_hint = ("Keys saved. Start (or restart) the session to use "
                        "them." if not saved else "Keys saved.")
        return json.dumps({"ok": True, "credentials": creds,
                           "message": restart_hint})

    def stop_session(self) -> str:
        try:
            if self.interp is not None:
                self.interp.stop()
            self._set_state("stopped")
            return json.dumps({"ok": True})
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"ok": False, "error": str(exc)})

    def set_paused(self, paused: bool) -> str:
        self._paused = bool(paused)
        self._set_state("paused" if self._paused else "running")
        return json.dumps({"ok": True})

    def clarify_approve(self, finding_id: str, text: str) -> str:
        if self.clarity is None:
            return json.dumps({"ok": False, "error": "clarity_off"})
        finding = self.clarity.approve(finding_id, text)
        if finding is None:
            return json.dumps({"ok": False, "error": "unknown_or_resolved"})
        self.push(bp.clarity_resolved_event(
            finding_id, "approved", finding.approved_text))
        return json.dumps({"ok": True})

    def clarify_dismiss(self, finding_id: str) -> str:
        if self.clarity is None:
            return json.dumps({"ok": False, "error": "clarity_off"})
        ok = self.clarity.dismiss(finding_id)
        if ok:
            self.push(bp.clarity_resolved_event(finding_id, "dismissed"))
        return json.dumps({"ok": ok})

    def shutdown(self) -> None:
        self._stop_poll.set()
        if self.clarity is not None:
            self.clarity.close()
        if self.interp is not None:
            try:
                self.interp.stop()
            except Exception:  # noqa: BLE001
                pass


backend = Backend()


def selftest() -> int:
    """Headless verification that the frozen build has everything it needs."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(name)s | %(message)s")
    _load_saved_keys()
    ok = True
    for mod in ("wave", "numpy", "requests", "sounddevice", "webview"):
        try:
            __import__(mod)
            print(f"OK   {mod}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"FAIL {mod}: {exc}")
    try:
        from stream_pipeline import StreamingInterpreter  # noqa: F401
        print("OK   stream_pipeline")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"FAIL stream_pipeline: {exc}")
    devices = _audio_devices()
    print(f"INFO devices: {len(devices.get('inputs', []))} in / "
          f"{len(devices.get('outputs', []))} out, error={devices.get('error')}")
    creds = _credentials()
    print(f"INFO credentials: {creds}")
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv=None):
    argv = list(argv if argv is not None else sys.argv[1:])
    if "--selftest" in argv:
        return selftest()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(name)s | "
                               "%(message)s")
    _load_saved_keys()
    window = webview.create_window(
        "ClearCommit — live interpretation",
        str(UI_DIR / "index.html"),
        js_api=backend,
        width=1280, height=800, min_size=(980, 640),
        background_color="#121417",
    )
    backend._window = window
    try:
        webview.start(debug=False)
    finally:
        backend.shutdown()


if __name__ == "__main__":
    main()
