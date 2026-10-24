"""Live streaming interpreter: two audio sources, no manual direction switch.

The design you described:
  * Person 1  = audio coming FROM the system (the Discord call playing through
    your machine). Captured via a **loopback** input device.
  * You (User) = audio from your **microphone**.

So identity is decided by the *source*, never by a manual flag:
  system/Discord audio -> STT -> translate to English -> captions (for you)
  your mic            -> STT -> translate to Person 1's lang -> TTS -> Discord
                          (played to a virtual microphone so Person 1 hears it;
                           this also avoids speaker->mic feedback)

Groq STT is file-based, so each source flushes detected utterances (VAD) to it;
translation (NVIDIA) and TTS (Deepgram/ElevenLabs) are genuinely streamed.
"""

from __future__ import annotations

import io
import itertools
import os
import queue
import re
import tempfile
import threading
import time
import wave
from collections import deque
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# Tuning for false-trigger / hallucination suppression.
# Captures whose *average* RMS energy is below this are treated as silence or
# background noise and dropped before they reach Whisper (which otherwise
# fabricates "Thank you" / "Grazie" style hallucinations on quiet audio).
MIN_SPEECH_RMS = 0.008
# Voice-activity threshold used to start an utterance (lower = more sensitive
# to distant/quiet speech; raise if background noise triggers false captures).
VAD_THRESHOLD = 0.008
# Whisper's classic low-energy hallucinations. Dropped before display/translate
# so they don't loop or pollute captions.
HALLUCINATION_PHRASES = {
    "thank you", "thanks for watching", "gracias", "merci", "obrigado",
    "danke", "arigato", "subscrib", "please subscribe",
}
# Two-tier hallucination filter (matching is punctuation/case-insensitive —
# Whisper always writes "Thank you." with the period, which the old exact
# match never caught).
# YouTube/music filler + the notorious "Thank you." family: dropped
# UNCONDITIONALLY — they are the #1 Whisper hallucination and the user
# explicitly wants them never translated or voiced, even when genuinely said.
_HALLUCINATION_EXACT = {
    "thanks for watching", "thank you for watching", "thanks for the video",
    "please subscribe", "subscribe", "subscribe to the channel",
    "see you in the next video", "see you next time", "see you next video",
    "thank you and bye", "thank you bye", "thanks for watching bye",
    "gracias", "merci", "obrigado", "danke", "arigato",
    # ONLY the bare hallucination words (user request):
    "thank you", "thanks", "bye", "bye bye",
}


def _normalize_hallucination(text: str) -> str:
    """Lowercase, strip punctuation/extra spaces: 'Thank you.' -> 'thank you'."""
    t = text.strip().lower()
    t = re.sub(r"[^\w\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()
# After TTS finishes, ignore captures for this long. Covers the acoustic echo
# latency of speaker -> mic feedback (real fix: route TTS to a virtual mic).
POST_TTS_COOLDOWN = 2.0  # echo guard; covers Discord round-trip latency too
# Your own TTS can physically leak into your microphone (speakers open,
# "Hear TTS locally" on, or a headset bleeding sound). Whisper then
# transcribes the PC's voice as YOURS — short hallucinated utterances like
# "you"/"Lei". Gating the mic during playback plus a short cooldown removes
# that echo while keeping post-TTS barge-in delay minimal.
MIC_ECHO_COOLDOWN = 1.0  # only used when mic_echo_guard is ON
# TTS jobs older than this (utterance accepted -> dequeue) are dropped instead
# of spoken — stale speech after a provider stall confuses everyone.
MAX_TTS_QUEUE_AGE = 10.0

from audio_capture import MicCapture, resolve_device
from config import (
    DEEPGRAM_API_KEY,
    GROQ_TRANSCRIBE_MODEL,
    GROQ_TRANSCRIBE_MODEL_FAST,
    GROQ_TRANSLATE_MODELS,
    REALTIME_TTS_BACKEND,
    Settings,
    groq_api_keys,
    language_name_to_code,
    select_realtime_tts_backend,
    select_translate_backend,
)
from deepgram_tts import DeepgramStreamTTS
from elevenlabs_ws import ElevenLabsStreamTTS, resolve_voice_id
from logger import get_logger
from nvidia_stream import NvidiaStreamTranslator
from translator import Translator

from transcriber import Transcriber, TranscriptionResult

logger = get_logger("stream_pipeline")


def _is_repeat_gibberish(text: str) -> bool:
    """True for Whisper-turbo style repetition hallucinations.

    Fingerprints: 3+ consecutive copies of the same word ("aiel aiel aiel",
    "αελ αελ αελ") or a 2-4 char block repeated 3+ times inside a word
    ("blablabla", "αελαελαελ"). Whisper fabricates these on unclear/quiet
    audio and the translator renders them as confident nonsense, so drop the
    utterance before it reaches display, translation, or TTS.
    """
    low = text.lower()
    words = re.findall(r"\S+", low)
    if len(words) >= 3:
        for i in range(len(words) - 2):
            if words[i] == words[i + 1] == words[i + 2]:
                return True
    for word in words:
        if len(word) >= 6 and re.search(r"(\S{2,4}?)\1{2,}", word):
            return True
    return False


# Sample rates ElevenLabs accepts for raw PCM output on its WS endpoint.
# NOTE: pcm_44100 is NOT allowed by the stream-input WS
# ("output_format_not_allowed"), so it must never be requested.
_ELEVEN_PCM_RATES = (16000, 22050, 24000, 48000)


def _nearest_eleven_rate(rate: int) -> int:
    return min(_ELEVEN_PCM_RATES, key=lambda r: abs(r - int(rate)))


class StreamingInterpreter:
    def __init__(
        self,
        person1_language: str,
        user_language: str = "en",
        *,
        device: Optional[int] = None,
        mic_device: Optional[int] = None,
        person1_device: Optional[int] = None,
        tts_output_device: Optional[int] = None,
        person1_loopback: bool = True,
        system_device: Optional[int] = None,
        voice_id: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        nvidia_api_key: Optional[str] = None,
        eleven_api_key: Optional[str] = None,
        deepgram_api_key: Optional[str] = None,
        realtime_backend: Optional[str] = None,
        translate_backend: Optional[str] = None,
        tts_mode: str = "utterance",
        noise_cancel: bool = False,
        # Back to the proven 0.7s: 0.45 chopped words on mid-sentence pauses
        # and fragmented utterances hurt Whisper accuracy more than the
        # latency win helped. Tune live with --silence-timeout.
        silence_timeout: float = 0.7,
        force_language: bool = True,
        elevenlabs_stability: float = 0.55,
        elevenlabs_similarity: float = 0.4,
        monitor_tts: bool = False,
        monitor_device: Optional[int] = None,
        consent_chime: bool = False,
        on_final_utterance: Optional[Callable] = None,
        mic_echo_guard: bool = False,
    ):
        self.person1_language = Settings.validate_language(person1_language)
        self.user_language = Settings.validate_language(user_language)

        # Audio routing. `person1_device` is the loopback input that captures
        # the system/Discord audio; `mic_device` is your microphone;
        # `tts_output_device` is where your translated speech is played (a
        # virtual microphone so Discord hears it, not your speakers).
        # Device args accept either an integer index or a name substring
        # (e.g. "CABLE Input"); resolve to a stable index here so the config
        # survives the OS re-enumerating devices.
        self.mic_device = resolve_device(
            mic_device if mic_device is not None else device, "input"
        )
        self.person1_device = resolve_device(person1_device, "input")
        self.tts_output_device = resolve_device(tts_output_device, "output")
        # Person 1 capture: by default use WASAPI loopback (captures whatever you
        # hear, incl. through a USB headset). Fall back to a recording device
        # (e.g. Stereo Mix) when person1_loopback is False.
        self.person1_loopback = person1_loopback
        self.system_device = resolve_device(system_device, "output")

        # Stored so we can re-resolve the TTS backend when the language changes.
        self._eleven_key = eleven_api_key
        self._deepgram_key = deepgram_api_key or DEEPGRAM_API_KEY
        self._voice_id = voice_id
        self._realtime_backend = realtime_backend
        self._translate_backend = select_translate_backend(translate_backend)
        self._tts_mode = tts_mode if tts_mode in ("utterance", "streaming") else "utterance"
        self._groq_key = groq_api_key or (groq_api_keys() or [None])[0]
        self._groq_keys = groq_api_keys()
        self._nvidia_key = nvidia_api_key or os.environ.get("NVIDIA_KEY_1")
        self._eleven_stability = elevenlabs_stability
        self._eleven_similarity = elevenlabs_similarity
        # Local TTS monitor: duplicate translated speech to your own speakers
        # (like Windows "Listen to this device", but in-app + toggleable).
        self._monitor_tts = monitor_tts
        self._monitor_device = monitor_device
        self._monitor_stream = None
        self._monitor_samplerate = None
        self._monitor_channels = None
        # Fair-play notice: announce at session start that the call is
        # auto-translated (toggleable; off = silent).
        self._consent_chime = consent_chime
        self._chime_done = False
        self._on_final_utterance = on_final_utterance
        # Drop mic captures while our own TTS is playing (speaker leakage
        # made Whisper transcribe the PC's voice as the user's).
        self._mic_echo_guard = bool(mic_echo_guard)
        # Per-service health reporting + language mismatch callback (wired
        # to the ClearCommit integration when connected; no-ops otherwise).
        self._service_health_cb = None
        self._lang_mismatch_cb = None
        # Krisp-style spectral gating on YOUR microphone so Whisper doesn't
        # hallucinate on steady fan/AC background noise.
        self._noise_cancel = noise_cancel
        # Seconds of silence that end an utterance (lower = snappier responses,
        # too low splits mid-thought pauses).
        self._silence_timeout = silence_timeout
        # True: force Whisper to the configured Person 1 language. False:
        # auto-detect what's actually spoken (Telugu, ...) and use that as
        # the translation source.
        self._force_language = force_language
        # End-to-end latency stats for the GUI (utterance accept -> first TTS
        # audio). Passive: written by worker/capture threads, polled by GUI.
        self._uid_t0: dict[int, float] = {}
        self._latencies: "deque[float]" = deque(maxlen=20)
        self.last_latency: Optional[float] = None

        # Both sides default to whisper-large-v3 (accuracy). Turbo (fast but
        # noticeably less accurate) remains selectable via the GUI dropdown /
        # set_stt_model(). Keys rotate across all Groq keys.
        # STT backend is configurable: "groq" (Whisper) or "deepgram" (Nova-3).
        self._stt_backend = os.environ.get("STT_BACKEND", "groq").lower()
        if self._stt_backend == "deepgram":
            from transcriber import DeepgramTranscriber
            self.transcriber_p1 = DeepgramTranscriber()
            self.transcriber_user = DeepgramTranscriber()
            logger.info("STT backend: Deepgram Nova-3 (multi-language)")
        else:
            self.transcriber_p1 = Transcriber(
                api_keys=self._groq_keys, model=GROQ_TRANSCRIBE_MODEL
            )
            self.transcriber_user = Transcriber(
                api_keys=self._groq_keys, model=GROQ_TRANSCRIBE_MODEL
            )
            logger.info("STT backend: Groq Whisper large-v3")
        # Backward-compatible alias (legacy callers use ``self.transcriber``).
        self.transcriber = self.transcriber_p1
        self.translator = self._build_translator()

        # Realtime TTS: Groq is not streaming-capable, so we use Deepgram Aura
        # by default and fall back to ElevenLabs for languages Aura lacks.
        self._tts_lang = None
        # Native PCM rate the engine was built for (None = backend default).
        self._tts_rate: Optional[int] = None
        self.tts = None
        self._ensure_tts(self.person1_language)

        # Live control hooks (used by the GUI / Stop button).
        self._stop_ev = threading.Event()
        self._captures: list[MicCapture] = []
        # Per-side utterance workers: capture threads only enqueue finalized
        # utterances; STT + translation run here so a slow provider call never
        # delays detection of the next utterance (per-side FIFO keeps order).
        self._work_qs: "dict[str, queue.Queue]" = {}
        self._translate_qs: dict = {"p1": queue.Queue(), "you": queue.Queue()}
        self._side_stop: "dict[str, threading.Event]" = {}

        # Echo guard: ignore input while TTS is playing (prevents the speaker's
        # own output from being re-transcribed into garbage), plus a cooldown
        # afterwards for delayed acoustic echo.
        self._tts_active = False
        self._last_tts_end = 0.0
        # Actual sample rate of the TTS output stream (device native); TTS PCM
        # is resampled to this so playback isn't at the wrong speed.
        self._tts_out_samplerate = None
        # Actual channel count of the TTS output stream (device native). Mono
        # TTS PCM is duplicated to this count (e.g. stereo VB-Cable).
        self._tts_out_channels = None

        # Queue of caption events for the GUI:
        #   ("orig", side, lang, text, uid)   side in ("p1", "you")
        #   ("trans", side, lang, text, uid)
        self._caption_q: "queue.Queue[tuple]" = queue.Queue()
        # Per-utterance id so the GUI can pair each translation with the exact
        # original it belongs to (translations arrive asynchronously).
        self._uid_counter = itertools.count(1)

        # TTS job queue (parallel TTS): the capture thread enqueues
        # (target_lang, translated_text) jobs and returns to capturing
        # immediately; a dedicated worker speaks them FIFO. This lets your next
        # sentence be STT'd + translated while the previous one is still
        # playing, instead of everything being serial + dropped.
        self._tts_job_q: "queue.Queue[Optional[tuple]]" = queue.Queue()
        self._tts_worker = None

    def _build_translator(self):
        """Construct the streaming translation backend selected by the user."""
        if self._translate_backend == "nvidia":
            return NvidiaStreamTranslator(self._nvidia_key)
        # Default: Groq gpt-oss-120b (fast + accurate); keys rotate round-robin.
        return Translator(api_keys=self._groq_keys)

    # ------------------------------------------------------------------ #
    def set_person1_language(self, code: str) -> None:
        """Switch Person 1's language at runtime (applies next utterance)."""
        self.person1_language = Settings.validate_language(code)
        logger.info("Language switched (next utterance): person1=%s", self.person1_language)

    def set_translate_model(self, model: str) -> None:
        """Set the Groq translation model at runtime (GUI dropdown)."""
        model = GROQ_TRANSLATE_MODELS.get(model, model)
        if isinstance(self.translator, Translator):
            self.translator.model = model
            logger.info("Translate model -> %s", model)

    def set_stt_model(self, fast: bool = False, *, side: str = "user") -> None:
        """Swap the STT model: fast (turbo) or accurate (large-v3)."""
        model = GROQ_TRANSCRIBE_MODEL_FAST if fast else GROQ_TRANSCRIBE_MODEL
        target = self.transcriber_user if side == "user" else self.transcriber_p1
        target.model = model
        logger.info("STT model (%s) -> %s", side, model)

    def set_voice(self, voice_id: str) -> None:
        """Hot-swap the ElevenLabs voice (applies on the next TTS job)."""
        if not voice_id or voice_id == self._voice_id:
            return
        self._voice_id = voice_id
        # Force an engine rebuild so the next job uses the new voice.
        self._tts_lang = None
        logger.info("TTS voice -> %s (next utterance)", voice_id)

    def list_voices(self) -> list[dict]:
        """Account voices for the GUI picker: [{id, name, desc}]."""
        import requests as _rq

        try:
            resp = _rq.get(
                "https://api.elevenlabs.io/v1/voices",
                headers={"xi-api-key": self._eleven_key},
                timeout=15,
            )
            out = []
            for v in resp.json().get("voices", []):
                labels = v.get("labels", {}) or {}
                bits = [labels.get(k) for k in
                        ("gender", "descriptive", "age", "accent") if labels.get(k)]
                out.append({
                    "id": v["voice_id"],
                    "name": v["name"],
                    "desc": " · ".join(bits),
                })
            return out
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not list voices: %s", exc)
            return []

    def stop(self) -> None:
        """Signal the running loop(s) to finish and release audio devices."""
        self._stop_ev.set()
        for c in self._captures:
            try:
                c.stop()
            except Exception:  # noqa: BLE001
                pass
        self._close_monitor()

    def _ensure_tts(self, lang: str, samplerate: Optional[int] = None) -> None:
        """(Re)build the realtime TTS engine if the target language changed.

        ``samplerate`` (the TTS output device's native rate) makes the engine
        emit PCM in that exact rate, so playback needs zero resampling —
        chunk-wise resampling would click at chunk boundaries during
        play-while-generating.
        """
        want_rate = int(samplerate) if samplerate else None
        if (
            self.tts is not None
            and self._tts_lang == lang
            and self._tts_rate == want_rate
        ):
            return
        rt_backend = select_realtime_tts_backend(lang, preferred=self._realtime_backend)
        if rt_backend == "deepgram":
            kwargs = {"sample_rate": want_rate} if want_rate else {}
            self.tts = DeepgramStreamTTS(
                api_key=self._deepgram_key, target_language=lang, **kwargs
            )
            logger.info(
                "Realtime TTS -> Deepgram Aura model=%s rate=%s",
                self.tts.model, self.tts.sample_rate,
            )
        else:
            if (self._realtime_backend or REALTIME_TTS_BACKEND).lower() == "deepgram":
                logger.warning(
                    "Deepgram Aura has no voice for '%s'; using ElevenLabs.", lang
                )
            vid = resolve_voice_id(self._eleven_key, preferred=self._voice_id)
            fmt = f"pcm_{_nearest_eleven_rate(want_rate)}" if want_rate else "pcm_24000"
            self.tts = ElevenLabsStreamTTS(
                voice_id=vid,
                api_key=self._eleven_key,
                output_format=fmt,
                stability=self._eleven_stability,
                similarity_boost=self._eleven_similarity,
            )
            logger.info("Realtime TTS -> ElevenLabs voice id=%s fmt=%s", vid, fmt)
        self._tts_lang = lang
        self._tts_rate = want_rate

    def pop_captions(self) -> list[str]:
        """Drain queued caption lines (for the GUI). Non-blocking."""
        lines: list[str] = []
        while True:
            try:
                lines.append(self._caption_q.get_nowait())
            except queue.Empty:
                break
        return lines

    # ------------------------------------------------------------------ #
    def _transcribe(self, wav_bytes: bytes, *, speak: bool = False,
                    language: Optional[str] = None) -> TranscriptionResult:
        """Transcribe a WAV with the per-role model (turbo for you, v3 for P1).

        ``language`` (ISO-639-1) locks Whisper to that language; ``None``
        lets it auto-detect (Telugu, etc.). Transient provider errors are
        retried (a network blip must not eat a spoken sentence); genuinely
        empty audio fails fast.
        """
        path = Path(tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name)
        path.write_bytes(wav_bytes)
        transcriber = self.transcriber_user if speak else self.transcriber_p1
        try:
            delay = 0.3
            last_exc: Optional[Exception] = None
            for attempt in range(1, 3):
                try:
                    return transcriber.transcribe(path, language=language)
                except Exception as exc:  # noqa: BLE001
                    if "empty text" in str(exc).lower():
                        raise  # silent/noisy audio — retrying cannot help
                    last_exc = exc
                    if attempt < 2:
                        logger.warning(
                            "STT attempt %d failed (%s); retrying in %.1fs",
                            attempt, exc, delay,
                        )
                        time.sleep(delay)
                        delay *= 2
            raise last_exc  # type: ignore[misc]
        finally:
            path.unlink(missing_ok=True)

    def _translate_with_retry(self, text: str, *, target: str, source: str,
                              attempts: int = 3) -> str:
        """Full translation with bounded retries; returns the joined text."""
        delay = 0.6
        last_exc: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                iterator = self.translator.translate_stream(
                    text, target_language=target, source_language=source
                )
                return "".join(iterator).strip()
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < attempts:
                    logger.warning(
                        "Translate attempt %d/%d failed (%s); retrying in %.1fs",
                        attempt, attempts, exc, delay,
                    )
                    time.sleep(delay)
                    delay *= 2
        raise last_exc  # type: ignore[misc]

    @staticmethod
    def _avg_rms(wav_bytes: bytes) -> float:
        """Average RMS (0..1) of a 16-bit mono WAV, used to drop silence."""
        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as w:
                raw = w.readframes(w.getnframes())
        except Exception:  # noqa: BLE001
            return 0.0
        if not raw:
            return 0.0
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if samples.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))

    # ------------------------------------------------------------------ #
    def _process_utterance(
        self, wav, *, source, target, speak, max_utterances, count
    ) -> int:
        """STT only - emits the original caption immediately and hands off
        to the translation pipeline. Returns the updated count.

        STT and translation run on SEPARATE pipeline stages so a slow
        translation never delays the next utterance's recognition.
        """
        stt_lang = source if self._force_language else None
        result = self._transcribe(wav, speak=speak, language=stt_lang)
        text = (result.text or "").strip()
        if not self._force_language:
            detected = language_name_to_code(getattr(result, "language", None))
            if detected:
                logger.info("STT auto-detected language: %s", detected)
                source = detected
        if not text:
            return count
        norm = _normalize_hallucination(text)
        if norm in _HALLUCINATION_EXACT:
            logger.info("Dropped hallucination: %r", text)
            return count
        if _is_repeat_gibberish(text):
            logger.info("Dropped repetition hallucination: %r", text)
            return count
        count += 1
        uid = next(self._uid_counter)
        self._uid_t0[uid] = time.time()
        side = "you" if speak else "p1"
        print(f"\U0001f5e3  [{source}] {text}")
        logger.info("UID %d [%s] %r -> translate to %s", uid, side, text, target)
        self._caption_q.put(("orig", side, source, text, uid))

        # Language mismatch detection (additive, never changes behaviour).
        if self._on_final_utterance is not None:
            try:
                detected_code = language_name_to_code(getattr(result, "language", None))
                if detected_code and detected_code != source:
                    hint = self._lang_mismatch_cb(
                        uid, detected_code, source) if self._lang_mismatch_cb else None
                    if hint:
                        self._caption_q.put(
                            ("error", side, target, hint[:160], 0))
            except Exception:
                pass

        self._report_service_health("stt", True)

        # Hand off to the translation stage (non-blocking for STT worker).
        self._translate_qs[side].put((uid, text, source, target, speak))
        return count

    def _start_translate_loops(self) -> list:
        """Per-side translation workers (parallel across sides).

        Each side gets its own FIFO translation worker, so P1 and You
        translate independently. TTS remains FIFO via _tts_job_q.
        """
        threads = []
        for side in ("p1", "you"):
            q = queue.Queue()
            self._translate_qs[side] = q

            def make_worker(sid, tq):
                def worker() -> None:
                    while True:
                        try:
                            item = tq.get(timeout=0.2)
                        except queue.Empty:
                            if self._stop_ev.is_set():
                                return
                            continue
                        if item is None or self._stop_ev.is_set():
                            return
                        uid, text, source, target, speak = item
                        try:
                            translated = self._translate_with_retry(
                                text, target=target, source=source
                            )
                            self._report_service_health("translate", True)
                        except Exception as exc:
                            logger.error("Translation failed uid=%d: %s", uid, exc)
                            self._caption_q.put(
                                ("error", sid, target,
                                 "Translation failed: " + str(exc)[:160], 0))
                            self._report_service_health("translate", False)
                            self._tts_lang = None
                            continue
                        if speak:
                            if translated:
                                self._tts_job_q.put(
                                    (sid, target, translated, uid))
                            else:
                                self._caption_q.put(
                                    ("trans", sid, target, translated, uid))
                        else:
                            self._caption_q.put(
                                ("trans", sid, target, translated, uid))
                        if self._on_final_utterance is not None:
                            try:
                                self._on_final_utterance(
                                    side=sid,
                                    source_lang=source,
                                    target_lang=target,
                                    original_text=text,
                                    translated_text=translated,
                                    uid=uid,
                                    accept_ts=self._uid_t0.get(uid),
                                )
                            except Exception:
                                logger.exception(
                                    "ClearCommit on_final_utterance failed")
                return worker

            t = threading.Thread(target=make_worker(side, q),
                                 daemon=True, name=f"translate-{side}")
            t.start()
            threads.append(t)
        return threads

    def _tts_worker_loop(self, player) -> None:
        """Speak enqueued (side, lang, text) jobs FIFO on the output stream."""
        while True:
            job = self._tts_job_q.get()
            if job is None or self._stop_ev.is_set():
                return
            side, lang, text, uid = job
            # Stale-job guard: after a provider stall the queue can hold
            # minutes-old sentences — speaking them late is worse than
            # dropping them. The caption still fills its placeholder.
            t0 = self._uid_t0.get(uid)
            if t0 is not None and (time.time() - t0) > MAX_TTS_QUEUE_AGE:
                logger.warning(
                    "TTS uid=%d dropped: queued %.1fs ago (> %.0fs cap)",
                    uid, time.time() - t0, MAX_TTS_QUEUE_AGE,
                )
                self._uid_t0.pop(uid, None)
                self._caption_q.put(("trans", side, lang, text, uid))
                continue
            t_start = time.time()
            try:
                self._ensure_tts(lang, samplerate=self._tts_out_samplerate)
                # One-time consent notice: P1 hears that the call is translated.
                if self._consent_chime and not self._chime_done:
                    self._chime_done = True
                    try:
                        notice = ("Heads up: this call is auto-translated. "
                                  "Everything you say is spoken in another language.")
                        logger.info("Consent notice playing.")
                        self._speak_utt(player, notice, uid=-1)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Consent notice failed: %s", exc)
                logger.info(
                    "TTS speaking uid=%d lang=%s engine=%s fmt=%s out=%sHz/%dch",
                    uid, lang, type(self.tts).__name__,
                    getattr(self.tts, "output_format", "-"),
                    self._tts_out_samplerate, self._tts_out_channels or 1,
                )
                self._tts_active = True
                if self._tts_mode == "utterance":
                    # One-shot full-utterance generation: the engine sees the
                    # whole sentence/paragraph so prosody and emotion stay
                    # continuous — but the audio PLAYS AS IT ARRIVES instead of
                    # waiting for generation to finish (cuts latency; the voice
                    # quality/prosody are identical).
                    self._speak_utt(player, text, uid=uid)
                else:
                    # Legacy streaming (lower latency, more seams): split at
                    # boundaries so wording/emotion stay natural.
                    for chunk in self._tts_sentence_chunks(text):
                        raw = self.tts.synthesize(chunk)
                        self._write_monitor(raw)
                        self._write_audio(player, self._adapt_audio(raw))
            except Exception as exc:  # noqa: BLE001
                logger.error("TTS job failed uid=%d after %.2fs: %s",
                             uid, time.time() - t_start, exc)
                self._caption_q.put(
                    ("error", side, lang, f"TTS failed: {exc}"[:160], 0)
                )
                self._tts_lang = None  # force engine rebuild next job
            finally:
                self._tts_active = False
                self._last_tts_end = time.time()
            logger.info("TTS finished uid=%d in %.2fs", uid, time.time() - t_start)
            self._caption_q.put(("trans", side, lang, text, uid))
            print()  # newline after spoken audio

    def _speak_utt(self, player, text: str, uid: Optional[int] = None) -> None:
        """Play a full-utterance TTS request chunk-by-chunk as it generates.

        The engine still sees the whole utterance (natural prosody), but PCM
        chunks are written to the speakers as they arrive instead of waiting
        for generation to complete. Retries once only if NOTHING was heard —
        a mid-stream failure can't be retried without double-speaking.
        """
        played = False
        for attempt in range(2):
            n_chunks = n_bytes = 0
            t0 = time.time()
            try:
                for chunk in self.tts.stream(iter([text])):
                    if not played:
                        logger.info(
                            "TTS uid=%d first audio in %.2fs", uid or -1, time.time() - t0
                        )
                        started = self._uid_t0.pop(uid, None) if uid else None
                        if started is not None:
                            self.last_latency = time.time() - started
                            self._latencies.append(self.last_latency)
                    played = True
                    n_chunks += 1
                    n_bytes += len(chunk)
                    self._write_monitor(chunk)  # local self-hear (toggle)
                    self._write_audio(player, self._adapt_audio(chunk))
                logger.info(
                    "TTS uid=%d streamed %d chunks / %d bytes in %.2fs",
                    uid or -1, n_chunks, n_bytes, time.time() - t0,
                )
                return
            except Exception as exc:  # noqa: BLE001
                if not played and attempt == 0:
                    logger.warning("TTS failed before any audio, retrying once: %s", exc)
                    self._tts_lang = None  # force engine rebuild next attempt/job
                    continue
                raise

    def _tts_sentence_chunks(self, text: str, max_len: int = 280) -> list[str]:
        """Split long translations into sentence-bounded chunks (<=max_len)."""
        if len(text) <= max_len:
            return [text]
        out, buf = [], ""
        for piece in re.split(r"(?<=[.!?…:;])\s+", text):
            if len(buf) + len(piece) + 1 > max_len and buf:
                out.append(buf.strip())
                buf = ""
            buf += piece + " "
        if buf.strip():
            out.append(buf.strip())
        return out or [text]

    # ------------------------------------------------------------------ #
    def _report_service_health(self, service: str, ok: bool) -> None:
        """Report a per-service call outcome to the integration callback."""
        if self._service_health_cb is not None:
            try:
                self._service_health_cb(service, ok)
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------ #
    @staticmethod
    def _should_drop_for_echo(role: Optional[str], tts_active: bool,
                              last_tts_end: float, now: float,
                              mic_echo_guard: bool = False) -> bool:
        """Echo-guard decision (pure; unit-testable).

        * Person 1 (loopback) always drops while TTS plays and for
          ``POST_TTS_COOLDOWN`` after — the loopback hears the translated
          voice through the call.
        * Your mic is NEVER blocked by default. The hallucination filter
          already catches TTS artifacts. Enable mic_echo_guard ONLY if you
          have speaker leakage and want to drop TTS echoes at the cost of
          not being able to talk while TTS plays.
        """
        if role == "p1":
            return tts_active or (now - last_tts_end) < POST_TTS_COOLDOWN
        if role == "user" and mic_echo_guard:
            return tts_active or (now - last_tts_end) < MIC_ECHO_COOLDOWN
        return False

    def _capture_loop(self, capture, *, source, target, speak, max_utterances,
                      role: Optional[str] = None):
        """Run one source's CAPTURE loop until stopped.

        Smoothing architecture: this thread only does time-sensitive work —
        echo guard, silence drop, live-language resolution — and then enqueues
        the finalized utterance WAV onto this side's work queue. Transcription
        + translation run on a dedicated per-side worker thread
        (``_work_loop``), so a slow Whisper call never delays the next
        utterance's detection or the other side's processing. Per-side FIFO
        workers preserve caption and TTS ordering within each side.

        In two-way mode ``role`` is ``"p1"`` or ``"user"`` and the source/target
        languages are re-read from the live ``self.person1_language`` on every
        utterance, so a GUI language change takes effect on the next sentence.
        With ``role=None`` (legacy single-mic) the fixed ``source``/``target``
        passed in apply.

        Echo guard: only Person 1's loopback thread drops captures while TTS
        plays (it can hear the translated speech through the call). YOUR mic
        thread never drops — TTS goes to CABLE Input, which cannot echo back
        into your headset mic, so your speech is always transcribed and queued.
        """
        side = role or ("user" if speak else "p1")
        try:
            capture.role = role or ("user" if speak else "p1")
            capture.start()
            logger.info("Capture started: source=%s device=%s speak=%s", source, capture.device, speak)
            self._captures.append(capture)
            for wav in capture.utterances():
                if self._stop_ev.is_set() or self._side_stop[side].is_set():
                    break
                # Echo guard (P1 loopback always; your mic when enabled) —
                # drops captures of our own TTS voice leaking into inputs.
                if self._should_drop_for_echo(
                    role, self._tts_active, self._last_tts_end, time.time(),
                    self._mic_echo_guard,
                ):
                    continue
                # Live language: follow the current Person 1 language mid-call.
                if role == "user":
                    source, target = self.user_language, self.person1_language
                elif role == "p1":
                    source, target = self.person1_language, self.user_language
                # Drop silence / background noise before hitting Whisper.
                rms = self._avg_rms(wav)
                if rms < MIN_SPEECH_RMS:
                    logger.debug("Dropped low-energy capture (rms=%.4f) — silence/noise", rms)
                    continue
                # Hand off to the per-side worker; the capture loop returns to
                # listening immediately (speech is never dropped — the queue
                # is unbounded and humanly bounded by speaking rate).
                self._work_qs[side].put((wav, source, target, speak))
        finally:
            try:
                capture.stop()
            except Exception:  # noqa: BLE001
                pass

    def _start_work_loop(self, side: str, max_utterances: Optional[int],
                         num_workers: int = 1) -> list:
        """Start STT workers for this side.

        Multiple STT workers pull from the same queue, so a slow Whisper
        call for one utterance doesn't delay the next one. Captions are
        ordered by uid (monotonic counter), not by processing order.
        """
        q = queue.Queue()
        self._work_qs[side] = q
        self._side_stop[side] = threading.Event()

        def worker() -> None:
            count = 0
            while True:
                try:
                    item = q.get(timeout=0.2)
                except queue.Empty:
                    if self._stop_ev.is_set() or self._side_stop[side].is_set():
                        return
                    continue
                if item is None or self._stop_ev.is_set() or self._side_stop[side].is_set():
                    return
                wav, source, target, speak = item
                try:
                    count = self._process_utterance(
                        wav, source=source, target=target, speak=speak,
                        max_utterances=max_utterances, count=count,
                    )
                except Exception as exc:
                    logger.error("Utterance failed, skipping: %s", exc)
                    self._caption_q.put(
                        ("error", "you" if speak else "p1", target,
                         f"Utterance failed: {exc}"[:160], 0))
                    self._report_service_health("translate", False)
                    self._tts_lang = None
                if max_utterances and count >= max_utterances:
                    self._side_stop[side].set()
                    return

        threads = []
        for i in range(num_workers):
            t = threading.Thread(target=worker, daemon=True,
                                 name=f"stt-worker-{side}-{i}")
            t.start()
            threads.append(t)
        return threads

    # ------------------------------------------------------------------ #

    def run(self, direction: Optional[str] = None, max_utterances: Optional[int] = None) -> None:
        # Explicit --direction => legacy single-mic mode. Otherwise, two-way
        # when loopback (system audio) or a Person 1 recording device is set.
        if direction is not None:
            self._single_run(direction=direction, max_utterances=max_utterances)
        elif self.person1_loopback or self.person1_device is not None:
            self._two_way_run(max_utterances=max_utterances)
        else:
            self._single_run(direction="p1", max_utterances=max_utterances)

    @staticmethod
    def _open_output_stream(device: Optional[int], samplerate: int = None, channels: int = None):
        """Open a sounddevice OutputStream at the device's NATIVE rate/channels.

        We intentionally omit ``samplerate`` so the stream runs at the device's
        default rate; TTS PCM is resampled (and duplicated to the device's
        channel count) in ``_process_utterance``. Returns
        ``(stream, actual_samplerate, actual_channels)``.
        """
        import sounddevice as sd

        info = None
        if device is not None:
            try:
                info = sd.query_devices(device)
            except Exception:  # noqa: BLE001
                info = None
            if info is not None and int(info.get("max_output_channels", 0)) < 1:
                raise RuntimeError(
                    f"--tts-output-device {device} ('{info.get('name')}') has NO "
                    f"output channels. Pick an OUTPUT device (speakers or a virtual "
                    f"microphone like VB-Cable / BlackHole). Run "
                    f"`python stream.py --list-devices` to see which devices have out>0."
                )

        # Open at the device's native channel count. Feeding a stereo-only device
        # (e.g. VB-Cable) mono PCM makes the backend treat it as interleaved
        # stereo -> garbled, double-speed audio. So we open stereo and duplicate.
        dev_max = int(info.get("max_output_channels", 2)) if info else 2
        if channels is None or channels > dev_max:
            channels = dev_max if dev_max >= 1 else 1
        if channels < 1:
            channels = 1

        try:
            kwargs = {"device": device, "channels": channels, "dtype": "int16"}
            if samplerate is not None:
                kwargs["samplerate"] = int(samplerate)
            stream = sd.OutputStream(**kwargs)
            stream.start()
            return stream, int(stream.samplerate), channels
        except Exception as exc:  # noqa: BLE001
            name = info.get("name") if info else "default"
            raise RuntimeError(
                f"Could not open TTS output device {device} ('{name}'): {exc}. "
                f"Use `python stream.py --list-devices` to choose a valid output device."
            ) from exc

    def _open_tts_output(self):
        """Open the TTS output stream at an engine-compatible rate.

        The device default can be a rate the TTS engine's WS endpoint rejects
        (e.g. CABLE Input defaults to 44100 Hz and ElevenLabs answers
        ``output_format_not_allowed`` for pcm_44100). Snap the rate to one the
        engine supports and open the stream explicitly at that rate — Windows
        shared-mode resamples internally, and playback stays click-free with
        zero per-chunk resampling. Falls back to the native-rate stream (+
        per-chunk resampling) if the explicit-rate reopen fails.
        """
        player, sr, ch = self._open_output_stream(self.tts_output_device)
        want = _nearest_eleven_rate(sr)
        if want != sr:
            try:
                player.stop()
                player.close()
            except Exception:  # noqa: BLE001
                pass
            try:
                player, sr, ch = self._open_output_stream(
                    self.tts_output_device, samplerate=want
                )
                logger.info("TTS output opened at engine-safe rate %d Hz.", sr)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Could not reopen TTS output at %d Hz (%s); keeping %d Hz "
                    "with per-chunk resampling.", want, exc, sr,
                )
        return player, sr, ch

    @staticmethod
    def _to_channels(data: bytes, dst_channels: int) -> bytes:
        """Convert mono 16-bit PCM to ``dst_channels`` (duplicate mono to stereo)."""
        if dst_channels <= 1:
            return data
        samples = np.frombuffer(data, dtype=np.int16)
        if samples.size == 0:
            return data
        dup = np.repeat(samples, dst_channels).astype(np.int16)
        return dup.tobytes()

    @staticmethod
    def _write_audio(player, data: bytes) -> None:
        """Write PCM to the TTS output stream, matching its channel layout.

        ``data`` is already adapt_audio'd (resampled + duplicated to the
        device's channel count). sounddevice's ``write`` expects an array of
        shape ``(frames, channels)``; a flat mono array on a multi-channel
        stream raises "number of channels must match".
        """
        arr = np.frombuffer(data, dtype=np.int16)
        if arr.size == 0:
            return
        channels = getattr(player, "channels", 1) if hasattr(player, "channels") else 1
        if channels > 1 and arr.ndim == 1:
            arr = arr.reshape(-1, channels)
        player.write(arr)

    def _adapt_audio(self, audio: bytes) -> bytes:
        """Resample TTS PCM to the device rate and match its channel count."""
        if self._tts_out_samplerate and self._tts_out_samplerate != self.tts.sample_rate:
            audio = self._resample_int16(audio, self.tts.sample_rate, self._tts_out_samplerate)
        return self._to_channels(audio, self._tts_out_channels)

    # ------------------------------------------------------------------ #
    # Local TTS monitor ("hear what the voice says", toggleable).
    # ------------------------------------------------------------------ #
    def set_monitor_tts(self, enabled: bool) -> None:
        """Duplicate TTS audio to your own speakers/headphones (toggle)."""
        enabled = bool(enabled)
        if enabled == self._monitor_tts:
            return
        self._monitor_tts = enabled
        if enabled:
            self._ensure_monitor()
            logger.info("TTS monitor ON (local speakers).")
        else:
            self._close_monitor()
            logger.info("TTS monitor OFF.")

    def _ensure_monitor(self) -> None:
        """Open an output stream on the default output device for monitoring."""
        if self._monitor_stream is not None:
            return
        try:
            player, sr, ch = self._open_output_stream(self._monitor_device)
            self._monitor_stream = player
            self._monitor_samplerate = sr
            self._monitor_channels = ch
            logger.info("Monitor stream open: %s Hz / %d ch", sr, ch)
        except Exception as exc:  # noqa: BLE001
            logger.error("Could not open TTS monitor stream: %s", exc)
            self._monitor_stream = None
            self._monitor_tts = False

    def _close_monitor(self) -> None:
        player = self._monitor_stream
        self._monitor_stream = None
        if player is not None:
            try:
                player.stop()
                player.close()
            except Exception:  # noqa: BLE001
                pass

    def _monitor_adapt(self, data: bytes) -> bytes:
        """Adapt raw TTS PCM to the monitor device's rate/channel layout."""
        rate = getattr(self.tts, "sample_rate", None)
        if rate and self._monitor_samplerate and rate != self._monitor_samplerate:
            data = self._resample_int16(data, rate, self._monitor_samplerate)
        return self._to_channels(data, self._monitor_channels or 1)

    def _write_monitor(self, raw_chunk: bytes) -> None:
        """Best-effort duplicate of a TTS chunk to the local monitor stream."""
        if not (self._monitor_tts and self._monitor_stream is not None):
            return
        try:
            self._write_audio(self._monitor_stream, self._monitor_adapt(raw_chunk))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Monitor write skipped: %s", exc)

    @staticmethod
    def _buffer_for_tts(tokens: Iterator[str]) -> Iterator[str]:
        """Group streamed translation tokens into sentence-sized chunks.

        LLM tokens for inflection-heavy languages (Greek, etc.) are often tiny
        sub-word fragments, and even word-level feeding makes the TTS pronounce
        units in isolation -> robotic, staccato audio. Buffering to sentence
        boundaries (or a generous length cap) lets the TTS render full sentences
        with natural prosody. Trade-off: audio starts after a sentence completes.
        """
        buf = ""
        sentence_ends = ".!?…:;"
        for tok in tokens:
            buf += tok
            stripped = buf.rstrip()
            if stripped and stripped[-1] in sentence_ends:
                yield buf
                buf = ""
            elif len(buf) >= 80 and " " in buf:
                # Very long sentence: emit at the last word boundary anyway.
                head, _, tail = buf.rpartition(" ")
                yield head
                buf = tail
        if buf.strip():
            yield buf

    @staticmethod
    def _resample_int16(data: bytes, src_rate: int, dst_rate: int) -> bytes:
        """Lightweight linear-resample of 16-bit PCM (good enough for TTS speech)."""
        if not data or src_rate == dst_rate:
            return data
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float64)
        if samples.size < 2:
            return data
        n = max(1, int(round(samples.size * dst_rate / src_rate)))
        x_old = np.linspace(0.0, samples.size - 1, samples.size)
        x_new = np.linspace(0.0, samples.size - 1, n)
        resampled = np.interp(x_new, x_old, samples).astype(np.int16)
        return resampled.tobytes()

    def _two_way_run(self, max_utterances: Optional[int] = None) -> None:
        """Person 1 = system/Discord audio (loopback); You = microphone."""
        if self.person1_loopback:
            from loopback_capture import LoopbackCapture

            p1_capture = LoopbackCapture(
                output_device=self.system_device,
                vad_threshold=VAD_THRESHOLD,
                silence_timeout=self._silence_timeout,
            )
            logger.info("Person 1 source: system-audio loopback (what you hear).")
        else:
            p1_capture = MicCapture(
                device=self.person1_device,
                vad_threshold=VAD_THRESHOLD,
                silence_timeout=self._silence_timeout,
            )
            logger.info("Person 1 source: recording device %s (Stereo Mix).", self.person1_device)
        user_capture = MicCapture(
            device=self.mic_device,
            vad_threshold=VAD_THRESHOLD,
            noise_cancel=self._noise_cancel,
            silence_timeout=self._silence_timeout,
        )

        # Open the output stream FIRST (at an engine-safe rate) so the TTS
        # engine emits PCM in exactly that rate — no resampling clicks while
        # playing chunks as they generate.
        player, sr, ch = self._open_tts_output()
        self._tts_out_samplerate = sr
        self._tts_out_channels = ch
        self._ensure_tts(self.person1_language, samplerate=sr)

        print(
            f"\n🎙️  Two-way live mode:\n"
            f"  • Person 1 (system/Discord audio) → English captions\n"
            f"  • You (mic) → {self.person1_language} TTS (to Discord)\n"
            f"Ctrl-C or Stop to end.\n"
        )

        threads = [
            threading.Thread(
                target=self._capture_loop,
                kwargs=dict(
                    capture=p1_capture,
                    source=self.person1_language,
                    target=self.user_language,
                    speak=False,
                    max_utterances=max_utterances,
                    role="p1",
                ),
                daemon=True,
            ),
            threading.Thread(
                target=self._capture_loop,
                kwargs=dict(
                    capture=user_capture,
                    source=self.user_language,
                    target=self.person1_language,
                    speak=True,
                    max_utterances=max_utterances,
                    role="user",
                ),
                daemon=True,
            ),
            threading.Thread(
                target=self._tts_worker_loop, args=(player,), daemon=True
            ),
        ]
        self._tts_worker = threads[-1]
        for t in threads:
            t.start()
        # Per-side utterance workers (2 STT workers per side = 4 total).
        work_threads = [
            *self._start_work_loop("p1", max_utterances),
            *self._start_work_loop("user", max_utterances),
        ]
        translate_threads = self._start_translate_loops()
        try:
            for t in threads:
                t.join()
        except KeyboardInterrupt:
            print("\n⏹  Stopped.")
        finally:
            self._work_qs.get("p1", queue.Queue()).put(None)
            self._work_qs.get("user", queue.Queue()).put(None)
            for tq in self._translate_qs.values():
                tq.put(None)
            self._tts_job_q.put(None)  # wake the TTS worker
            self.stop()
            for wt in work_threads:
                wt.join(timeout=2)
            for tt in translate_threads:
                tt.join(timeout=2)
            if self._tts_worker is not None:
                self._tts_worker.join(timeout=2)
            if player is not None:
                try:
                    player.stop()
                    player.close()
                except Exception:  # noqa: BLE001
                    pass

    def _single_run(self, direction: str = "p1", max_utterances: Optional[int] = None) -> None:
        """Legacy single-mic mode (one source; --direction picks who is at it)."""
        if direction not in ("p1", "user"):
            raise ValueError("direction must be 'p1' or 'user'.")
        speak = direction == "user"
        player = None
        if speak:
            # Player first (engine-safe rate): build the TTS engine for it.
            player, sr, ch = self._open_tts_output()
            self._tts_out_samplerate = sr
            self._tts_out_channels = ch
            self._ensure_tts(self.person1_language, samplerate=sr)

        print(
            f"\n🎙️  Live mode: "
            f"{'Person 1 → You (text)' if not speak else 'You → Person 1 (voice)'} "
            f"(target={self.person1_language}). Speak now. Ctrl-C or Stop to end.\n"
        )

        capture = MicCapture(
            device=self.mic_device,
            vad_threshold=VAD_THRESHOLD,
            noise_cancel=self._noise_cancel,
            silence_timeout=self._silence_timeout,
        )
        side = "user" if speak else "p1"
        work_threads = self._start_work_loop(side, max_utterances)
        translate_threads = self._start_translate_loops()
        worker = None
        if speak:
            worker = threading.Thread(
                target=self._tts_worker_loop, args=(player,), daemon=True
            )
            self._tts_worker = worker
            worker.start()
        try:
            self._capture_loop(
                capture,
                source=self.person1_language if direction == "p1" else self.user_language,
                target=self.user_language if direction == "p1" else self.person1_language,
                speak=speak,
                max_utterances=max_utterances,
            )
        finally:
            self._work_qs.get(side, queue.Queue()).put(None)
            for tq in self._translate_qs.values():
                tq.put(None)
            self._tts_job_q.put(None)  # wake the TTS worker
            for wt in work_threads:
                wt.join(timeout=2)
            for tt in translate_threads:
                tt.join(timeout=2)
            if worker is not None:
                worker.join(timeout=2)
            if player is not None:
                try:
                    player.stop()
                    player.close()
                except Exception:  # noqa: BLE001
                    pass
