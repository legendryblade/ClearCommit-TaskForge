"""System-audio (WASAPI loopback) capture via pyaudiowpatch.

This captures *everything you hear* on an output device (e.g. your headset
during a Discord call) — the robust, device-agnostic replacement for the
fragile "Stereo Mix" recording endpoint, which only mirrors one specific
output and silently captures nothing when audio plays through a USB headset.

Requires:  pip install pyaudiowpatch
"""

from __future__ import annotations

import io
import queue
import wave
from typing import Iterator, Optional

import numpy as np

logger = None  # set lazily to avoid importing logger at import time if unused


def _get_logger():
    global logger
    if logger is None:
        from logger import get_logger

        logger = get_logger("loopback_capture")
    return logger


try:  # pragma: no cover - import guard (installed at runtime on Windows)
    import pyaudiowpatch as pyaudio
except Exception:  # noqa: BLE001
    pyaudio = None  # type: ignore


def _resample_int16(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or samples.size < 2:
        return samples
    n = max(1, int(round(samples.size * dst_rate / src_rate)))
    x_old = np.linspace(0.0, samples.size - 1, samples.size)
    x_new = np.linspace(0.0, samples.size - 1, n)
    return np.interp(x_new, x_old, samples.astype(np.float64)).astype(np.int16)


class LoopbackCapture:
    """Capture system audio (output-device loopback) with local VAD.

    Mirrors the ``MicCapture`` interface (``start`` / ``utterances`` / ``stop``)
    and yields 16 kHz mono WAV bytes ready for Whisper.
    """

    def __init__(
        self,
        output_device: Optional[int] = None,
        rate: int = 16000,
        chunk_ms: int = 20,
        vad_threshold: float = 0.02,
        silence_timeout: float = 0.7,
        max_utterance_s: float = 15.0,
    ):
        if pyaudio is None:
            raise RuntimeError(
                "pyaudiowpatch is required for system-audio capture. "
                "Install it with `pip install pyaudiowpatch`."
            )
        self.output_device = output_device  # OUTPUT device to mirror; None = default
        self.rate = rate  # target rate for Whisper (16 kHz)
        self.chunk_ms = chunk_ms
        self.vad_threshold = vad_threshold
        self.silence_timeout = silence_timeout
        self.max_utterance_samples = int(rate * max_utterance_s)

        self._utt_q: "queue.Queue[bytes]" = queue.Queue()
        self._stop = False
        self._buf: list[np.ndarray] = []
        self._speaking = False
        self._silence_samples = 0
        self._dev_rate = rate
        self._dev_channels = 2
        self._stream = None
        self._pa = None
        self.last_rms = 0.0  # live input level for the GUI meter
        self.cb_count = 0    # increments per audio callback (watchdog liveness)
        self.device = None  # set in start() to the loopback device index

    # ------------------------------------------------------------------ #
    def _find_loopback_device(self):
        p = pyaudio.PyAudio()
        try:
            wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        except Exception as exc:  # noqa: BLE001
            p.terminate()
            raise RuntimeError(
                "WASAPI is not available (required for loopback capture)."
            ) from exc
        default_out = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
        target_name = default_out["name"]
        if self.output_device is not None:
            target_name = p.get_device_info_by_index(self.output_device)["name"]
        for i in range(p.get_device_count()):
            d = p.get_device_info_by_index(i)
            if d.get("isLoopbackDevice") and str(d["name"]).startswith(target_name):
                return d, p
        p.terminate()
        raise RuntimeError(
            f"No WASAPI loopback device found for output '{target_name}'. "
            f"Make sure the output device is enabled and (for a headset) that "
            f"audio is actually playing through it."
        )

    # ------------------------------------------------------------------ #
    def _callback(self, in_data, frame_count, time_info, status):
        self.cb_count += 1
        if in_data is None:
            return (None, pyaudio.paContinue)
        samples = np.frombuffer(in_data, dtype=np.int16)
        if self._dev_channels == 2 and samples.size % 2 == 0:
            mono = samples.reshape(-1, 2).mean(axis=1).astype(np.int16)
        else:
            mono = samples
        mono16 = _resample_int16(mono, self._dev_rate, self.rate)
        rms = float(np.sqrt(np.mean(mono16.astype(np.float32) ** 2)) / 32768.0)
        # GUI VU meter polls this (atomic float write; audio-thread safe).
        self.last_rms = rms
        self._process(mono16, rms)
        return (None, pyaudio.paContinue)

    def _process(self, samples: np.ndarray, rms: float) -> None:
        if rms > self.vad_threshold:
            if not self._speaking:
                self._speaking = True
                self._silence_samples = 0
            self._buf.append(samples)
            if sum(len(s) for s in self._buf) >= self.max_utterance_samples:
                self._flush()
        elif self._speaking:
            self._silence_samples += len(samples)
            self._buf.append(samples)
            if self._silence_samples >= self.silence_timeout * self.rate:
                self._flush()

    def _flush(self) -> None:
        if self._buf:
            audio = np.concatenate(self._buf)
            self._utt_q.put(self._to_wav(audio))
        self._buf = []
        self._speaking = False
        self._silence_samples = 0

    @staticmethod
    def _to_wav(audio_int16: np.ndarray) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(audio_int16.tobytes())
        return buf.getvalue()

    # ------------------------------------------------------------------ #
    def start(self) -> "LoopbackCapture":
        dev, pa = self._find_loopback_device()
        self._pa = pa
        self._dev_rate = int(dev.get("defaultSampleRate", self.rate))
        self._dev_channels = int(dev.get("maxInputChannels", 2)) or 2
        frames_per_buffer = int(self._dev_rate * self.chunk_ms / 1000)
        self._stream = pa.open(
            format=pyaudio.paInt16,
            channels=self._dev_channels,
            rate=self._dev_rate,
            input=True,
            input_device_index=dev["index"],
            frames_per_buffer=max(1, frames_per_buffer),
            stream_callback=self._callback,
        )
        self._stream.start_stream()
        self.device = dev["index"]
        _get_logger().info(
            "Loopback capture started on '%s' (%d Hz, %d ch)",
            dev["name"], self._dev_rate, self._dev_channels,
        )
        return self

    def stop(self) -> None:
        self._stop = True
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            finally:
                self._stream = None
        if self._pa is not None:
            try:
                self._pa.terminate()
            finally:
                self._pa = None

    def utterances(self) -> Iterator[bytes]:
        while not self._stop:
            try:
                yield self._utt_q.get(timeout=0.2)
            except queue.Empty:
                continue
        while not self._utt_q.empty():
            yield self._utt_q.get_nowait()
