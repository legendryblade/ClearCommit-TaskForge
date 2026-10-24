"""Low-latency microphone capture with lightweight VAD.

Groq's Whisper API is file-based (no streaming/WebSocket STT). To keep the
pipeline's latency low we capture audio in small 20 ms PCM chunks and detect
utterance boundaries locally with a simple RMS voice-activity detector, then
flush each completed utterance as a WAV for fast transcription. This is the
practical stand-in for a "streaming STT" WebSocket.
"""

from __future__ import annotations

import io
import queue
import wave
from typing import Iterator, Optional

import numpy as np


class MicCapture:
    def __init__(
        self,
        rate: int = 16000,
        chunk_ms: int = 20,
        vad_threshold: float = 0.012,
        silence_timeout: float = 0.7,
        max_utterance_s: float = 15.0,
        device: Optional[int] = None,
        noise_cancel: bool = False,
    ):
        self.rate = rate
        self.chunk_ms = chunk_ms
        self.chunk_samples = int(rate * chunk_ms / 1000)
        self.vad_threshold = vad_threshold
        self.silence_timeout = silence_timeout
        self.max_utterance_samples = int(rate * max_utterance_s)
        self.device = device

        # Krisp-style spectral-gating noise suppression. A stationary noise
        # profile (fan hum, AC, PC fans) is learned during idle/silence, then
        # each utterance's spectrum is masked against it so only speech
        # survives — Whisper stops hallucinating on the mic's background noise.
        self.noise_cancel = noise_cancel
        self._noise_learn_s = 2.0  # seconds of idle audio to build the profile
        self._n_fft = 512  # 32 ms frames @ 16 kHz
        self._noise_mag: Optional[np.ndarray] = None
        self._noise_frames: list[np.ndarray] = []

        self._utt_q: "queue.Queue[bytes]" = queue.Queue()
        self._stop = False
        self._buf: list[np.ndarray] = []
        self._speaking = False
        self._silence_samples = 0
        self._stream = None
        self._dev_rate = None  # actual capture rate (set in start)
        self.last_rms = 0.0  # live input level for the GUI meter
        self.cb_count = 0    # increments per audio callback (watchdog liveness)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        if src_rate == dst_rate or samples.size < 2:
            return samples
        n = max(1, int(round(samples.size * dst_rate / src_rate)))
        x_old = np.linspace(0.0, samples.size - 1, samples.size)
        x_new = np.linspace(0.0, samples.size - 1, n)
        return np.interp(x_new, x_old, samples.astype(np.float64)).astype(np.int16)

    def _callback(self, indata, frames, time_info, status):
        self.cb_count += 1
        samples = indata[:, 0] if indata.ndim > 1 else indata
        samples = samples.astype(np.int16)
        # Resample device-native audio down to the 16 kHz Whisper expects.
        if self._dev_rate and self._dev_rate != self.rate:
            samples = self._resample(samples, self._dev_rate, self.rate)
        rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)) / 32768.0)
        # GUI VU meter polls this (atomic float write; audio-thread safe).
        self.last_rms = rms
        if self.noise_cancel and not self._speaking and rms <= self.vad_threshold:
            self._learn_noise(samples)
        self._process(samples, rms)

    # ------------------------------------------------------------------ #
    # Krisp-style spectral gating.
    # ------------------------------------------------------------------ #
    def _learn_noise(self, samples: np.ndarray) -> None:
        """Accumulate idle audio into a stationary noise magnitude estimate."""
        if self._noise_mag is not None:
            return  # profile already built; the fan is steady
        if samples.dtype != np.int16:
            samples = samples.astype(np.int16)
        self._noise_frames.append(samples.astype(np.float32) / 32768.0)
        target = int(self._noise_learn_s * self.rate / self.chunk_samples)
        if len(self._noise_frames) < target:
            return
        window = np.hanning(self._n_fft)
        specs = []
        for frame in self._noise_frames:
            pad = np.zeros(self._n_fft, dtype=np.float32)
            pad[: min(len(frame), self._n_fft)] = frame[: self._n_fft]
            specs.append(np.abs(np.fft.rfft(pad * window)))
        self._noise_mag = np.mean(specs, axis=0).astype(np.float32)
        self._noise_frames = []

    def _spectral_gate(self, samples: np.ndarray) -> np.ndarray:
        """Mask steady-state noise out of an utterance via overlap-add FFT."""
        if self._noise_mag is None:
            return samples
        if samples.dtype == np.int16:
            x = samples.astype(np.float32) / 32768.0
        else:
            x = samples.astype(np.float32)
        n_fft, hop = self._n_fft, self._n_fft // 2
        window = np.hanning(n_fft).astype(np.float32)
        # Hann touches zero at its ends -> the overlap-add norm is 0 there and
        # the division blows up on the first/last samples. Lift the edges.
        window[0] = window[-1] = 1e-3
        # Rumble cut below ~60 Hz (fan hum is low-frequency) regardless of mask.
        low_bins = int(60.0 * n_fft / self.rate) + 1
        eps = 1e-12

        n_frames = max(1, 1 + (len(x) - n_fft) // hop)
        y = np.zeros(len(x) + n_fft, dtype=np.float32)
        norm = np.zeros(len(x) + n_fft, dtype=np.float32)
        for i in range(n_frames):
            start = i * hop
            frame = x[start : start + n_fft]
            if len(frame) < n_fft:
                frame = np.pad(frame, (0, n_fft - len(frame)))
            spec = np.fft.rfft(frame * window)
            mag = np.abs(spec)
            # Spectral subtraction gate: keep speech, attenuate noise bins.
            # Gentle strength — 0.7x over-subtraction and a 0.3 floor only
            # shave the steady fan/AC hum; soft/distant speech passes through
            # (aggressive gating made Whisper hallucinate and ate quiet voices).
            mask = np.clip(1.0 - (self._noise_mag * 0.7) / (mag + eps), 0.0, 1.0)
            mask = np.maximum(mask, 0.3)  # floor so speech is never fully cut
            mask[:low_bins] = 0.2
            spec = spec * mask
            out = np.fft.irfft(spec, n=n_fft) * window
            y[start : start + n_fft] += out
            norm[start : start + n_fft] += window * window
        norm = np.maximum(norm, eps)
        y = np.divide(y, norm)
        # The first/last samples have near-zero overlap-add norm (window
        # edges), where dividing blows up numerically. Pass the raw input
        # through there — those regions are a few microseconds of audio.
        edge = norm < 1e-4
        if edge.any():
            pad = np.zeros(len(x) + n_fft, dtype=np.float32)
            pad[: len(x)] = x
            y = np.where(edge, pad, y)
        y = np.clip(y, -1.0, 1.0)
        return (y[: len(x)] * 32767.0).astype(np.int16)

    def _process(self, samples: np.ndarray, rms: float) -> None:
        if rms > self.vad_threshold:
            if not self._speaking:
                self._speaking = True
                self._silence_samples = 0
            self._buf.append(samples)
            if sum(len(s) for s in self._buf) >= self.max_utterance_samples:
                self._flush()
        else:
            if self._speaking:
                self._silence_samples += len(samples)
                self._buf.append(samples)  # keep a little tail so words aren't cut
                if self._silence_samples >= self.silence_timeout * self.rate:
                    self._flush()

    def _flush(self) -> None:
        if self._buf:
            audio = np.concatenate(self._buf)
            if self.noise_cancel:
                audio = self._spectral_gate(audio)
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
    def start(self) -> "MicCapture":
        import sounddevice as sd  # lazy import; only needed at runtime

        # Open at the device's native/default rate (many Windows devices reject
        # 16000 Hz) and resample to 16 kHz in the callback for Whisper.
        dev_rate = self.rate
        if self.device is not None:
            try:
                dev_rate = int(sd.query_devices(self.device).get("default_samplerate", self.rate))
            except Exception:  # noqa: BLE001
                dev_rate = self.rate
        self._dev_rate = dev_rate

        self._stream = sd.InputStream(
            samplerate=dev_rate,
            channels=1,
            dtype="int16",
            blocksize=int(dev_rate * self.chunk_ms / 1000),
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()
        return self

    def stop(self) -> None:
        self._stop = True
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None

    def utterances(self) -> Iterator[bytes]:
        """Yield completed WAV utterances until ``stop()`` is called."""
        while not self._stop:
            try:
                yield self._utt_q.get(timeout=0.2)
            except queue.Empty:
                continue
        # drain remaining
        while not self._utt_q.empty():
            yield self._utt_q.get_nowait()

    def __enter__(self) -> "MicCapture":
        return self.start()

    def __exit__(self, *_):
        self.stop()


def list_microphones() -> list[tuple[int, str]]:
    import sounddevice as sd

    return [(i, d["name"]) for i, d in enumerate(sd.query_devices()) if d.get("max_input_channels", 0) > 0]


def list_audio_devices() -> list[tuple[int, str, int, int]]:
    """All devices with (index, name, max_input_channels, max_output_channels)."""
    import sounddevice as sd

    out = []
    for i, d in enumerate(sd.query_devices()):
        out.append(
            (
                i,
                d["name"],
                int(d.get("max_input_channels", 0)),
                int(d.get("max_output_channels", 0)),
            )
        )
    return out


def resolve_device(spec: Optional[int], kind: str = "any") -> Optional[int]:
    """Resolve a device argument to a stable index.

    ``spec`` may be an integer (used directly) or a substring matched
    case-insensitively against device names. ``kind`` is ``"input"``,
    ``"output"``, or ``"any"`` and filters by channel availability so, e.g., an
    output device can't be matched when a microphone is requested. This avoids
    the fragility of numeric indices, which change whenever the OS
    re-enumerates audio devices.
    """
    if spec is None:
        return None
    try:
        return int(spec)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        pass
    needle = str(spec).lower()
    for idx, name, max_in, max_out in list_audio_devices():
        if kind == "input" and max_in < 1:
            continue
        if kind == "output" and max_out < 1:
            continue
        if needle in name.lower():
            return idx
    raise RuntimeError(
        f"No audio device matching '{spec}' (kind={kind}). "
        f"Run `python stream.py --list-devices` to see available names."
    )
