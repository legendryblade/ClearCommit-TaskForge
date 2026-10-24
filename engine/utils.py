"""Small resilience + media utilities used by the pipeline."""

from __future__ import annotations

import random
import threading
import time
from typing import Callable, TypeVar

import requests

from logger import get_logger

logger = get_logger("utils")

T = TypeVar("T")


class KeyRotator:
    """Round-robin key selector (thread-safe) to spread API rate limits.

    Callers pass the list of API keys (e.g. two Groq keys); each ``next()``
    returns the following key so parallel requests land on different keys.
    """

    def __init__(self, keys) -> None:
        self._keys = [k for k in (keys or []) if k]
        self._idx = 0
        self._lock = threading.Lock()

    @property
    def keys(self) -> list:
        return list(self._keys)

    def next(self):
        if not self._keys:
            return None
        with self._lock:
            key = self._keys[self._idx % len(self._keys)]
            self._idx += 1
            return key

# Errors that are worth retrying (transient network / rate-limit issues).
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
_RETRYABLE_EXCEPTIONS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)


def with_retry(
    func: Callable[[], T],
    *,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 15.0,
    label: str = "request",
) -> T:
    """Call ``func`` with exponential backoff + jitter.

    Retries on transient HTTP status codes and network exceptions. Raises the
    last exception if all attempts fail.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except _RETRYABLE_EXCEPTIONS as exc:  # network level
            last_exc = exc
        except requests.exceptions.RequestException as exc:  # HTTP level
            status = getattr(exc.response, "status_code", None)
            if status in _RETRYABLE_STATUS:
                last_exc = exc
            else:
                raise

        if attempt == max_attempts:
            break

        delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
        delay += random.uniform(0, 0.5)  # jitter to avoid thundering herd
        logger.warning(
            "%s failed (attempt %d/%d); retrying in %.1fs: %s",
            label, attempt, max_attempts, delay, last_exc,
        )
        time.sleep(delay)

    logger.error("%s permanently failed after %d attempts", label, max_attempts)
    assert last_exc is not None
    raise last_exc


def audio_duration_seconds(path: str) -> float | None:
    """Best-effort audio duration in seconds using only stdlib fallbacks.

    Tries ``ffprobe`` (if installed) then falls back to a lightweight header
    probe for WAV. Returns ``None`` when the duration cannot be determined.
    """
    import shutil
    import struct
    import wave

    # Preferred: ffprobe is reliable for any container.
    if shutil.which("ffprobe"):
        try:
            import subprocess

            out = subprocess.check_output(
                [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    path,
                ],
                stderr=subprocess.DEVNULL,
            )
            return float(out.strip())
        except Exception:  # noqa: BLE001 - best effort
            pass

    # Fallback: raw WAV header math (no extra deps).
    try:
        with wave.open(path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate:
                return frames / float(rate)
    except Exception:  # noqa: BLE001 - not a wav / unreadable
        pass

    # Fallback: minimal RIFF/WAVE header parse (handles non-seekable quirk).
    try:
        with open(path, "rb") as fh:
            header = fh.read(44)
        if header[:4] == b"RIFF" and header[8:12] == b"WAVE":
            sample_rate = struct.unpack_from("<I", header, 24)[0]
            byte_rate = struct.unpack_from("<I", header, 28)[0]
            if byte_rate:
                import os

                size = os.path.getsize(path)
                return max(0.0, (size - 44) / float(byte_rate))
            if sample_rate:
                bits = struct.unpack_from("<H", header, 34)[0]
                channels = struct.unpack_from("<H", header, 22)[0]
                import os

                size = os.path.getsize(path)
                return max(0.0, (size - 44) / (sample_rate * channels * bits / 8))
    except Exception:  # noqa: BLE001
        pass

    return None
