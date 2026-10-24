"""Isolated asynchronous AI service foundation for ClearCommit (Stage 3A).

Pure stdlib. No network, no API keys, no GUI, no audio. Provider-agnostic so a
real backend (Groq / Nemotron / other) can be plugged in later without touching
the translation pipeline or the UI.

Design safety properties:
    * AI analysis never blocks transcription, translation, caption rendering,
      or TTS. Submission is non-blocking and never spawns threads per request.
    * Active and queued work are both bounded. Queue-full submission returns an
      immediate rejection result without blocking the caller.
    * Failures (provider error, timeout, cancellation, queue-full) cannot stall
      or crash the translation pipeline.
    * Result delivery is thread-safe; callback exceptions are caught. Tk
      widgets are never touched from the worker thread — future GUI integration
      must marshal callbacks through the existing UI queue / Tk polling.
    * Only finalized transcript text + immutable metadata is accepted; never
      raw audio.
    * No autonomous confirmation, no consent inference, no native actions.

Request state machine (deterministic):

    PENDING  -> RUNNING  -> COMPLETED
                |          ^
                v          |
              CANCELLED    |
                |          |
                v          |
              FAILED ------+
                |
                v
              TIMED_OUT

    PENDING -> CANCELLED (cancelled before work started)
    RUNNING  -> TIMED_OUT  (per-call timeout elapsed; late result discarded)
    RUNNING  -> CANCELLED  (cancel requested while running; late result discarded)
    Any -> FAILED (provider raised)

    COMPLETED, FAILED, CANCELLED, TIMED_OUT are terminal.

Cancellation semantics:
    * A request cancelled while PENDING becomes CANCELLED immediately and is
      never submitted to the worker.
    * A request cancelled while RUNNING is flagged; when the provider finally
      returns, the late result is discarded and the request is reported as
      CANCELLED — never COMPLETED.
    * Cancellation is cooperative; a running provider thread cannot be
      forcibly stopped in Python.

Timeout semantics:
    * A timeout is NOT reported as cancellation. It is reported as TIMED_OUT.
    * The late provider result is discarded and never delivered.
    * Duplicate completion callbacks are prevented: once terminal, later
      results/cancellations are no-ops.

Stuck-provider recovery + circuit breaker:
    * Python threads cannot be killed, so a non-cooperative provider that never
      returns will occupy the single worker thread until it does.
    * Each timeout retires the current executor and creates a fresh one, so a
      stuck provider cannot consume the worker forever.
    * Repeated timeouts trip a circuit breaker: after ``max_timeouts``
      timeouts, the service stops accepting new provider work and reports
      subsequent submissions as FAILED/service_unavailable. The circuit breaker
      is not auto-reset (manual intervention required), which avoids an
      unbounded accumulation of stuck daemon threads.
    * Callers should set conservative timeouts; providers should be
      cooperative and return promptly.

Shutdown:
    * ``close()`` has a bounded wait. New submissions are rejected immediately.
    * Queued (PENDING) requests are dropped and reported CANCELLED.
    * Running requests are abandoned: their results are discarded, the worker
      thread is daemon (won't block exit), and the executor is shut down with
      a bounded wait.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from threading import RLock, Thread, Event
from typing import Callable, Optional, Protocol
import time
import uuid

try:  # stdlib on all supported Python versions
    from concurrent.futures import (
        ThreadPoolExecutor,
        Future,
        TimeoutError as _FuturesTimeoutError,
    )
except Exception:  # pragma: no cover - stdlib should always be present
    ThreadPoolExecutor = None  # type: ignore[assignment]
    Future = object  # type: ignore[assignment, misc]

    class _FuturesTimeoutError(Exception):  # type: ignore[no-redef]
        pass


# --------------------------------------------------------------------------- #
# Statuses
# --------------------------------------------------------------------------- #


class AIRequestStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


TERMINAL_STATUSES = frozenset(
    {
        AIRequestStatus.COMPLETED,
        AIRequestStatus.FAILED,
        AIRequestStatus.CANCELLED,
        AIRequestStatus.TIMED_OUT,
    }
)


# --------------------------------------------------------------------------- #
# Typed request / result
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AIRequest:
    """A single AI analysis request. Finalized transcript text + metadata only.

    ``text`` is the finalized utterance. ``side`` is "p1" or "you". ``uid`` is
    the transcript event id used to reference the original evidence. ``mode``
    is the communication mode ("social" / "professional"). ``timestamp`` is
    the epoch-second accept time of the utterance. No raw audio is ever stored
    or transmitted here.
    """

    id: str
    text: str
    side: str
    uid: int
    mode: str
    timestamp: float


@dataclass(frozen=True)
class AIResult:
    """Outcome of one AI request."""

    request_id: str
    status: AIRequestStatus
    output: Optional[str] = None
    error: Optional[str] = None
    elapsed_ms: Optional[float] = None
    provider: Optional[str] = None


# --------------------------------------------------------------------------- #
# Provider interface
# --------------------------------------------------------------------------- #


class AIProvider(Protocol):
    """Pluggable AI backend. Implement ``analyze`` to return coaching text.

    Providers must be cooperative: return promptly, respect timeouts, and never
    block forever. The service enforces per-call timeouts but cannot forcibly
    stop a running provider thread.
    """

    def analyze(self, request: AIRequest) -> str:
        """Return a coaching suggestion string for ``request``.

        Raise on failure; the service records it as FAILED.
        """
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Deterministic fake provider (tests only — no network, no API keys)
# --------------------------------------------------------------------------- #


@dataclass
class FakeAIProvider:
    """Deterministic stand-in provider for tests.

    Configurable delay, failure, output, and permanent hang (simulates a stuck
    provider that never returns). ``hang_release`` is a shared class-level
    event: setting it unblocks every hanging fake so test processes can exit
    naturally (Python threads cannot be forcibly stopped).
    """

    output: str = "suggested coaching text"
    delay_ms: float = 0.0
    fail: bool = False
    error_message: str = "provider failure"
    hang: bool = False
    call_count: int = 0

    def analyze(self, request: AIRequest) -> str:
        self.call_count += 1
        if self.hang:
            # Simulate a provider that never returns (until the shared
            # module-level release event is set by the test runner — Python
            # threads cannot be forcibly stopped).
            _FAKE_HANG_RELEASE.wait()
            raise RuntimeError("unreachable")
        if self.fail:
            raise RuntimeError(self.error_message)
        if self.delay_ms > 0:
            time.sleep(self.delay_ms / 1000.0)
        return self.output


#: Shared release event for all hanging fakes (module-level singleton).
_FAKE_HANG_RELEASE = Event()


# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #


class AIService:
    """Bounded, non-blocking asynchronous AI analysis service.

    Thread-safe. Results are delivered through caller-supplied callbacks. The
    service never touches Tkinter or any UI toolkit.
    """

    def __init__(
        self,
        *,
        max_concurrency: int = 1,
        max_queue: int = 8,
        default_timeout_s: float = 5.0,
        max_timeouts: int = 3,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        if max_queue < 0:
            raise ValueError("max_queue must be >= 0")
        if default_timeout_s <= 0:
            raise ValueError("default_timeout_s must be > 0")
        if max_timeouts < 0:
            raise ValueError("max_timeouts must be >= 0")

        self._max_concurrency = max_concurrency
        self._max_queue = max_queue
        self._default_timeout_s = float(default_timeout_s)
        self._max_timeouts = max_timeouts

        self._lock = RLock()
        self._enabled = False
        self._provider: Optional[AIProvider] = None
        self._closed = False
        # Circuit breaker: number of timeouts observed so far. When it reaches
        # _max_timeouts, new submissions are rejected until the service is
        # recreated. This bounds stuck-worker accumulation from repeated timeouts.
        self._timeout_count = 0

        # Request state.
        self._requests: dict[str, AIRequest] = {}
        self._results: dict[str, AIResult] = {}
        self._timeouts: dict[str, float] = {}
        self._cancel_requested: set[str] = set()
        self._callbacks: dict[str, list[Callable[[AIResult], None]]] = {}
        self._terminal: set[str] = set()

        # Worker + executor + wake signal.
        self._worker: Optional[Thread] = None
        self._worker_stop = Event()
        self._wake = Event()
        self._queue: list[str] = []
        self._executor: Optional[ThreadPoolExecutor] = None
        self._executor_lock = RLock()

        self._start_worker()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def _start_worker(self) -> None:
        self._worker_stop.clear()
        self._wake.clear()
        self._worker = Thread(
            target=self._worker_loop, daemon=True, name="ai-service-worker"
        )
        self._worker.start()

    def _ensure_executor(self) -> Optional[ThreadPoolExecutor]:
        with self._executor_lock:
            if self._executor is None and ThreadPoolExecutor is not None:
                self._executor = ThreadPoolExecutor(max_workers=self._max_concurrency)
            return self._executor

    def _retire_executor(self) -> None:
        """Abandon the current executor (stuck-provider recovery) and start fresh.

        The old executor's daemon thread may linger but won't block process
        exit. A fresh executor is created on the next task submission so new
        work is not blocked by a stuck provider.
        """
        with self._executor_lock:
            old = self._executor
            self._executor = None
            if old is not None:
                try:
                    old.shutdown(wait=False, cancel_futures=True)
                except Exception:  # noqa: BLE001 - best effort
                    pass

    def set_provider(self, provider: Optional[AIProvider]) -> None:
        """Attach or detach a provider. No-op when closed."""
        with self._lock:
            if self._closed:
                return
            self._provider = provider

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable AI analysis. Defaults to disabled."""
        with self._lock:
            if self._closed:
                return
            self._enabled = bool(enabled)

    def is_enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def is_circuit_open(self) -> bool:
        """True when repeated timeouts have tripped the circuit breaker."""
        with self._lock:
            return self._timeout_count >= self._max_timeouts

    def close(self) -> None:
        """Shut down with a bounded wait.

        New submissions are rejected immediately. PENDING requests are reported
        CANCELLED. The running task (if any) is abandoned — its result is
        discarded and will not be delivered. The worker thread is daemon, so it
        cannot block process exit. The executor shutdown uses a bounded wait.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._enabled = False
            self._worker_stop.set()
            self._wake.set()

        # Cancel every non-terminal request: queued (PENDING) *and* running.
        # Running requests have already been popped from the queue, so clearing
        # the queue alone would leave them stuck in RUNNING. Marking them
        # CANCELLED here is safe because _make_terminal_locked is idempotent —
        # any late result from a provider that eventually returns will be a
        # no-op once the request is already terminal.
        with self._lock:
            for rid in list(self._requests.keys()):
                if rid not in self._terminal:
                    self._make_terminal_locked(
                        rid, AIRequestStatus.CANCELLED, error="service_shutdown"
                    )

        self._retire_executor()
        if self._worker is not None:
            self._worker.join(timeout=2.0)

    # ------------------------------------------------------------------ #
    # Submission (non-blocking, never spawns threads per request)
    # ------------------------------------------------------------------ #
    def submit(
        self,
        request: AIRequest,
        *,
        timeout_s: Optional[float] = None,
    ) -> AIResult:
        """Submit a request and return an immediate result handle.

        Non-blocking: never waits on the provider and never spawns a thread per
        request. Returns an :class:`AIResult`:

        * status=PENDING when accepted for asynchronous processing.
        * status=FAILED with an explanatory error when immediately rejected
          (service disabled, closed, queue full, or circuit open).

        Results for accepted requests are delivered later via
        :meth:`on_complete` callbacks.

        Duplicate request ids are rejected as FAILED so a caller cannot create
        two live states for the same id.
        """
        timeout_s = self._default_timeout_s if timeout_s is None else float(timeout_s)

        with self._lock:
            if self._closed:
                return self._rejection(request.id, "service_closed")
            if not self._enabled:
                return self._rejection(request.id, "ai_service_disabled")
            if self.is_circuit_open():
                return self._rejection(
                    request.id, "service_unavailable_circuit_open",
                )
            if len(self._queue) >= self._max_queue:
                return self._rejection(
                    request.id, "request_dropped_queue_full",
                )
            # Reject duplicate live ids (prevents two terminals for one id).
            if request.id in self._requests and request.id not in self._terminal:
                return self._rejection(request.id, "duplicate_request_id")
            # Accept: store as PENDING and enqueue.
            self._requests[request.id] = request
            self._timeouts[request.id] = timeout_s
            self._results[request.id] = AIResult(
                request_id=request.id, status=AIRequestStatus.PENDING
            )
            self._queue.append(request.id)

        self._wake.set()
        return self._results[request.id]

    @staticmethod
    def _rejection(request_id: str, error: str) -> AIResult:
        """Build an immediate FAILED result for a rejected submission.

        Rejections mutate NO service state: the id was never accepted, so it
        must not enter ``_requests``/``_results``/``_terminal`` (storing it
        would leak memory for never-accepted ids and, for the duplicate-id
        path, would clobber the live original request).
        """
        return AIResult(
            request_id=request_id,
            status=AIRequestStatus.FAILED,
            error=error,
        )

    def on_complete(self, request_id: str,
                    callback: Callable[[AIResult], None]) -> None:
        """Register a result callback.

        If the request is already terminal, the callback is invoked immediately
        (on the caller's thread, wrapped in exception handling). Otherwise it is
        invoked from the completion thread when the request finishes.

        Callbacks must be fast and must not touch Tk widgets; GUI integration
        should marshal through the existing UI queue / Tk polling.

        Note: callers should register callbacks only for ACCEPTED requests
        (submit returned PENDING). Callbacks registered for ids that were
        rejected at submit time are stored but can never be delivered — the
        integration layer gates registration on the submit handle's status.
        """
        with self._lock:
            already_terminal = (
                request_id in self._terminal
                and request_id in self._results
            )
            self._callbacks.setdefault(request_id, []).append(callback)

        if already_terminal:
            self._deliver_callback(request_id, callback)

    def cancel(self, request_id: str) -> bool:
        """Request cancellation.

        Returns True if the cancellation was accepted (request moved to
        CANCELLED or flagged for late-discard). False if unknown or already
        terminal.
        """
        with self._lock:
            if request_id not in self._requests or request_id in self._terminal:
                return False
            if request_id in self._queue:
                self._queue = [rid for rid in self._queue if rid != request_id]
                self._make_terminal_locked(
                    request_id, AIRequestStatus.CANCELLED, error="request_cancelled"
                )
                return True
            # RUNNING: flag for late-discard when the provider returns.
            self._cancel_requested.add(request_id)
            return True

    def get_result(self, request_id: str) -> Optional[AIResult]:
        with self._lock:
            return self._results.get(request_id)

    def get_status(self, request_id: str) -> Optional[AIRequestStatus]:
        result = self.get_result(request_id)
        return result.status if result is not None else None

    # ------------------------------------------------------------------ #
    # Worker loop (single dedicated thread)
    # ------------------------------------------------------------------ #
    def _worker_loop(self) -> None:
        while not self._worker_stop.is_set():
            rid = self._pop_queue()
            if rid is None:
                # Wait briefly for new work or shutdown.
                self._wake.wait(0.1)
                self._wake.clear()
                continue

            process = False
            with self._lock:
                request = self._requests.get(rid)
                if request is None or rid in self._terminal:
                    continue
                if rid in self._cancel_requested:
                    self._cancel_requested.discard(rid)
                    self._make_terminal_locked(
                        rid, AIRequestStatus.CANCELLED, error="request_cancelled"
                    )
                    continue
                self._mark_running_locked(rid)
                process = True

            if process:
                self._process_request(request)

    def _pop_queue(self) -> Optional[str]:
        with self._lock:
            if not self._queue:
                return None
            return self._queue.pop(0)

    def _mark_running_locked(self, rid: str) -> None:
        req = self._requests.get(rid)
        if req is None:
            return
        self._results[rid] = AIResult(
            request_id=rid, status=AIRequestStatus.RUNNING
        )

    def _process_request(self, request: AIRequest) -> None:
        rid = request.id
        timeout_s = self._timeouts.pop(rid, self._default_timeout_s)
        start = time.perf_counter()

        executor = self._ensure_executor()
        if executor is None:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self._make_terminal(
                rid, AIRequestStatus.FAILED,
                error="executor_unavailable", elapsed_ms=elapsed_ms,
            )
            return

        future: Optional[Future] = None
        try:
            future = executor.submit(self._run_provider, request)
            try:
                output = future.result(timeout=timeout_s)
            except _FuturesTimeoutError:
                # Timeout is NOT cancellation. Discard late result, mark TIMED_OUT,
                # retire the stuck executor so it cannot consume the worker forever.
                if future is not None:
                    future.cancel()
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                self._record_timeout(rid, elapsed_ms=elapsed_ms)
                self._retire_executor()
                return
            except Exception as exc:  # noqa: BLE001 - future raised
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                self._make_terminal(
                    rid, AIRequestStatus.FAILED,
                    error=f"provider_error:{exc}",
                    elapsed_ms=elapsed_ms,
                )
                return

            # Provider returned. Check cancellation first (late-discard).
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            with self._lock:
                cancelled = rid in self._cancel_requested
                self._cancel_requested.discard(rid)
            if cancelled:
                self._make_terminal(
                    rid, AIRequestStatus.CANCELLED,
                    error="request_cancelled", elapsed_ms=elapsed_ms,
                )
                return

            self._make_terminal(
                rid, AIRequestStatus.COMPLETED,
                output=output, elapsed_ms=elapsed_ms, provider="provider",
            )
        except Exception as exc:  # noqa: BLE001 - submission/infrastructure failure
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self._make_terminal(
                rid, AIRequestStatus.FAILED,
                error=f"service_error:{exc}", elapsed_ms=elapsed_ms,
            )
            # Infrastructure failure may leave executor tainted; retire to be safe.
            self._retire_executor()

    def _run_provider(self, request: AIRequest) -> str:
        provider = self._provider
        if provider is None:
            raise RuntimeError("no_provider")
        return provider.analyze(request)

    def _record_timeout(self, request_id: str, elapsed_ms: float) -> None:
        """Mark a request TIMED_OUT, trip/increment the circuit breaker, and
        deliver callbacks."""
        with self._lock:
            self._timeout_count += 1
            callbacks = self._make_terminal_locked(
                request_id, AIRequestStatus.TIMED_OUT,
                error=f"provider_timeout_{self._default_timeout_s}s",
                elapsed_ms=elapsed_ms,
            )
        for cb in callbacks:
            self._deliver_callback(request_id, cb)

    # ------------------------------------------------------------------ #
    # Terminal-state transitions (single writer per request id)
    # ------------------------------------------------------------------ #
    def _make_terminal(self, request_id: str, status: AIRequestStatus,
                       *, error: Optional[str] = None,
                       output: Optional[str] = None,
                       elapsed_ms: Optional[float] = None,
                       provider: Optional[str] = None) -> None:
        with self._lock:
            callbacks = self._make_terminal_locked(
                request_id, status, error=error, output=output,
                elapsed_ms=elapsed_ms, provider=provider,
            )
        # Invoke callbacks outside the lock to avoid re-entrancy and to keep
        # the worker from being stalled by a slow callback.
        for cb in callbacks:
            self._deliver_callback(request_id, cb)

    def _make_terminal_locked(self, request_id: str, status: AIRequestStatus,
                              *, error: Optional[str] = None,
                              output: Optional[str] = None,
                              elapsed_ms: Optional[float] = None,
                              provider: Optional[str] = None) -> list[Callable[[AIResult], None]]:
        """Idempotent terminal transition. Assumes caller holds ``self._lock``.

        Returns the list of callbacks to invoke (outside the lock by the
        caller).
        """
        if request_id in self._terminal:
            return []
        self._terminal.add(request_id)
        result = AIResult(
            request_id=request_id,
            status=status,
            output=output,
            error=error,
            elapsed_ms=elapsed_ms,
            provider=provider,
        )
        self._results[request_id] = result
        callbacks = list(self._callbacks.pop(request_id, []))
        return callbacks

    def _deliver_callback(self, request_id: str,
                          callback: Callable[[AIResult], None]) -> None:
        """Invoke a callback, catching exceptions so one bad callback cannot
        prevent other deliveries or stall the worker."""
        result = self._results.get(request_id)
        if result is None:
            return
        try:
            callback(result)
        except Exception:  # noqa: BLE001 - never let a callback kill delivery
            pass
