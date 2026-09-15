"""Batched MuZero inference service (Stage 4.6, brief S9-S13, S17, S30).

Stage 4.5 called ``model.initial_inference`` / ``model.recurrent_inference``
directly, one root at a time, so every MCTS leaf expansion paid a full neural
forward pass.  Stage 4.6 keeps the *same* model calls but adds a scheduling
layer between search and the network::

    MCTS root A leaf -+
    MCTS root B leaf -+--> InferenceRequest --> batching server --> model batch
    MCTS root C leaf -+                               |
                                                      +--> InferenceResponse

Two backends implement the same tiny protocol, so search code never knows where
inference happens:

* :class:`LocalInferenceBackend` - the Stage 4.5 path (call the model directly,
  in process, with no serialization at all).  It is the reference the batched
  path is checked against.
* :class:`RemoteInferenceBackend` - a client that ships request arrays to
  :class:`BatchedInferenceServer` through queues and waits for the response.
  Collector workers therefore never hold a GPU model copy (brief S11).

The server batches requests until either ``max_inference_batch_size`` requests
are queued or ``max_batch_wait_ms`` has elapsed (brief S12), so latency stays
bounded even when a single collector is active.

Numerical contract (brief S13): the batched path performs the *same* tensor
operations as the unbatched path, only with more rows.  Nothing here changes
priors, rewards, values or latents beyond floating-point associativity, which
``tests/test_muzero_batched_inference.py`` verifies directly.

Weight synchronization (brief S16-S17) is atomic from the inference server's
point of view: the server owns an *active* and a *spare* model copy and swaps
them inside the same lock that guards a running batch, so no batch can observe a
half-loaded parameter set.

Backpressure (brief S7) is placed where it belongs - on the *trajectory* queue,
not on the request path: a worker can have at most one inference request in
flight per search round, so the request and response queues are naturally
bounded by the number of collectors and never block the server.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

from forexmind.muzero.types import NetworkOutput

__all__ = [
    "INITIAL_INFERENCE",
    "RECURRENT_INFERENCE",
    "BatchedInferenceServer",
    "InferenceBackend",
    "InferenceRequest",
    "InferenceResponse",
    "InferenceServiceError",
    "InferenceStats",
    "InferenceTimeoutError",
    "LocalInferenceBackend",
    "RemoteInferenceBackend",
    "compute_batch",
    "slice_network_output",
    "slice_network_output_rows",
]

INITIAL_INFERENCE = "initial"
RECURRENT_INFERENCE = "recurrent"

#: Bound on retained per-batch samples so diagnostics cannot grow without limit.
_MAX_RETAINED_SAMPLES = 50_000


class InferenceServiceError(RuntimeError):
    """Raised when the inference service fails or refuses a request."""


class InferenceTimeoutError(InferenceServiceError):
    """Raised when a request does not receive a response before its deadline."""


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


@dataclass(slots=True)
class InferenceStats:
    """Counters and batch-size/wait/latency samples for one backend (S30)."""

    initial_calls: int = 0
    initial_items: int = 0
    recurrent_calls: int = 0
    recurrent_items: int = 0
    max_batch_size_observed: int = 0
    errors: int = 0
    _batch_sizes: list[int] = field(default_factory=list, repr=False)
    _wait_ms: list[float] = field(default_factory=list, repr=False)
    _latency_ms: list[float] = field(default_factory=list, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(
        self,
        kind: str,
        batch_size: int,
        *,
        wait_ms: float = 0.0,
        latency_ms: float = 0.0,
    ) -> None:
        with self._lock:
            if kind == INITIAL_INFERENCE:
                self.initial_calls += 1
                self.initial_items += int(batch_size)
            else:
                self.recurrent_calls += 1
                self.recurrent_items += int(batch_size)
            self.max_batch_size_observed = max(self.max_batch_size_observed, int(batch_size))
            if len(self._batch_sizes) < _MAX_RETAINED_SAMPLES:
                self._batch_sizes.append(int(batch_size))
                self._wait_ms.append(float(wait_ms))
                self._latency_ms.append(float(latency_ms))

    def record_error(self) -> None:
        with self._lock:
            self.errors += 1

    @property
    def calls(self) -> int:
        return self.initial_calls + self.recurrent_calls

    @property
    def items(self) -> int:
        return self.initial_items + self.recurrent_items

    def snapshot(self) -> dict[str, Any]:
        """JSON-friendly summary of batching behaviour."""
        with self._lock:
            sizes = list(self._batch_sizes)
            waits = list(self._wait_ms)
            latencies = list(self._latency_ms)
            payload: dict[str, Any] = {
                "initial_calls": self.initial_calls,
                "initial_items": self.initial_items,
                "recurrent_calls": self.recurrent_calls,
                "recurrent_items": self.recurrent_items,
                "calls": self.calls,
                "items": self.items,
                "errors": self.errors,
                "max_batch_size_observed": self.max_batch_size_observed,
            }
        payload.update(
            {
                "mean_batch_size": _mean(sizes),
                "median_batch_size": _percentile(sizes, 50.0),
                "p90_batch_size": _percentile(sizes, 90.0),
                "max_batch_size": float(max(sizes)) if sizes else 0.0,
                "mean_batch_wait_ms": _mean(waits),
                "p90_batch_wait_ms": _percentile(waits, 90.0),
                "max_batch_wait_ms": float(max(waits)) if waits else 0.0,
                "mean_inference_latency_ms": _mean(latencies),
                "p90_inference_latency_ms": _percentile(latencies, 90.0),
                "max_inference_latency_ms": float(max(latencies)) if latencies else 0.0,
                "samples": float(len(sizes)),
            }
        )
        return payload

    def to_dict(self) -> dict[str, Any]:
        return self.snapshot()


# --------------------------------------------------------------------------- #
# backend protocol
# --------------------------------------------------------------------------- #


@runtime_checkable
class InferenceBackend(Protocol):
    """What search needs: two batched calls and a network version."""

    def initial_inference(
        self, observations: Any, action_masks: Any | None = None
    ) -> NetworkOutput: ...

    def recurrent_inference(
        self, latent_states: Any, actions: Any, action_masks: Any | None = None
    ) -> NetworkOutput: ...

    @property
    def network_version(self) -> int: ...


class LocalInferenceBackend:
    """Direct in-process model calls (the Stage 4.5 behaviour).

    Nothing is converted, copied or queued: arguments reach the model exactly as
    search built them, which is what makes this the reference implementation for
    the batched/remote path.
    """

    def __init__(self, model: Any, *, stats: InferenceStats | None = None) -> None:
        self.model = model
        self.stats = stats if stats is not None else InferenceStats()

    @property
    def network_version(self) -> int:
        return int(getattr(self.model, "network_version", 0))

    def initial_inference(
        self, observations: Any, action_masks: Any | None = None
    ) -> NetworkOutput:
        start = time.perf_counter()
        with torch.no_grad():
            output = self.model.initial_inference(observations, action_masks)
        self.stats.record(
            INITIAL_INFERENCE,
            int(output.latent_state.shape[0]),
            latency_ms=(time.perf_counter() - start) * 1e3,
        )
        return output

    def recurrent_inference(
        self, latent_states: Any, actions: Any, action_masks: Any | None = None
    ) -> NetworkOutput:
        start = time.perf_counter()
        with torch.no_grad():
            output = self.model.recurrent_inference(latent_states, actions, action_masks)
        self.stats.record(
            RECURRENT_INFERENCE,
            int(output.latent_state.shape[0]),
            latency_ms=(time.perf_counter() - start) * 1e3,
        )
        return output

    def diagnostics(self) -> dict[str, Any]:
        """Same shape as the remote client's diagnostics, so pools can report it."""
        return {
            "mode": "local",
            "worker_id": None,
            "network_version": self.network_version,
            "round_trips": 0,
            "mean_round_trip_ms": 0.0,
            **self.stats.snapshot(),
        }


# --------------------------------------------------------------------------- #
# wire format
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class InferenceRequest:
    """One batched inference request travelling to the service."""

    worker_id: int
    request_id: int
    kind: str
    observations: np.ndarray | None = None
    action_masks: np.ndarray | None = None
    latent_states: np.ndarray | None = None
    actions: np.ndarray | None = None
    sent_at: float = 0.0

    def batch_size(self) -> int:
        array = self.observations if self.kind == INITIAL_INFERENCE else self.latent_states
        if array is None:
            return 0
        return int(np.asarray(array).shape[0])


@dataclass(slots=True)
class InferenceResponse:
    """The service's answer: decoded arrays, or a hard error (never a deadlock)."""

    request_id: int
    ok: bool
    network_version: int = 0
    latent_state: np.ndarray | None = None
    policy_logits: np.ndarray | None = None
    value: np.ndarray | None = None
    reward: np.ndarray | None = None
    value_logits: np.ndarray | None = None
    reward_logits: np.ndarray | None = None
    error: str | None = None
    wait_ms: float = 0.0


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu").numpy()
    return np.asarray(value)


def broadcast_mask(mask: Any, batch_size: int) -> np.ndarray | None:
    """Normalise an action mask to ``[batch_size, num_actions]`` on the wire."""
    if mask is None:
        return None
    array = np.asarray(_as_numpy(mask), dtype=bool)
    if array.ndim == 1:
        return np.ascontiguousarray(np.broadcast_to(array, (batch_size, array.shape[0])))
    if array.ndim != 2:
        raise ValueError(f"action mask must be 1-D or 2-D, got shape {array.shape}")
    if array.shape[0] == 1 and batch_size > 1:
        array = np.broadcast_to(array, (batch_size, array.shape[1]))
    if array.shape[0] != batch_size:
        raise ValueError(
            f"action mask batch {array.shape[0]} does not match request batch {batch_size}"
        )
    return np.ascontiguousarray(array)


def all_true_mask(batch_size: int, num_actions: int) -> np.ndarray:
    return np.ones((batch_size, num_actions), dtype=bool)


def num_action_columns(requests: Sequence[InferenceRequest]) -> int:
    """Width of the action axis, taken from the first request that carries one."""
    for request in requests:
        if request.action_masks is not None:
            array = np.asarray(request.action_masks)
            return int(array.reshape(array.shape[0], -1).shape[-1])
    raise ValueError("num_action_columns requires at least one request with an action mask")


def network_output_to_response(
    output: NetworkOutput, *, request_id: int, network_version: int, wait_ms: float = 0.0
) -> InferenceResponse:
    """Convert a :class:`NetworkOutput` into the picklable wire response."""
    return InferenceResponse(
        request_id=request_id,
        ok=True,
        network_version=int(network_version),
        latent_state=_as_numpy(output.latent_state),
        policy_logits=_as_numpy(output.policy_logits),
        value=_as_numpy(output.value),
        reward=_as_numpy(output.reward),
        value_logits=None if output.value_logits is None else _as_numpy(output.value_logits),
        reward_logits=None if output.reward_logits is None else _as_numpy(output.reward_logits),
        wait_ms=float(wait_ms),
    )


def response_to_network_output(response: InferenceResponse) -> NetworkOutput:
    """Rebuild a :class:`NetworkOutput` from a service response."""
    if not response.ok:
        raise InferenceServiceError(response.error or "inference service failure")
    if response.latent_state is None or response.policy_logits is None:
        raise InferenceServiceError("inference response is missing its arrays")
    assert response.value is not None and response.reward is not None
    return NetworkOutput(
        latent_state=torch.from_numpy(_contiguous(response.latent_state)),
        policy_logits=torch.from_numpy(_contiguous(response.policy_logits)),
        value=torch.from_numpy(_contiguous(response.value)),
        reward=torch.from_numpy(_contiguous(response.reward)),
        value_logits=(
            None
            if response.value_logits is None
            else torch.from_numpy(_contiguous(response.value_logits))
        ),
        reward_logits=(
            None
            if response.reward_logits is None
            else torch.from_numpy(_contiguous(response.reward_logits))
        ),
    )


def _contiguous(array: np.ndarray) -> np.ndarray:
    return array if array.flags["C_CONTIGUOUS"] else np.ascontiguousarray(array)


def slice_network_output(output: NetworkOutput, index: int) -> NetworkOutput:
    """Row ``index`` of a batched output, as a 1-row :class:`NetworkOutput`."""

    def row(tensor: torch.Tensor | None) -> torch.Tensor | None:
        return None if tensor is None else tensor[index : index + 1]

    return NetworkOutput(
        latent_state=output.latent_state[index : index + 1],
        policy_logits=output.policy_logits[index : index + 1],
        value=output.value[index : index + 1],
        reward=output.reward[index : index + 1],
        value_logits=row(output.value_logits),
        reward_logits=row(output.reward_logits),
    )


def compute_batch(model: Any, requests: Sequence[InferenceRequest]) -> list[InferenceResponse]:
    """Run one homogeneous batch of requests through ``model`` (brief S9-S13).

    ``requests`` must share a ``kind`` (the caller groups them).  Tensors are
    built exactly the way the unbatched path builds them, so a batch of one row
    is identical to a single call.

    One response is produced **per request**, holding exactly the rows that
    request contributed; a request may itself carry several rows (one per MCTS
    root in the worker), which is what makes a worker's batched call still a
    single round trip.
    """
    if not requests:
        return []
    kind = requests[0].kind
    if any(request.kind != kind for request in requests):
        raise ValueError("compute_batch requires requests of a single kind")
    row_counts = [request.batch_size() for request in requests]
    if any(count <= 0 for count in row_counts):
        raise InferenceServiceError("every inference request must carry at least one row")
    offsets = np.concatenate([[0], np.cumsum(row_counts)])
    version = int(getattr(model, "inference_network_version", 0))
    with torch.no_grad():
        if kind == INITIAL_INFERENCE:
            rows = [np.atleast_2d(np.asarray(request.observations)) for request in requests]
            observations = torch.as_tensor(_stack_rows(rows))
            if all(request.action_masks is None for request in requests):
                output = model.initial_inference(observations, None)
            else:
                width = num_action_columns(requests)
                masks = _stack_rows(
                    [
                        broadcast_mask(request.action_masks, request.batch_size())
                        if request.action_masks is not None
                        else all_true_mask(request.batch_size(), width)
                        for request in requests
                    ]
                )
                output = model.initial_inference(observations, torch.as_tensor(masks))
        elif kind == RECURRENT_INFERENCE:
            rows = [np.atleast_2d(np.asarray(request.latent_states)) for request in requests]
            latents = torch.as_tensor(_stack_rows(rows))
            actions = torch.as_tensor(
                np.ascontiguousarray(
                    np.concatenate(
                        [np.asarray(request.actions).reshape(-1) for request in requests]
                    ),
                    dtype=np.int64,
                )
            )
            if all(request.action_masks is None for request in requests):
                output = model.recurrent_inference(latents, actions, None)
            else:
                width = num_action_columns(requests)
                mask_rows = [
                    broadcast_mask(request.action_masks, request.batch_size())
                    if request.action_masks is not None
                    else all_true_mask(request.batch_size(), width)
                    for request in requests
                ]
                masks = _stack_rows(mask_rows)
                if bool(masks.all()):
                    output = model.recurrent_inference(latents, actions, None)
                else:
                    output = model.recurrent_inference(latents, actions, torch.as_tensor(masks))
        else:
            raise ValueError(f"unknown inference request kind {kind!r}")

    count = int(output.latent_state.shape[0])
    if count != sum(row_counts):
        raise InferenceServiceError(
            f"model returned {count} rows for {sum(row_counts)} requested rows; batched "
            "inference requires one output row per requested row"
        )
    responses: list[InferenceResponse] = []
    for request, start, stop in zip(requests, offsets[:-1], offsets[1:], strict=True):
        wait_ms = (
            0.0
            if request.sent_at <= 0.0
            else max(0.0, (time.perf_counter() - request.sent_at) * 1e3)
        )
        responses.append(
            network_output_to_response(
                slice_network_output_rows(output, int(start), int(stop)),
                request_id=request.request_id,
                network_version=version,
                wait_ms=wait_ms,
            )
        )
    return responses


def _stack_rows(rows: Sequence[np.ndarray]) -> np.ndarray:
    return np.ascontiguousarray(np.concatenate([np.atleast_2d(row) for row in rows], axis=0))


def slice_network_output_rows(output: NetworkOutput, start: int, stop: int) -> NetworkOutput:
    """Rows ``[start, stop)`` of a batched output as a :class:`NetworkOutput`."""

    def rows(tensor: torch.Tensor | None) -> torch.Tensor | None:
        return None if tensor is None else tensor[start:stop]

    return NetworkOutput(
        latent_state=output.latent_state[start:stop],
        policy_logits=output.policy_logits[start:stop],
        value=output.value[start:stop],
        reward=output.reward[start:stop],
        value_logits=rows(output.value_logits),
        reward_logits=rows(output.reward_logits),
    )


# --------------------------------------------------------------------------- #
# server (runs in the trainer process)
# --------------------------------------------------------------------------- #


class BatchedInferenceServer:
    """Central inference service with a bounded batch wait (brief S11-S12)."""

    def __init__(
        self,
        model_factory: Callable[[], Any],
        *,
        request_queue: Any,
        response_queues: dict[int, Any],
        max_batch_size: int = 32,
        max_batch_wait_ms: float = 2.0,
        poll_timeout_s: float = 0.05,
        stats: InferenceStats | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError(f"max_batch_size must be >= 1, got {max_batch_size}")
        if max_batch_wait_ms < 0.0:
            raise ValueError(f"max_batch_wait_ms must be >= 0, got {max_batch_wait_ms}")
        self.model_factory = model_factory
        self.request_queue = request_queue
        self.response_queues = dict(response_queues)
        self.max_batch_size = int(max_batch_size)
        self.max_batch_wait_ms = float(max_batch_wait_ms)
        self.poll_timeout_s = float(poll_timeout_s)
        self.stats = stats if stats is not None else InferenceStats()
        self.device = torch.device(device) if device is not None else None

        self._active = self._build_model()
        self._spare = self._build_model()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fatal: BaseException | None = None
        self._version = 0
        self._model_version = ""
        self._batches = 0

    # -- lifecycle ------------------------------------------------------------

    def _build_model(self) -> Any:
        model = self.model_factory()
        if self.device is not None:
            model.to(self.device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("inference server already started")
        self._thread = threading.Thread(
            target=self._serve_forever, name="muzero-inference-server", daemon=True
        )
        self._thread.start()

    def stop(self, *, timeout_s: float = 10.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout_s)
            if thread.is_alive():  # pragma: no cover - defensive
                raise InferenceServiceError("inference server thread did not stop")
        self._thread = None

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def fatal_error(self) -> BaseException | None:
        return self._fatal

    @property
    def network_version(self) -> int:
        return self._version

    @property
    def model_version(self) -> str:
        return self._model_version

    # -- weight synchronization (S16-S17) -------------------------------------

    def sync_weights(
        self, model: Any, *, version: int, model_version_string: str = ""
    ) -> dict[str, Any]:
        """Atomically publish ``model``'s weights as inference version ``version``.

        The load happens into the *spare* copy while holding the same lock that
        guards a running batch, so inference never observes a partially loaded
        model; the publication itself is a reference assignment.
        """
        start = time.perf_counter()
        state = model.state_dict() if hasattr(model, "state_dict") else dict(model)
        with self._lock:
            self._spare.load_state_dict({key: value for key, value in state.items()}, strict=True)
            self._active, self._spare = self._spare, self._active
            self._version = int(version)
            self._model_version = str(model_version_string)
        return {
            "version": int(version),
            "model_version": str(model_version_string),
            "sync_seconds": time.perf_counter() - start,
        }

    # -- server loop ----------------------------------------------------------

    def _serve_forever(self) -> None:  # pragma: no cover - exercised through the pool
        while not self._stop.is_set():
            try:
                first = self._get_request(self.poll_timeout_s)
            except queue.Empty:
                continue
            if first is None:
                continue
            batch = [first]
            deadline = time.perf_counter() + self.max_batch_wait_ms / 1e3
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    break
                try:
                    item = self._get_request(remaining)
                except queue.Empty:
                    break
                if item is not None:
                    batch.append(item)
            self._serve_batch(batch)

    def _get_request(self, timeout_s: float) -> InferenceRequest | None:
        item = self.request_queue.get(timeout=timeout_s)
        if item is None:
            self._stop.set()
            return None
        if not isinstance(item, InferenceRequest):  # pragma: no cover - defensive
            raise InferenceServiceError(f"unexpected request type {type(item).__name__}")
        return item

    def _serve_batch(self, batch: Sequence[InferenceRequest]) -> None:
        started = time.perf_counter()
        with self._lock:
            if self._fatal is not None:
                self._respond_with_error(batch, self._fatal)
                return
            try:
                model = self._active
                for kind, group in _group_by_kind(batch):
                    for request, response in zip(
                        group, compute_batch(model, group), strict=True
                    ):
                        response.network_version = self._version
                        self.stats.record(
                            kind,
                            request.batch_size(),
                            wait_ms=response.wait_ms,
                            latency_ms=(time.perf_counter() - started) * 1e3,
                        )
                        self._send(request.worker_id, response)
                self._batches += 1
            except BaseException as exc:
                self._fatal = exc
                self.stats.record_error()
                self._respond_with_error(batch, exc)

    def _respond_with_error(self, batch: Sequence[InferenceRequest], exc: BaseException) -> None:
        message = f"{type(exc).__name__}: {exc}"
        for request in batch:
            self._send(
                request.worker_id,
                InferenceResponse(
                    request_id=request.request_id,
                    ok=False,
                    network_version=self._version,
                    error=message,
                ),
            )

    def _send(self, worker_id: int, response: InferenceResponse) -> None:
        target = self.response_queues.get(int(worker_id))
        if target is None:  # pragma: no cover - defensive
            raise InferenceServiceError(f"no response queue registered for worker {worker_id}")
        target.put(response)

    # -- diagnostics ----------------------------------------------------------

    def diagnostics(self) -> dict[str, Any]:
        return {
            "alive": self.is_alive,
            "batches": int(self._batches),
            "network_version": int(self._version),
            "model_version": self._model_version,
            "max_batch_size": self.max_batch_size,
            "max_batch_wait_ms": self.max_batch_wait_ms,
            "fatal_error": (
                None
                if self._fatal is None
                else f"{type(self._fatal).__name__}: {self._fatal}"
            ),
            **self.stats.snapshot(),
        }


def _group_by_kind(
    requests: Sequence[InferenceRequest],
) -> list[tuple[str, list[InferenceRequest]]]:
    """Split a batch into consecutive runs of the same request kind."""
    groups: list[tuple[str, list[InferenceRequest]]] = []
    for request in requests:
        if groups and groups[-1][0] == request.kind:
            groups[-1][1].append(request)
        else:
            groups.append((request.kind, [request]))
    return groups


# --------------------------------------------------------------------------- #
# client (runs inside collector workers)
# --------------------------------------------------------------------------- #


class RemoteInferenceBackend:
    """Inference backend that talks to a :class:`BatchedInferenceServer`.

    Every call is one request/response round trip.  Failures raise
    :class:`InferenceServiceError` / :class:`InferenceTimeoutError` instead of
    blocking forever, so a broken service fails loudly (brief S38).
    """

    def __init__(
        self,
        *,
        worker_id: int,
        request_queue: Any,
        response_queue: Any,
        timeout_s: float = 120.0,
        stats: InferenceStats | None = None,
    ) -> None:
        self.worker_id = int(worker_id)
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.timeout_s = float(timeout_s)
        self.stats = stats if stats is not None else InferenceStats()
        self._next_request_id = 0
        self._version = 0
        self._model_version = ""
        self._round_trips = 0
        self._round_trip_seconds = 0.0

    @property
    def network_version(self) -> int:
        return self._version

    @property
    def model_version(self) -> str:
        return self._model_version

    def _next_id(self) -> int:
        self._next_request_id += 1
        return self._next_request_id

    def initial_inference(
        self, observations: Any, action_masks: Any | None = None
    ) -> NetworkOutput:
        array = np.asarray(_as_numpy(observations), dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2:
            raise ValueError(f"observations must be 1-D or 2-D, got shape {array.shape}")
        batch = int(array.shape[0])
        request = InferenceRequest(
            worker_id=self.worker_id,
            request_id=self._next_id(),
            kind=INITIAL_INFERENCE,
            observations=np.ascontiguousarray(array),
            action_masks=(
                None
                if action_masks is None
                else broadcast_mask(action_masks, batch)
            ),
            sent_at=time.perf_counter(),
        )
        return self._round_trip(request)

    def recurrent_inference(
        self, latent_states: Any, actions: Any, action_masks: Any | None = None
    ) -> NetworkOutput:
        latents = np.asarray(_as_numpy(latent_states), dtype=np.float32)
        if latents.ndim == 1:
            latents = latents.reshape(1, -1)
        if latents.ndim != 2:
            raise ValueError(f"latent_states must be 1-D or 2-D, got shape {latents.shape}")
        batch = int(latents.shape[0])
        action_array = np.asarray(_as_numpy(actions)).reshape(-1)
        if action_array.size == 1 and batch > 1:
            action_array = np.repeat(action_array, batch)
        if action_array.size != batch:
            raise ValueError(f"actions must have {batch} entries, got {action_array.size}")
        request = InferenceRequest(
            worker_id=self.worker_id,
            request_id=self._next_id(),
            kind=RECURRENT_INFERENCE,
            latent_states=np.ascontiguousarray(latents),
            actions=np.ascontiguousarray(action_array, dtype=np.int64),
            action_masks=(
                None if action_masks is None else broadcast_mask(action_masks, batch)
            ),
            sent_at=time.perf_counter(),
        )
        return self._round_trip(request)

    def _round_trip(self, request: InferenceRequest) -> NetworkOutput:
        start = time.perf_counter()
        self.request_queue.put(request)
        try:
            response = self.response_queue.get(timeout=self.timeout_s)
        except queue.Empty as exc:
            raise InferenceTimeoutError(
                f"no inference response for request {request.request_id} within "
                f"{self.timeout_s:.1f}s"
            ) from exc
        elapsed = time.perf_counter() - start
        self._round_trips += 1
        self._round_trip_seconds += elapsed
        if not isinstance(response, InferenceResponse):
            raise InferenceServiceError(f"unexpected response type {type(response).__name__}")
        if response.request_id != request.request_id:
            raise InferenceServiceError(
                f"response {response.request_id} does not match request {request.request_id}"
            )
        self.stats.record(
            request.kind,
            request.batch_size(),
            wait_ms=response.wait_ms,
            latency_ms=elapsed * 1e3,
        )
        if not response.ok:
            self.stats.record_error()
            raise InferenceServiceError(response.error or "inference service failure")
        self._version = int(response.network_version)
        return response_to_network_output(response)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "network_version": self._version,
            "round_trips": self._round_trips,
            "mean_round_trip_ms": (
                1e3 * self._round_trip_seconds / self._round_trips if self._round_trips else 0.0
            ),
            **self.stats.snapshot(),
        }
