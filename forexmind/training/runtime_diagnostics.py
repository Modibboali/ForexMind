"""Runtime diagnostics for multiprocessing training.

These helpers are intentionally lightweight: they report the real process tree
and thread settings of the machine they are running on, without trying to
emulate a larger Kaggle CPU allocation locally.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterable
from importlib import import_module
from typing import Any, cast


def _psutil_module() -> Any | None:
    try:
        return import_module("psutil")
    except ImportError:
        return None


def logical_cpu_count() -> int:
    psutil = _psutil_module()
    if psutil is not None:
        count = psutil.cpu_count(logical=True)
        if count is not None:
            return int(count)
    return int(os.cpu_count() or 1)


def cpu_affinity(pid: int | None = None) -> list[int] | None:
    target_pid = os.getpid() if pid is None else pid
    if hasattr(os, "sched_getaffinity") and target_pid == os.getpid():
        sched_getaffinity = os.sched_getaffinity
        return sorted(int(c) for c in sched_getaffinity(0))

    psutil = _psutil_module()
    if psutil is None:
        return None
    try:
        proc = psutil.Process(target_pid)
        affinity = getattr(proc, "cpu_affinity", None)
        if affinity is None:
            return None
        return [int(c) for c in affinity()]
    except Exception:
        return None


def torch_thread_report() -> dict[str, int | str]:
    try:
        import torch
    except ImportError:
        return {"available": "no"}
    report: dict[str, int | str] = {
        "available": "yes",
        "num_threads": int(torch.get_num_threads()),
    }
    try:
        report["num_interop_threads"] = int(torch.get_num_interop_threads())
    except RuntimeError:
        report["num_interop_threads"] = "unavailable"
    return report


def thread_env_report() -> dict[str, str]:
    return {
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", ""),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", ""),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", ""),
    }


def _alive_processes(pids: Iterable[int]) -> list[Any]:
    psutil = _psutil_module()
    if psutil is None:
        return []
    procs: list[Any] = []
    for pid in pids:
        try:
            proc = psutil.Process(int(pid))
            if proc.is_running():
                procs.append(proc)
        except Exception:
            continue
    return procs


def sample_process_tree(
    *,
    trainer_pid: int | None = None,
    worker_pids: Iterable[int] = (),
    sample_seconds: float = 1.0,
) -> dict[str, float | int | str]:
    """Sample trainer, worker, and full process-tree CPU percent.

    Percent values follow ``psutil`` semantics: 100.0 roughly means one fully
    utilized CPU core, so 201.0 is about two effective cores.
    """
    psutil = _psutil_module()
    if psutil is None:
        return {
            "psutil_available": "no",
            "trainer_cpu_percent": 0.0,
            "worker_cpu_percent": 0.0,
            "process_tree_cpu_percent": 0.0,
            "effective_cores_utilized": 0.0,
        }

    root_pid = os.getpid() if trainer_pid is None else int(trainer_pid)
    try:
        root = psutil.Process(root_pid)
    except Exception:
        return {
            "psutil_available": "yes",
            "trainer_cpu_percent": 0.0,
            "worker_cpu_percent": 0.0,
            "process_tree_cpu_percent": 0.0,
            "effective_cores_utilized": 0.0,
        }

    workers = _alive_processes(worker_pids)
    children = root.children(recursive=True)
    proc_by_pid: dict[int, Any] = {int(root.pid): root}
    for proc in [*children, *workers]:
        proc_by_pid[int(proc.pid)] = proc

    for proc in proc_by_pid.values():
        try:
            proc.cpu_percent(None)
        except Exception:
            continue
    if sample_seconds > 0:
        time.sleep(sample_seconds)

    cpu_by_pid: dict[int, float] = {}
    for pid, proc in proc_by_pid.items():
        try:
            cpu_by_pid[pid] = float(proc.cpu_percent(None))
        except Exception:
            cpu_by_pid[pid] = 0.0

    worker_pid_set = {int(p.pid) for p in workers}
    trainer_cpu = cpu_by_pid.get(int(root.pid), 0.0)
    worker_cpu = sum(cpu_by_pid.get(pid, 0.0) for pid in worker_pid_set)
    total_cpu = sum(cpu_by_pid.values())
    return {
        "psutil_available": "yes",
        "trainer_cpu_percent": round(trainer_cpu, 3),
        "worker_cpu_percent": round(worker_cpu, 3),
        "process_tree_cpu_percent": round(total_cpu, 3),
        "effective_cores_utilized": round(total_cpu / 100.0, 3),
    }


class ProcessTreeCpuSampler:
    """Prime process CPU counters, then report usage since ``start()``."""

    def __init__(
        self,
        *,
        trainer_pid: int | None = None,
        worker_pids: Iterable[int] = (),
    ) -> None:
        self._trainer_pid = os.getpid() if trainer_pid is None else int(trainer_pid)
        self._worker_pids = [int(pid) for pid in worker_pids]
        self._procs: dict[int, Any] = {}

    def start(self) -> None:
        psutil = _psutil_module()
        if psutil is None:
            self._procs = {}
            return
        try:
            root = psutil.Process(self._trainer_pid)
        except Exception:
            self._procs = {}
            return
        procs: dict[int, Any] = {int(root.pid): root}
        for proc in [*root.children(recursive=True), *_alive_processes(self._worker_pids)]:
            procs[int(proc.pid)] = proc
        for proc in procs.values():
            try:
                proc.cpu_percent(None)
            except Exception:
                continue
        self._procs = procs

    @property
    def has_procs(self) -> bool:
        """True when psutil is available and at least the trainer was captured."""
        return bool(self._procs)

    def stop(self) -> dict[str, float | int | str]:
        psutil = _psutil_module()
        if psutil is None or not self._procs:
            return {
                "psutil_available": "no",
                "trainer_cpu_percent": 0.0,
                "worker_cpu_percent": 0.0,
                "process_tree_cpu_percent": 0.0,
                "effective_cores_utilized": 0.0,
            }
        cpu_by_pid: dict[int, float] = {}
        for pid, proc in self._procs.items():
            try:
                cpu_by_pid[pid] = float(proc.cpu_percent(None))
            except Exception:
                cpu_by_pid[pid] = 0.0
        worker_pid_set = set(self._worker_pids)
        trainer_cpu = cpu_by_pid.get(self._trainer_pid, 0.0)
        worker_cpu = sum(cpu_by_pid.get(pid, 0.0) for pid in worker_pid_set)
        total_cpu = sum(cpu_by_pid.values())
        return {
            "psutil_available": "yes",
            "trainer_cpu_percent": round(trainer_cpu, 3),
            "worker_cpu_percent": round(worker_cpu, 3),
            "process_tree_cpu_percent": round(total_cpu, 3),
            "effective_cores_utilized": round(total_cpu / 100.0, 3),
        }


def collect_runtime_report(
    *,
    worker_pids: Iterable[int] = (),
    sample_seconds: float = 1.0,
) -> dict[str, object]:
    pids = [int(pid) for pid in worker_pids]
    affinity = cpu_affinity()
    cpu_sample = sample_process_tree(worker_pids=pids, sample_seconds=sample_seconds)
    return {
        "logical_cpus": logical_cpu_count(),
        "process_pid": os.getpid(),
        "parent_pid": os.getppid(),
        "worker_pids": pids,
        "alive_workers": len(_alive_processes(pids)),
        "cpu_affinity": affinity,
        "cpu_affinity_count": len(affinity) if affinity is not None else None,
        "torch": torch_thread_report(),
        "env": thread_env_report(),
        **cpu_sample,
    }


def print_process_tree_report(
    *,
    worker_pids: Iterable[int] = (),
    workers_configured: int | None = None,
    sample_seconds: float = 1.0,
    cpu_sample: dict[str, object] | None = None,
    producer_ids: Iterable[int] | None = None,
    label: str = "PROCESS TREE",
) -> dict[str, object]:
    report = collect_runtime_report(
        worker_pids=worker_pids,
        sample_seconds=0.0 if cpu_sample is not None else sample_seconds,
    )
    if cpu_sample is not None:
        report.update(cpu_sample)
    pids = report["worker_pids"]
    torch_report = report["torch"]
    env_report = cast(dict[str, str], report["env"])
    print("=" * 60)
    print(label)
    print("=" * 60)
    print(f"Trainer PID: {report['process_pid']}")
    if workers_configured is not None:
        print(f"Workers configured: {workers_configured}")
    print(f"Worker PIDs: {pids}")
    print(f"Workers alive: {report['alive_workers']}")
    if producer_ids is not None:
        print(f"Workers producing transitions: {len(set(int(p) for p in producer_ids))}")
    else:
        print("Workers producing transitions: requires benchmark counters")
    print("")
    print(f"Trainer CPU: {report['trainer_cpu_percent']}")
    print(f"Worker CPU aggregate: {report['worker_cpu_percent']}")
    print(f"Process-tree CPU: {report['process_tree_cpu_percent']}")
    print("")
    print(f"CPU cores available: {report['logical_cpus']}")
    print(f"Effective cores utilized: {report['effective_cores_utilized']}")
    print(f"CPU affinity: {report['cpu_affinity']}")
    print("")
    print(f"PyTorch threads: {torch_report}")
    print(f"OMP_NUM_THREADS: {env_report['OMP_NUM_THREADS']}")
    print(f"MKL_NUM_THREADS: {env_report['MKL_NUM_THREADS']}")
    print(f"OPENBLAS_NUM_THREADS: {env_report['OPENBLAS_NUM_THREADS']}")
    print("=" * 60)
    return report
