"""Memory planning for multiprocessing training (PPO memory audit).

Pure helpers used by the trainer and the diagnostic tools to turn
``compute.num_workers: auto`` into a RAM-safe worker count and to warn when
the estimated process-tree memory exceeds a configured budget fraction.
"""

from __future__ import annotations


def total_ram_mb() -> float:
    """Total system RAM in MB, or 0 when undetectable."""
    try:
        from importlib import import_module

        psutil = import_module("psutil")
        return float(psutil.virtual_memory().total / 1e6)
    except Exception:
        return 0.0


def estimate_auto_worker_count(
    *,
    worker_rss_mb: float,
    memory_limit_fraction: float = 0.70,
    trainer_reserve_mb: float = 2048.0,
    cap: int = 256,
) -> int:
    """Safe worker count so ``worker_count * worker_rss + reserve`` fits RAM.

    ``cap`` keeps auto selection from wildly exceeding the CPU count; the
    memory/throughput benchmark is the real source of truth.
    """
    ram_mb = total_ram_mb()
    if ram_mb <= 0:
        return 8
    if worker_rss_mb <= 0:
        return 8
    budget_mb = memory_limit_fraction * ram_mb
    avail_mb = max(0.0, budget_mb - trainer_reserve_mb)
    n = int(avail_mb // worker_rss_mb)
    return max(1, min(n, cap))


def estimate_process_tree_mb(
    *,
    worker_rss_mb: float,
    workers: int,
    trainer_reserve_mb: float = 2048.0,
) -> float:
    """Estimated process-tree RSS: trainer reserve + workers x per-worker RSS."""
    return trainer_reserve_mb + workers * worker_rss_mb
