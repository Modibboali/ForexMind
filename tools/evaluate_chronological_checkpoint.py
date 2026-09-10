"""Evaluate a PPO checkpoint as one continuous account path per instrument.

Results are written per instrument as soon as each path completes. Re-running
the command resumes by skipping files with the same checkpoint hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from forexmind.evaluation.chronological import ChronologicalEvaluator
from forexmind.observation.encoder import EncoderConfig, ObservationEncoder
from forexmind.observation.window import WindowConfig
from forexmind.training.benchmark import load_checkpoint_policy
from forexmind.training.checkpoint import resolve_checkpoint
from forexmind.training.config import ExperimentConfig
from forexmind.training.data import DEFAULT_INSTRUMENT_ORDER, DEFAULT_PROCESSED_DIR
from forexmind.training.dataset_mmap import resolve_dataset
from forexmind.training.trainer import build_env_config


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _completed_report(path: Path, checkpoint_sha256: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    return report if report.get("checkpoint_sha256") == checkpoint_sha256 else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--instruments", nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/reports/stage35_chronological")
    parser.add_argument("--combine-only", action="store_true")
    args = parser.parse_args()

    checkpoint = resolve_checkpoint(args.checkpoint, run_root=args.run_root)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = ExperimentConfig.from_dict(state.get("config") or {})
    configured = tuple(config.environment.instruments) or DEFAULT_INSTRUMENT_ORDER
    selected = tuple(value.upper() for value in args.instruments) or configured
    unknown = sorted(set(selected) - set(configured))
    if unknown:
        raise ValueError(f"instruments absent from checkpoint configuration: {unknown}")
    if state.get("algorithm") != "ppo" or state.get("action_policy") != "categorical_v1":
        raise ValueError("chronological Stage 3.5 validation requires categorical PPO")

    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dataset, backend = resolve_dataset(
        processed_dir=DEFAULT_PROCESSED_DIR,
        instruments=configured,
        backend=config.compute.dataset_backend,
    )
    env_config = build_env_config(config.environment)
    encoder = ObservationEncoder(
        EncoderConfig(
            context_length=config.environment.context_length,
            initial_balance=env_config.margin.initial_balance,
        )
    )
    window = WindowConfig(context_length=config.environment.context_length)
    policy, algorithm = load_checkpoint_policy(
        checkpoint, encoder.config.spec.encoded_shape[0], config.model
    )
    evaluator = ChronologicalEvaluator(dataset, env_config, encoder, window)

    if not args.combine_only:
        for instrument in selected:
            path = out / f"{instrument}.json"
            if _completed_report(path, digest) is not None:
                print(f"skip {instrument}: matching completed report", flush=True)
                continue
            print(f"start {instrument} chronological {args.split}", flush=True)
            result = evaluator.evaluate(
                policy,
                algorithm,
                split=args.split,
                instruments=[instrument],
                seed=args.seed + configured.index(instrument),
            )
            report = {
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": digest,
                "checkpoint_step": int(state.get("env_steps", 0)),
                "dataset_backend": backend,
                **result["per_instrument"][instrument],
            }
            _write_json(path, report)
            print(
                f"done {instrument}: periods={report['n_periods']} "
                f"return={report['total_return']:+.8f} sharpe={report['sharpe']:+.4f}",
                flush=True,
            )

    reports: dict[str, Any] = {}
    for instrument in configured:
        completed = _completed_report(out / f"{instrument}.json", digest)
        if completed is not None:
            reports[instrument] = completed
    summary = {
        "evaluation_type": "chronological_per_instrument",
        "is_portfolio_path": False,
        "combined_portfolio_metrics": None,
        "combined_portfolio_metrics_unavailable_reason": (
            "Independent instrument accounts are not a multi-instrument portfolio"
        ),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": digest,
        "checkpoint_step": int(state.get("env_steps", 0)),
        "split": args.split,
        "seed": args.seed,
        "completed_instruments": sorted(reports),
        "missing_instruments": sorted(set(configured) - set(reports)),
        "per_instrument": reports,
    }
    if args.combine_only or selected == configured:
        _write_json(out / "chronological_validation.json", summary)
        print(
            f"summary: {len(reports)}/{len(configured)} instruments complete -> "
            f"{out / 'chronological_validation.json'}",
            flush=True,
        )
    else:
        print(f"partial: {len(reports)}/{len(configured)} instruments complete", flush=True)


if __name__ == "__main__":
    main()
