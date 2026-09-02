"""PPO policy diagnosis: multi-agent comparison on identical validation episodes.

Runs PPO + 6 baselines (flat, long, short, momentum, mean_reversion, sma_crossover)
on identical validation episode specs, computes per-agent metrics, PPO action
statistics (mean/std/min/max, % long/short/flat, position changes, turnover),
and saves per-episode and per-instrument results.

Usage:
    python -m tools.diagnose_ppo --checkpoint best.pt --episodes 100 --seed 42
    python -m tools.diagnose_ppo --checkpoint runs/ppo_cpu_seed42/checkpoints/best.pt --episodes 100
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from decimal import Decimal

from forexmind.baselines.base import make_agent
from forexmind.observation.encoder import EncoderConfig, ObservationEncoder
from forexmind.observation.window import WindowConfig
from forexmind.training.benchmark import (
    load_checkpoint_policy,
    build_test_episode_specs,
)
from forexmind.training.checkpoint import resolve_checkpoint
from forexmind.training.config import ExperimentConfig
from forexmind.training.data import (
    DEFAULT_INSTRUMENT_ORDER,
    DEFAULT_PROCESSED_DIR,
    make_training_dataset,
)
from forexmind.training.policies import PolicyAgent
from forexmind.training.trainer import build_env_config
from forexmind.evaluation.runner import EvaluationRunner
from forexmind.evaluation.metrics import compute_series_metrics
from forexmind.evaluation.aggregation import aggregate_across_instruments


BASELINE_AGENTS = ("flat", "long", "short", "momentum", "mean_reversion", "sma_crossover")


def _f(value: Any, default: float = 0.0) -> float:
    """Coerce an unknown value to float (handles Decimal-serialized strings)."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _trade_notional(trade: dict[str, Any]) -> float:
    """Account-currency notional of a trade, falling back to quote notional."""
    if "notional_account" in trade:
        return abs(_f(trade.get("notional_account")))
    units = trade.get("units_delta", 0.0)
    price = trade.get("execution_price") or 0.0
    return abs(_f(units)) * _f(price)


def _pooled_turnover(trajectories: list[Any]) -> float:
    """Turnover from a list of trajectories."""
    total_notional = 0.0
    capital = 0.0
    for t in trajectories:
        capital += _f(t.info.get("initial_balance"))
        for trade in t.trade_log:
            total_notional += _trade_notional(trade)
    return total_notional / capital if capital > 0 else 0.0


def _ppo_action_statistics(trajectories: list[Any]) -> dict[str, Any]:
    """Compute PPO action statistics (mean/std/min/max, % long/short/flat, position changes)."""
    all_actions = []
    all_position_units = []
    all_trades = 0
    
    for t in trajectories:
        all_actions.extend(t.actions)
        all_position_units.extend(t.position_units)
        all_trades += len(t.trade_log)
    
    if not all_actions:
        return {
            "action_mean": 0.0,
            "action_std": 0.0,
            "action_min": 0.0,
            "action_max": 0.0,
            "action_mean_abs": 0.0,
            "pct_long": 0.0,
            "pct_short": 0.0,
            "pct_flat": 0.0,
            "position_changes": 0,
            "n_trades": 0,
        }
    
    actions_arr = np.array(all_actions, dtype=np.float32)
    position_units_arr = np.array(all_position_units, dtype=np.float32)
    
    # Count position changes (consecutive position_units changes)
    position_changes = np.sum(np.abs(np.diff(position_units_arr)) > 1e-6)
    
    # Action categorization (% long/short/flat)
    n_long = np.sum(actions_arr > 1e-6)
    n_short = np.sum(actions_arr < -1e-6)
    n_flat = np.sum(np.abs(actions_arr) <= 1e-6)
    total = len(actions_arr)
    
    return {
        "action_mean": float(np.mean(actions_arr)),
        "action_std": float(np.std(actions_arr)),
        "action_min": float(np.min(actions_arr)),
        "action_max": float(np.max(actions_arr)),
        "action_mean_abs": float(np.mean(np.abs(actions_arr))),
        "pct_long": float(100.0 * n_long / total) if total > 0 else 0.0,
        "pct_short": float(100.0 * n_short / total) if total > 0 else 0.0,
        "pct_flat": float(100.0 * n_flat / total) if total > 0 else 0.0,
        "position_changes": int(position_changes),
        "n_trades": int(all_trades),
    }


def run_diagnosis(
    checkpoint: Path,
    split: str = "validation",
    n_episodes: int = 100,
    seed: int = 42,
) -> dict[str, Any]:
    """Run multi-agent diagnosis on identical validation episodes."""
    
    # Load checkpoint state and config.
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    algorithm = state.get("algorithm", "ppo")
    
    cfg_dict = state.get("config") or {}
    config = (
        ExperimentConfig.from_dict(cfg_dict)
        if isinstance(cfg_dict, dict)
        else ExperimentConfig()
    )
    
    instruments = (
        tuple(config.environment.instruments)
        if config.environment.instruments
        else DEFAULT_INSTRUMENT_ORDER
    )
    
    # Build dataset, env, encoder, window.
    dataset = make_training_dataset(DEFAULT_PROCESSED_DIR, None, instruments)
    env_config = build_env_config(config.environment)
    encoder = ObservationEncoder(
        EncoderConfig(
            context_length=config.environment.context_length,
            initial_balance=env_config.margin.initial_balance,
        )
    )
    window_config = WindowConfig(context_length=config.environment.context_length)
    
    # Build identical episode specs for all agents.
    specs = build_test_episode_specs(
        dataset,
        split=split,
        n_episodes=n_episodes,
        horizon=config.evaluation.eval_horizon,
        context_length=config.environment.context_length,
        seed=seed,
    )
    
    # Load PPO policy.
    policy, _ = load_checkpoint_policy(
        checkpoint, encoder.config.spec.encoded_shape[0], config.model
    )
    
    # Run evaluation.
    runner = EvaluationRunner(dataset, env_config, encoder, window_config)
    periods_per_year = runner.periods_per_year(split)
    
    # Evaluate all agents on shared specs.
    agent_results: dict[str, dict[str, Any]] = {}
    ppo_trajectories: list[Any] = []
    
    # PPO first
    print(f"Evaluating PPO (frozen policy)...")
    ppo_agent = PolicyAgent(policy, algorithm, name="ppo_eval")
    ppo_ev = runner.run_agent(ppo_agent, specs)
    ppo_trajs = [t for trajs in ppo_ev.trajectories_by_instrument.values() for t in trajs]
    ppo_trajectories = ppo_trajs
    
    grouped = ppo_ev.trajectories_by_instrument
    _ts, agg_log, per_instr_series = aggregate_across_instruments(grouped)
    metrics = compute_series_metrics(agg_log, periods_per_year)
    metrics["turnover"] = _pooled_turnover(ppo_trajs)
    metrics["mean_reward"] = float(
        np.mean([r for trajs in grouped.values() for t in trajs for r in t.rewards])
    )
    
    # PPO action stats
    action_stats = _ppo_action_statistics(ppo_trajs)
    
    per_instr_ppo = {}
    for instr, series in per_instr_series.items():
        instr_trajs = grouped.get(instr, [])
        instr_metrics = compute_series_metrics(series, periods_per_year)
        instr_metrics["turnover"] = _pooled_turnover(instr_trajs)
        per_instr_ppo[instr] = instr_metrics
    
    agent_results["ppo"] = {
        "metrics": {k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v 
                    for k, v in metrics.items()},
        "action_stats": action_stats,
        "per_instrument": {k: {kk: float(vv) if isinstance(vv, (int, float, np.floating, np.integer)) else vv 
                              for kk, vv in v.items()}
                           for k, v in per_instr_ppo.items()},
    }
    
    # Baselines
    for baseline_name in BASELINE_AGENTS:
        print(f"Evaluating {baseline_name}...")
        agent = make_agent(baseline_name)
        ev = runner.run_agent(agent, specs)
        trajs = [t for trajs in ev.trajectories_by_instrument.values() for t in trajs]
        
        grouped = ev.trajectories_by_instrument
        _ts, agg_log, per_instr_series = aggregate_across_instruments(grouped)
        metrics = compute_series_metrics(agg_log, periods_per_year)
        metrics["turnover"] = _pooled_turnover(trajs)
        metrics["mean_reward"] = float(
            np.mean([r for trajs in grouped.values() for t in trajs for r in t.rewards])
        )
        
        per_instr_baseline = {}
        for instr, series in per_instr_series.items():
            instr_trajs = grouped.get(instr, [])
            instr_metrics = compute_series_metrics(series, periods_per_year)
            instr_metrics["turnover"] = _pooled_turnover(instr_trajs)
            per_instr_baseline[instr] = instr_metrics
        
        agent_results[baseline_name] = {
            "metrics": {k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v 
                        for k, v in metrics.items()},
            "per_instrument": {k: {kk: float(vv) if isinstance(vv, (int, float, np.floating, np.integer)) else vv 
                                  for kk, vv in v.items()}
                               for k, v in per_instr_baseline.items()},
        }
    
    return {
        "checkpoint": str(checkpoint),
        "algorithm": algorithm,
        "env_steps": state.get("env_steps"),
        "gradient_updates": state.get("gradient_updates"),
        "split": split,
        "n_episodes": n_episodes,
        "seed": seed,
        "agents": agent_results,
    }


def print_summary(result: dict[str, Any]) -> None:
    """Print a human-readable diagnostic summary."""
    print("\n" + "=" * 80)
    print("PPO POLICY DIAGNOSIS")
    print("=" * 80)
    print(f"Checkpoint  : {result['checkpoint']}")
    print(f"Algorithm   : {result['algorithm']}")
    print(f"Env steps   : {result['env_steps']:,}")
    print(f"Gradient updates : {result['gradient_updates']:,}")
    print(f"Split       : {result['split']}")
    print(f"Episodes    : {result['n_episodes']}")
    print(f"Seed        : {result['seed']}")
    print()
    
    # Per-agent metrics table
    print("AGENT COMPARISON (validation, identical episodes)")
    print("-" * 120)
    
    agents = result["agents"]
    ppo_metrics = agents.get("ppo", {}).get("metrics", {})
    
    # Print column headers
    print(f"{'Agent':<18} {'Return':<12} {'Sharpe':<12} {'Sortino':<12} {'Max DD':<12} {'Turnover':<12} {'Trades':<10}")
    print("-" * 120)
    
    # PPO
    print(f"{'ppo':<18} {ppo_metrics.get('total_return', 0):<12.6f} "
          f"{ppo_metrics.get('sharpe', 0):<12.4f} {ppo_metrics.get('sortino', 0):<12.4f} "
          f"{ppo_metrics.get('max_drawdown_pct', 0):<12.6f} {ppo_metrics.get('turnover', 0):<12.6f} "
          f"{agents.get('ppo', {}).get('action_stats', {}).get('n_trades', 0):<10}")
    
    # Baselines
    for baseline_name in BASELINE_AGENTS:
        baseline_data = agents.get(baseline_name, {})
        baseline_metrics = baseline_data.get("metrics", {})
        print(f"{baseline_name:<18} {baseline_metrics.get('total_return', 0):<12.6f} "
              f"{baseline_metrics.get('sharpe', 0):<12.4f} {baseline_metrics.get('sortino', 0):<12.4f} "
              f"{baseline_metrics.get('max_drawdown_pct', 0):<12.6f} {baseline_metrics.get('turnover', 0):<12.6f} "
              f"{0:<10}")
    
    print()
    
    # PPO action statistics
    ppo_action_stats = agents.get("ppo", {}).get("action_stats", {})
    print("PPO ACTION STATISTICS")
    print("-" * 80)
    print(f"Mean action         : {ppo_action_stats.get('action_mean', 0):>10.6f}")
    print(f"Std action          : {ppo_action_stats.get('action_std', 0):>10.6f}")
    print(f"Min action          : {ppo_action_stats.get('action_min', 0):>10.6f}")
    print(f"Max action          : {ppo_action_stats.get('action_max', 0):>10.6f}")
    print(f"Mean abs action     : {ppo_action_stats.get('action_mean_abs', 0):>10.6f}")
    print(f"% long (> 0)        : {ppo_action_stats.get('pct_long', 0):>10.2f}%")
    print(f"% short (< 0)       : {ppo_action_stats.get('pct_short', 0):>10.2f}%")
    print(f"% flat (≈ 0)        : {ppo_action_stats.get('pct_flat', 0):>10.2f}%")
    print(f"Position changes    : {ppo_action_stats.get('position_changes', 0):>10}")
    print(f"Total trades        : {ppo_action_stats.get('n_trades', 0):>10}")
    print()


def save_results(result: dict[str, Any], out_dir: str | Path) -> dict[str, Path]:
    """Save per-agent, per-instrument, and action stats CSVs + JSON."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    # JSON summary
    json_path = out / "diagnosis.json"
    json_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    
    paths = {"json": json_path}
    
    # Per-agent summary CSV
    agent_csv = out / "agents_summary.csv"
    with agent_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "agent",
                "total_return",
                "sharpe",
                "sortino",
                "max_drawdown_pct",
                "turnover",
                "mean_reward",
            ],
        )
        writer.writeheader()
        for agent_name, agent_data in result["agents"].items():
            metrics = agent_data.get("metrics", {})
            writer.writerow({
                "agent": agent_name,
                "total_return": metrics.get("total_return", ""),
                "sharpe": metrics.get("sharpe", ""),
                "sortino": metrics.get("sortino", ""),
                "max_drawdown_pct": metrics.get("max_drawdown_pct", ""),
                "turnover": metrics.get("turnover", ""),
                "mean_reward": metrics.get("mean_reward", ""),
            })
    paths["agents"] = agent_csv
    
    # Per-instrument CSV for PPO
    if "ppo" in result["agents"]:
        instr_csv = out / "ppo_per_instrument.csv"
        with instr_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "instrument",
                    "total_return",
                    "sharpe",
                    "sortino",
                    "max_drawdown_pct",
                    "turnover",
                ],
            )
            writer.writeheader()
            for instr, metrics in result["agents"]["ppo"].get("per_instrument", {}).items():
                writer.writerow({
                    "instrument": instr,
                    "total_return": metrics.get("total_return", ""),
                    "sharpe": metrics.get("sharpe", ""),
                    "sortino": metrics.get("sortino", ""),
                    "max_drawdown_pct": metrics.get("max_drawdown_pct", ""),
                    "turnover": metrics.get("turnover", ""),
                })
        paths["ppo_per_instrument"] = instr_csv
    
    # PPO action stats CSV
    ppo_actions_csv = out / "ppo_action_stats.csv"
    with ppo_actions_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(result["agents"]["ppo"]["action_stats"].keys()))
        writer.writeheader()
        writer.writerow(result["agents"]["ppo"]["action_stats"])
    paths["ppo_action_stats"] = ppo_actions_csv
    
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose PPO policy on validation split.")
    parser.add_argument("--checkpoint", type=str, default="best.pt", help="Checkpoint path.")
    parser.add_argument("--split", type=str, default="validation", choices=["validation", "test"])
    parser.add_argument("--episodes", type=int, default=100, help="Number of validation episodes.")
    parser.add_argument("--seed", type=int, default=42, help="Episode sampling seed.")
    parser.add_argument("--out", type=str, default=None, help="Output directory (default: data/reports/ppo_diagnosis).")
    args = parser.parse_args()
    
    checkpoint = resolve_checkpoint(args.checkpoint)
    print(f"Resolved checkpoint: {checkpoint}")
    
    result = run_diagnosis(
        checkpoint,
        split=args.split,
        n_episodes=args.episodes,
        seed=args.seed,
    )
    
    print_summary(result)
    
    out_dir = Path(args.out) if args.out else Path("data/reports/ppo_diagnosis")
    paths = save_results(result, out_dir)
    
    print(f"\nResults saved to {out_dir}:")
    for name, path in paths.items():
        print(f"  {name:<20} {path}")


if __name__ == "__main__":
    main()
