"""Stage 3.4: bounded categorical PPO experiment and fixed validation report.

Run: python -m tools.validate_categorical_ppo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import numpy as np
from forexmind.training.benchmark import load_checkpoint_policy
from forexmind.training.config import ExperimentConfig
from forexmind.training.evaluator import PolicyEvaluator
from forexmind.training.ppo import PPOTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs/stage34_categorical_seed42")
    run = Path(parser.parse_args().run_dir)
    if (run / "checkpoints/final.pt").exists():
        raise RuntimeError("Validation run already exists; preserve its artifacts.")
    run.mkdir(parents=True, exist_ok=True)
    cfg = ExperimentConfig.from_yaml("configs/ppo_stable.yaml")
    cfg.training.total_env_steps = 100352
    cfg.training.collect_batch = 2048
    cfg.training.ppo_target_kl = 0.02
    cfg.training.finite_check = True
    cfg.compute.num_workers = 2
    cfg.compute.torch_threads = 1
    cfg.compute.torch_interop_threads = 1
    cfg.compute.collect_backend = "process"
    cfg.compute.dataset_backend = "mmap"
    cfg.logging.log_every_env_steps = 10240
    cfg.logging.evaluate_every_env_steps = 50176
    cfg.logging.checkpoint_every_env_steps = 50176
    cfg.evaluation.validation_episodes = 28
    cfg.evaluation.eval_seed = 34042
    cfg.run_id = "stage34_categorical"
    cfg.to_yaml(run / "config.yaml")
    trainer = PPOTrainer(cfg, run)
    summary = trainer.train()
    policy, algorithm = load_checkpoint_policy(
        run / "checkpoints/final.pt", trainer.obs_dim, cfg.model
    )
    evaluator = PolicyEvaluator(
        trainer.dataset,
        trainer.env_config,
        trainer.encoder,
        trainer.window_config,
        eval_horizon=512,
        eval_seed=34042,
        context_length=64,
    )
    validation = evaluator.evaluate(policy, algorithm, "validation", 28)
    updates = trainer.update_history
    checks = {
        "all_updates_recorded": len(updates) == trainer.gradient_updates,
        "steps_reached": trainer.env_steps == cfg.training.total_env_steps,
        "episode_lengths_valid": all(
            n == cfg.environment.horizon for n in trainer._episode_lengths
        ),
        "no_unexpected_forced_executions": summary["action_diagnostics"]["forced_executions"] == 0,
        "stable_kl": max(d["approx_kl_log"] for d in updates) < 0.05,
        "reasonable_clipping": max(d["clip_fraction"] for d in updates) < 0.5,
        "two_workers_produced": summary["workers_producing_transitions"] == 2,
        "all_update_metrics_finite": all(np.isfinite(list(d.values())).all() for d in updates),
        "no_nonfinite_tensors": trainer._nonfinite_total == 0,
        "unchanged_policy_ratio": max(d["initial_ratio_max_error"] for d in updates) < 1e-5,
        "invalid_actions_never_sampled": all(d["invalid_actions_sampled"] == 0 for d in updates),
        "hold_never_executes": summary["action_diagnostics"]["hold_executions"] == 0,
        "all_actions_reached": all(
            summary["action_diagnostics"][f"count_{a}"] > 0
            for a in (
                "HOLD",
                "FLAT",
                "SHORT_100",
                "SHORT_75",
                "SHORT_50",
                "SHORT_25",
                "LONG_25",
                "LONG_50",
                "LONG_75",
                "LONG_100",
            )
        ),
        "finite_validation_reward": bool(
            np.isfinite(cast(float, validation.metrics["mean_reward"]))
        ),
    }
    stability = {
        k: {
            "min": min(d[k] for d in updates),
            "mean": float(np.mean([d[k] for d in updates])),
            "max": max(d[k] for d in updates),
            "last": updates[-1][k],
        }
        for k in (
            "entropy",
            "approx_kl_log",
            "clip_fraction",
            "actor_loss",
            "critic_loss",
            "initial_ratio_max_error",
        )
    }
    report = {
        "training": summary,
        "checks": checks,
        "stability": stability,
        "validation": validation.metrics,
        "per_instrument": validation.per_instrument,
        "validation_specs": [
            t.spec.to_dict() for ts in validation.trajectories.values() for t in ts
        ],
    }
    (run / "validation_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str), flush=True)
    if not all(checks.values()):
        raise RuntimeError("Categorical validation correctness check failed; see report")


if __name__ == "__main__":
    main()
