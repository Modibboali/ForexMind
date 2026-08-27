"""ForexMind Phase 3 — Model-free RL training infrastructure.

Public entry points:

* ``SACTrainer`` / ``PPOTrainer`` — model-free RL trainers.
* ``ExperimentConfig`` — the single serializable experiment configuration.
* CLI launchers: ``python -m forexmind.training.train_sac`` and
  ``python -m forexmind.training.train_ppo``.
"""

from forexmind.training.config import ExperimentConfig
from forexmind.training.ppo import PPOTrainer
from forexmind.training.sac import SACTrainer

__all__ = ["ExperimentConfig", "PPOTrainer", "SACTrainer"]
