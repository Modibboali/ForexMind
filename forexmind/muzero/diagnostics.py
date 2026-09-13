"""Integrated MuZero search/collection diagnostics (Stage 4.5).

The integrated loop must be able to answer five separate questions, and never
collapse them into one number:

* is the **network prior** any good (entropy, HOLD share, top-1),
* is **search** improving on that prior (KL, argmax changes, visit entropy),
* is the **reward model** fitting real rewards,
* is the **value model** agreeing with search,
* are **masks** (and therefore the economic contract) respected.

Everything here is observational: nothing in this module feeds a loss, a
reward or an action choice.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from forexmind.muzero.actions import MUZERO_ACTION_NAMES, MUZERO_NUM_ACTIONS

__all__ = [
    "RootSearchRecord",
    "action_frequency_diagnostics",
    "assert_records_are_legal",
    "search_summary",
    "staleness_summary",
]

HOLD = 0
FLAT = 1
SHORT_ACTIONS = (2, 3)
LONG_ACTIONS = (4, 5)


@dataclass(slots=True)
class RootSearchRecord:
    """One real decision: network prior, MCTS outcome, and the real reward.

    All arrays are six-wide in MuZero action order.  ``prior`` is the
    **noise-free** network prior (masked softmax); ``policy`` is the normalised
    MCTS visit distribution ``N/ΣN`` (no temperature applied), so the
    search-improvement comparison is not confounded by the sampling temperature.
    """

    prior: np.ndarray
    visits: np.ndarray
    policy: np.ndarray
    q_values: np.ndarray
    predicted_rewards: np.ndarray
    mask: np.ndarray
    action: int
    network_value: float
    search_value: float
    tree_depth: int
    real_reward: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("prior", "visits", "policy", "q_values", "predicted_rewards", "mask"):
            array = np.asarray(getattr(self, name))
            if array.shape != (MUZERO_NUM_ACTIONS,):
                raise ValueError(
                    f"{name} must have shape ({MUZERO_NUM_ACTIONS},), got {array.shape}"
                )
            setattr(self, name, array)

    @property
    def predicted_reward(self) -> float:
        """Reward the dynamics model predicts for the *selected* action."""
        return float(self.predicted_rewards[self.action])

    @property
    def prior_argmax(self) -> int:
        masked = np.where(self.mask, self.prior, -np.inf)
        return int(np.argmax(masked))

    @property
    def search_argmax(self) -> int:
        return int(np.argmax(self.visits))

    def to_dict(self) -> dict[str, Any]:
        return {
            "prior": self.prior.tolist(),
            "visits": self.visits.tolist(),
            "policy": self.policy.tolist(),
            "mask": self.mask.tolist(),
            "action": int(self.action),
            "network_value": float(self.network_value),
            "search_value": float(self.search_value),
            "tree_depth": int(self.tree_depth),
            "real_reward": None if self.real_reward is None else float(self.real_reward),
        }


def assert_records_are_legal(records: list[RootSearchRecord]) -> None:
    """Fail loudly on any mask violation (brief S22/S41)."""
    for index, record in enumerate(records):
        mask = record.mask
        if mask.shape != (MUZERO_NUM_ACTIONS,):
            raise ValueError(f"record {index}: action mask must have {MUZERO_NUM_ACTIONS} entries")
        if not mask[HOLD]:
            raise ValueError(f"record {index}: HOLD must always be valid")
        if not mask.any():
            raise ValueError(f"record {index}: at least one action must be valid")
        if not mask[record.action]:
            raise ValueError(f"record {index}: selected action {record.action} is invalid")
        invalid_visits = float(record.visits[~mask].sum())
        if invalid_visits != 0.0:
            raise ValueError(
                f"record {index}: invalid actions received {invalid_visits} visits; "
                "search must never explore illegal actions"
            )
        if float(record.policy[~mask].sum()) > 1e-9:
            raise ValueError(f"record {index}: visit policy places mass on an invalid action")


# --------------------------------------------------------------------------- #
# summaries
# --------------------------------------------------------------------------- #


def _entropy(probabilities: np.ndarray) -> float:
    p = np.asarray(probabilities, dtype=np.float64)
    p = p[p > 0.0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p)).sum())


def _kl(target: np.ndarray, reference: np.ndarray) -> float:
    """``KL(target || reference)`` with the ``0 log 0 = 0`` convention."""
    target = np.asarray(target, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    positive = target > 0.0
    if not positive.any():
        return 0.0
    safe = np.clip(reference[positive], 1e-12, None)
    return float((target[positive] * (np.log(target[positive]) - np.log(safe))).sum())


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2:
        return None
    left_std = float(left.std())
    right_std = float(right.std())
    if left_std <= 0.0 or right_std <= 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def action_frequency_diagnostics(
    actions: np.ndarray | Sequence[int], *, prefix: str = "selected"
) -> dict[str, float]:
    """Action shares per action and per family (fractions, not counts)."""
    actions = np.asarray(actions, dtype=np.int64).reshape(-1)
    total = int(actions.size)
    if total == 0:
        return {}
    shares: dict[str, float] = {}
    for index, name in enumerate(MUZERO_ACTION_NAMES):
        shares[f"{prefix}_{name.lower()}_fraction"] = float(
            np.count_nonzero(actions == index) / total
        )
    shares["group_hold_fraction"] = float(np.count_nonzero(actions == HOLD) / total)
    shares["group_flat_fraction"] = float(np.count_nonzero(actions == FLAT) / total)
    shares["group_short_fraction"] = float(
        np.count_nonzero(np.isin(actions, SHORT_ACTIONS)) / total
    )
    shares["group_long_fraction"] = float(np.count_nonzero(np.isin(actions, LONG_ACTIONS)) / total)
    return shares


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def search_summary(records: list[RootSearchRecord]) -> dict[str, float]:
    """Aggregate per-root search, prior, value, reward and mask diagnostics."""
    if not records:
        raise ValueError("search_summary requires at least one recorded root")
    assert_records_are_legal(records)

    prior_entropies: list[float] = []
    search_entropies: list[float] = []
    kls: list[float] = []
    agreements: list[float] = []
    changed: list[float] = []
    max_visit_fractions: list[float] = []
    effective_actions: list[float] = []
    depths: list[float] = []
    network_values: list[float] = []
    search_values: list[float] = []
    value_deltas: list[float] = []
    predicted_rewards: list[float] = []
    real_rewards: list[float] = []
    valid_actions: list[float] = []
    flat_masked: list[float] = []
    exposure_masked: list[float] = []

    per_action_visits: dict[int, list[float]] = {i: [] for i in range(MUZERO_NUM_ACTIONS)}
    per_action_q: dict[int, list[float]] = {i: [] for i in range(MUZERO_NUM_ACTIONS)}
    per_action_prior: dict[int, list[float]] = {i: [] for i in range(MUZERO_NUM_ACTIONS)}

    for record in records:
        prior_entropies.append(_entropy(record.prior))
        search_entropies.append(_entropy(record.policy))
        kls.append(_kl(record.policy, record.prior))
        agreements.append(1.0 if record.prior_argmax == record.search_argmax else 0.0)
        changed.append(1.0 if record.prior_argmax != record.search_argmax else 0.0)
        visits = np.asarray(record.visits, dtype=np.float64)
        total_visits = float(visits.sum())
        max_visit_fractions.append(float(visits.max() / total_visits) if total_visits else 0.0)
        effective_actions.append(float(np.exp(_entropy(record.policy))))
        depths.append(float(record.tree_depth))
        network_values.append(float(record.network_value))
        search_values.append(float(record.search_value))
        value_deltas.append(float(record.search_value - record.network_value))
        predicted_rewards.append(record.predicted_reward)
        if record.real_reward is not None:
            real_rewards.append(float(record.real_reward))
        valid_actions.append(float(record.mask.sum()))
        flat_masked.append(1.0 if not record.mask[FLAT] else 0.0)
        exposure_masked.append(
            1.0 if not record.mask[list(SHORT_ACTIONS) + list(LONG_ACTIONS)].any() else 0.0
        )
        for action in range(MUZERO_NUM_ACTIONS):
            if action != record.action:
                continue
            per_action_visits[action].append(float(visits[action]))
            per_action_q[action].append(float(record.q_values[action]))
            per_action_prior[action].append(float(record.prior[action]))

    summary: dict[str, float] = {
        "roots": float(len(records)),
        "prior_entropy_mean": _mean(prior_entropies),
        "search_entropy_mean": _mean(search_entropies),
        "mcts_network_kl_mean": _mean(kls),
        "prior_search_top1_agreement": _mean(agreements),
        "search_changed_argmax_fraction": _mean(changed),
        "root_visit_entropy_mean": _mean(search_entropies),
        "max_visit_fraction_mean": _mean(max_visit_fractions),
        "effective_actions_mean": _mean(effective_actions),
        "tree_depth_mean": _mean(depths),
        "tree_depth_median": float(np.median(depths)),
        "tree_depth_max": float(np.max(depths)),
        "network_root_value_mean": _mean(network_values),
        "mcts_root_value_mean": _mean(search_values),
        "root_value_delta_mean": _mean(value_deltas),
        "root_value_delta_median": float(np.median(value_deltas)),
        "root_value_abs_delta_mean": _mean([abs(value) for value in value_deltas]),
        "mean_valid_actions": _mean(valid_actions),
        "fraction_flat_masked": _mean(flat_masked),
        "fraction_exposure_masked": _mean(exposure_masked),
    }

    if real_rewards:
        errors = np.asarray(predicted_rewards, dtype=np.float64) - np.asarray(
            real_rewards, dtype=np.float64
        )
        summary.update(
            {
                "reward_model_mae": float(np.abs(errors).mean()),
                "reward_model_rmse": float(np.sqrt((errors**2).mean())),
                "reward_model_bias": float(errors.mean()),
                "reward_model_correlation": _correlation(
                    np.asarray(predicted_rewards, dtype=np.float64),
                    np.asarray(real_rewards, dtype=np.float64),
                )
                or 0.0,
                "predicted_reward_mean": _mean(predicted_rewards),
                "real_reward_mean": _mean(real_rewards),
            }
        )

    # "Next-state" quality: the same comparisons evaluated on the roots that
    # follow a real action, which is what a one-step dynamics check needs.
    if len(records) > 1:
        following = records[1:]
        summary["next_state_value_mae"] = _mean(
            [abs(record.search_value - record.network_value) for record in following]
        )
        summary["next_state_policy_kl"] = _mean(
            [_kl(record.policy, record.prior) for record in following]
        )

    for action, name in enumerate(MUZERO_ACTION_NAMES):
        key = name.lower()
        summary[f"selected_{key}_mean_visits"] = _mean(per_action_visits[action])
        summary[f"selected_{key}_mean_q"] = _mean(per_action_q[action])
        summary[f"selected_{key}_mean_prior"] = _mean(per_action_prior[action])
    selected = action_frequency_diagnostics([record.action for record in records])
    prior_argmax = action_frequency_diagnostics(
        [record.prior_argmax for record in records], prefix="prior_argmax"
    )
    search_argmax = action_frequency_diagnostics(
        [record.search_argmax for record in records], prefix="search_argmax"
    )
    summary.update(selected)
    # Family shares are keyed "group_*"; keep the selected-action ones and give
    # the prior/search argmax families distinct names.
    for key, value in prior_argmax.items():
        summary[key if not key.startswith("group_") else f"prior_{key}"] = value
    for key, value in search_argmax.items():
        summary[key if not key.startswith("group_") else f"search_{key}"] = value
    return summary


def staleness_summary(staleness: np.ndarray) -> dict[str, float]:
    """``current network version - trajectory network version`` statistics."""
    values = np.asarray(staleness, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {"mean_staleness": 0.0, "median_staleness": 0.0, "max_staleness": 0.0}
    return {
        "mean_staleness": float(values.mean()),
        "median_staleness": float(np.median(values)),
        "max_staleness": float(values.max()),
        "min_staleness": float(values.min()),
    }
