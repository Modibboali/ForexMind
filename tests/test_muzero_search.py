"""Stage 4.2 MuZero MCTS / PUCT search tests.

Uses a fully predictable table-driven toy model (exact rewards/values/transitions)
so search behaviour can be asserted precisely, plus a small real-network
integration test.  No environment is used for imagined transitions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from forexmind.muzero import (
    MUZERO_NUM_ACTIONS,
    MinMaxStats,
    MuZeroConfig,
    MuZeroMCTS,
    NetworkOutput,
    PlanningState,
    SearchConfig,
    apply_action_mask,
    build_muzero_network,
    discounted_backup,
    visit_count_policy,
)
from forexmind.muzero.actions import FLAT, HOLD, LONG_50, LONG_100, SHORT_100
from forexmind.muzero.node import Node

ALL_VALID = np.ones(MUZERO_NUM_ACTIONS, dtype=bool)


class ToyModel:
    """Deterministic table lookup model implementing the Stage 4.1 API."""

    def __init__(
        self,
        *,
        value: list[float] | np.ndarray,
        reward: np.ndarray,
        transitions: np.ndarray,
        logits: np.ndarray | None = None,
    ) -> None:
        self.config = MuZeroConfig(
            obs_dim=1, latent_dim=1, hidden_dim=4, num_layers=1, action_embedding_dim=2
        )
        self._value = np.asarray(value, dtype=np.float32)
        self._reward = np.asarray(reward, dtype=np.float32)
        self._transitions = np.asarray(transitions, dtype=np.int64)
        states = self._value.shape[0]
        self._logits = (
            np.zeros((states, MUZERO_NUM_ACTIONS), dtype=np.float32)
            if logits is None
            else np.asarray(logits, dtype=np.float32)
        )
        self.recurrent_calls = 0

    def initial_inference(
        self, observation: Any, action_mask: np.ndarray | None = None
    ) -> NetworkOutput:
        state = int(np.asarray(observation).reshape(-1)[0])
        return NetworkOutput(
            latent_state=torch.tensor([[float(state)]]),
            policy_logits=torch.tensor(self._logits[state]).reshape(1, -1),
            value=torch.tensor([[float(self._value[state])]]),
            reward=torch.zeros(1, 1),
        )

    def recurrent_inference(
        self,
        latent_state: Any,
        action: Any,
        action_mask: np.ndarray | None = None,
    ) -> NetworkOutput:
        latent = (
            latent_state
            if isinstance(latent_state, torch.Tensor)
            else torch.as_tensor(np.asarray(latent_state))
        )
        if latent.ndim == 1:
            latent = latent.unsqueeze(0)
        states = [int(s) for s in latent.reshape(-1).tolist()]
        actions = _broadcast_actions(action, len(states))
        next_states = [int(self._transitions[s, a]) for s, a in zip(states, actions, strict=True)]
        self.recurrent_calls += len(states)
        return NetworkOutput(
            latent_state=torch.tensor([[float(s)] for s in next_states]),
            policy_logits=torch.tensor(self._logits[next_states]),
            value=torch.tensor([[float(self._value[s])] for s in next_states]),
            reward=torch.tensor(
                [[float(self._reward[s, a])] for s, a in zip(states, actions, strict=True)]
            ),
        )


def _broadcast_actions(action: Any, batch: int) -> list[int]:
    if isinstance(action, torch.Tensor):
        if action.ndim == 0:
            return [int(action)] * batch
        return [int(a) for a in action.reshape(-1)]
    return [int(action)] * batch


def _search(model: Any, *, simulations: int = 64, mask: np.ndarray | None = None, **kwargs: Any):
    config = SearchConfig(num_simulations=simulations, seed=0, **kwargs)
    return MuZeroMCTS(model, config).search(0, ALL_VALID if mask is None else mask)


def _uniform_to_shared_state(rewards_row: list[float], transitions_row: list[int]) -> ToyModel:
    """All actions lead to one shared state whose value is zero."""
    reward = np.zeros((2, MUZERO_NUM_ACTIONS), dtype=np.float32)
    reward[0] = rewards_row
    transitions = np.zeros((2, MUZERO_NUM_ACTIONS), dtype=np.int64)
    transitions[0] = transitions_row
    transitions[1] = 1
    return ToyModel(value=[0.0, 0.0], reward=reward, transitions=transitions)


def _iter_tree(root: Node):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        if node.children is not None:
            stack.extend(child for child in node.children if child is not None)


# --------------------------------------------------------------------------- #
# Toy tree: search concentrates on the best backed-up return
# --------------------------------------------------------------------------- #


def test_search_concentrates_visits_on_the_best_immediate_return() -> None:
    rewards = [0.0, -0.1, -0.3, -0.05, 0.2, 0.05]
    model = _uniform_to_shared_state(rewards, [1] * MUZERO_NUM_ACTIONS)
    result = _search(model, simulations=64)
    assert int(np.argmax(result.visit_counts)) == LONG_50
    assert result.action == LONG_50
    assert result.visit_counts[LONG_50] == result.visit_counts.max()


def test_search_visits_sum_to_the_simulation_budget() -> None:
    model = _uniform_to_shared_state([0.0] * MUZERO_NUM_ACTIONS, [1] * MUZERO_NUM_ACTIONS)
    result = _search(model, simulations=32)
    assert result.visit_counts.sum() == 32
    assert result.diagnostics.num_simulations == 32
    assert result.diagnostics.recurrent_inference_calls == 32
    assert result.diagnostics.expanded_nodes == 33  # root + one leaf per simulation
    assert result.diagnostics.tree_depth >= 1
    assert model.recurrent_calls == 32


def test_search_never_calls_the_environment() -> None:
    model = _uniform_to_shared_state([0.0] * MUZERO_NUM_ACTIONS, [1] * MUZERO_NUM_ACTIONS)
    search = MuZeroMCTS(model, SearchConfig(num_simulations=8, seed=0))
    search.search(0, ALL_VALID)
    assert model.recurrent_calls == 8  # all imagined transitions came from the model


# --------------------------------------------------------------------------- #
# Multi-step planning: best long-term action differs from best immediate action
# --------------------------------------------------------------------------- #


def test_search_plans_beyond_the_immediate_reward() -> None:
    # SHORT_100: immediate +1 but lands on a bad state (value -10).
    # LONG_50:   immediate  0 but lands on a good state (value +5).
    reward = np.zeros((4, MUZERO_NUM_ACTIONS), dtype=np.float32)
    reward[0] = [-5.0, -5.0, 1.0, -5.0, 0.0, -5.0]
    transitions = np.zeros((4, MUZERO_NUM_ACTIONS), dtype=np.int64)
    transitions[0] = [3, 3, 1, 3, 2, 3]
    for state in range(1, 4):
        transitions[state] = state  # absorbing
    model = ToyModel(value=[0.0, -10.0, 5.0, -10.0], reward=reward, transitions=transitions)

    greedy_reward = int(np.argmax(reward[0]))
    assert greedy_reward == SHORT_100  # the greedy choice is the wrong one

    result = _search(model, simulations=64)
    assert result.action == LONG_50
    assert int(np.argmax(result.visit_counts)) == LONG_50
    assert result.visit_counts[LONG_50] > result.visit_counts[SHORT_100]
    # The backed-up Q reflects reward + discounted future value, not the reward alone.
    q = result.diagnostics.q_values
    assert q[LONG_50] > q[SHORT_100]
    assert 4.5 < q[LONG_50] <= 5.0
    assert -10.0 < q[SHORT_100] < -8.0


# --------------------------------------------------------------------------- #
# HOLD / FLAT are ordinary actions
# --------------------------------------------------------------------------- #


def _one_best_action(best: int) -> tuple[ToyModel, np.ndarray]:
    reward = np.zeros((3, MUZERO_NUM_ACTIONS), dtype=np.float32)
    transitions = np.zeros((3, MUZERO_NUM_ACTIONS), dtype=np.int64)
    for action in range(MUZERO_NUM_ACTIONS):
        transitions[0, action] = 1 if action == best else 2
    transitions[1] = 1  # absorbing good state
    transitions[2] = 2  # absorbing bad state
    return ToyModel(value=[0.0, 10.0, -10.0], reward=reward, transitions=transitions), ALL_VALID


def test_search_selects_hold_when_hold_is_best() -> None:
    model, mask = _one_best_action(HOLD)
    result = _search(model, simulations=48, mask=mask)
    assert result.action == HOLD
    assert int(np.argmax(result.visit_counts)) == HOLD


def test_search_selects_flat_when_flat_is_best() -> None:
    model, mask = _one_best_action(FLAT)
    result = _search(model, simulations=48, mask=mask)
    assert result.action == FLAT
    assert int(np.argmax(result.visit_counts)) == FLAT


def test_search_selects_long_50_when_long_50_is_best() -> None:
    model, mask = _one_best_action(LONG_50)
    result = _search(model, simulations=48, mask=mask)
    assert result.action == LONG_50
    assert int(np.argmax(result.visit_counts)) == LONG_50


def test_search_does_not_suppress_hold_when_hold_is_worst() -> None:
    reward = np.zeros((3, MUZERO_NUM_ACTIONS), dtype=np.float32)
    transitions = np.zeros((3, MUZERO_NUM_ACTIONS), dtype=np.int64)
    for action in range(MUZERO_NUM_ACTIONS):
        transitions[0, action] = 2 if action == HOLD else 1
    transitions[1] = 1
    transitions[2] = 2
    model = ToyModel(value=[0.0, 10.0, -10.0], reward=reward, transitions=transitions)
    result = _search(model, simulations=48)
    assert result.action != HOLD
    assert result.visit_counts[HOLD] < result.visit_counts.max()


def test_search_has_no_action_specific_special_casing() -> None:
    import forexmind.muzero.search as search_module

    for name in ("HOLD", "FLAT", "SHORT_100", "SHORT_50", "LONG_50", "LONG_100", "HOLD_PENALTY"):
        assert not hasattr(search_module, name), f"search must not special-case {name}"


def test_hold_prior_is_not_penalised() -> None:
    model = _uniform_to_shared_state([0.0] * MUZERO_NUM_ACTIONS, [1] * MUZERO_NUM_ACTIONS)
    result = _search(model, simulations=1)
    assert result.root_priors[HOLD] == pytest.approx(1.0 / MUZERO_NUM_ACTIONS)


# --------------------------------------------------------------------------- #
# Action masks
# --------------------------------------------------------------------------- #


def test_invalid_root_action_gets_zero_prior_and_zero_visits() -> None:
    model = _uniform_to_shared_state([0.0, 999.0, 0.0, 0.0, 0.0, 0.0], [1] * MUZERO_NUM_ACTIONS)
    mask = ALL_VALID.copy()
    mask[FLAT] = False
    result = _search(model, simulations=32, mask=mask)
    assert result.root_priors[FLAT] == 0.0
    assert result.visit_counts[FLAT] == 0
    assert result.action != FLAT
    assert result.policy[FLAT] == 0.0


def test_invalid_action_is_impossible_to_select_even_if_best() -> None:
    # FLAT would be by far the best action, but it is masked out.
    model = _uniform_to_shared_state([0.0, 999.0, 0.0, 0.0, 0.0, 0.0], [1] * MUZERO_NUM_ACTIONS)
    mask = ALL_VALID.copy()
    mask[FLAT] = False
    result = _search(model, simulations=48, mask=mask)
    assert result.visit_counts[FLAT] == 0
    assert result.action != FLAT


def test_flat_account_root_mask_matches_the_planning_state() -> None:
    model = _uniform_to_shared_state([0.0] * MUZERO_NUM_ACTIONS, [1] * MUZERO_NUM_ACTIONS)
    state = PlanningState.flat()
    result = _search(model, simulations=16, mask=state.action_mask())
    assert not result.root_action_mask[FLAT]
    assert result.visit_counts[FLAT] == 0
    assert result.root_action_mask[HOLD]


def test_imagined_nodes_inherit_the_deterministic_planning_mask() -> None:
    model = _uniform_to_shared_state([0.0] * MUZERO_NUM_ACTIONS, [1] * MUZERO_NUM_ACTIONS)
    search = MuZeroMCTS(model, SearchConfig(num_simulations=48, seed=0))
    search.search(0, PlanningState.flat().action_mask())
    root = search.last_root
    assert root is not None
    checked = 0
    for node in _iter_tree(root):
        if node.planning_state is None or not node.is_expanded():
            continue
        assert node.children is not None
        expected = node.planning_state.action_mask()
        for action in range(MUZERO_NUM_ACTIONS):
            assert (node.children[action] is not None) == bool(expected[action])
        assert node.children[FLAT] is None or not node.planning_state.is_flat
        checked += 1
    assert checked > 2


def test_hold_remains_valid_in_every_planning_state() -> None:
    for exposure in (0.0, -1.0, -0.5, 0.0, 0.5, 1.0):
        assert PlanningState.from_exposure(exposure).action_mask()[HOLD]


def test_planning_state_mismatch_is_rejected() -> None:
    model = _uniform_to_shared_state([0.0] * MUZERO_NUM_ACTIONS, [1] * MUZERO_NUM_ACTIONS)
    search = MuZeroMCTS(model, SearchConfig(num_simulations=4, seed=0))
    with pytest.raises(ValueError, match="planning_state"):
        search.search(0, ALL_VALID, planning_state=PlanningState.flat())


# --------------------------------------------------------------------------- #
# Root Dirichlet noise
# --------------------------------------------------------------------------- #


def test_root_noise_is_reproducible_for_a_fixed_seed() -> None:
    model = _uniform_to_shared_state([0.0] * MUZERO_NUM_ACTIONS, [1] * MUZERO_NUM_ACTIONS)
    first = _search(model, simulations=16, add_root_noise=True)
    second = _search(model, simulations=16, add_root_noise=True)
    assert np.allclose(first.root_priors, second.root_priors)
    assert np.array_equal(first.visit_counts, second.visit_counts)


def test_training_search_adds_noise_and_evaluation_does_not() -> None:
    model = _uniform_to_shared_state([0.0] * MUZERO_NUM_ACTIONS, [1] * MUZERO_NUM_ACTIONS)
    plain = _search(model, simulations=16, add_root_noise=False)
    noisy = _search(model, simulations=16, add_root_noise=True)
    uniform = np.full(MUZERO_NUM_ACTIONS, 1.0 / MUZERO_NUM_ACTIONS)
    assert np.allclose(plain.root_priors, uniform)
    assert not np.allclose(noisy.root_priors, uniform)
    assert noisy.root_priors.sum() == pytest.approx(1.0)


def test_noise_is_restricted_to_valid_actions() -> None:
    model = _uniform_to_shared_state([0.0] * MUZERO_NUM_ACTIONS, [1] * MUZERO_NUM_ACTIONS)
    mask = ALL_VALID.copy()
    mask[FLAT] = False
    result = _search(model, simulations=16, mask=mask, add_root_noise=True)
    assert result.root_priors[FLAT] == 0.0
    assert result.root_priors[mask].sum() == pytest.approx(1.0)


def test_config_training_and_evaluation_helpers() -> None:
    base = SearchConfig(num_simulations=16)
    assert base.add_root_noise is False
    assert base.training().add_root_noise is True
    assert base.evaluation().add_root_noise is False
    assert base.evaluation().temperature == 0.0


def test_search_config_validates_hyperparameters() -> None:
    with pytest.raises(ValueError):
        SearchConfig(num_simulations=0)
    with pytest.raises(ValueError):
        SearchConfig(discount=0.0)
    with pytest.raises(ValueError):
        SearchConfig(pb_c_base=0.0)
    with pytest.raises(ValueError):
        SearchConfig(root_dirichlet_alpha=0.0)
    with pytest.raises(ValueError):
        SearchConfig(root_exploration_fraction=1.5)
    with pytest.raises(ValueError):
        SearchConfig(temperature=-1.0)


# --------------------------------------------------------------------------- #
# Backup mathematics
# --------------------------------------------------------------------------- #


def test_backup_matches_hand_computed_discounted_returns() -> None:
    # G2 = 3 ; G1 = 2 + 0.9 * 3 ; G0 = 1 + 0.9 * G1
    values = discounted_backup([1.0, 2.0], 3.0, 0.9)
    assert values == pytest.approx([1.0 + 0.9 * (2.0 + 0.9 * 3.0), 2.0 + 0.9 * 3.0, 3.0])
    assert values[0] == pytest.approx(5.23)
    assert values[1] == pytest.approx(4.7)


def test_backup_uses_a_single_sign_convention() -> None:
    """Single-agent backup: rewards are added, never sign-flipped."""
    positive = discounted_backup([1.0], 0.0, 0.9)
    negative = discounted_backup([-1.0], 0.0, 0.9)
    assert positive[0] == pytest.approx(1.0)
    assert negative[0] == pytest.approx(-1.0)
    assert positive[0] > negative[0]
    # Two-step with mixed signs stays a plain discounted sum.
    assert discounted_backup([1.0, -2.0], 3.0, 0.5)[0] == pytest.approx(
        1.0 + 0.5 * (-2.0 + 0.5 * 3.0)
    )


def test_backup_with_no_transitions_returns_the_leaf_value() -> None:
    assert discounted_backup([], 2.5, 0.9) == [pytest.approx(2.5)]


def test_backup_validates_discount() -> None:
    with pytest.raises(ValueError):
        discounted_backup([1.0], 1.0, 1.5)


# --------------------------------------------------------------------------- #
# Visit-count policy
# --------------------------------------------------------------------------- #


def test_visit_policy_is_argmax_at_zero_temperature() -> None:
    counts = np.array([1.0, 3.0, 0.0, 2.0, 0.0, 0.0])
    policy = visit_count_policy(counts, 0.0)
    assert policy.sum() == pytest.approx(1.0)
    assert policy[1] == pytest.approx(1.0)


def test_visit_policy_respects_temperature() -> None:
    counts = np.array([1.0, 3.0, 0.0, 0.0, 0.0, 0.0])
    policy = visit_count_policy(counts, 1.0)
    assert policy.sum() == pytest.approx(1.0)
    assert policy[1] == pytest.approx(0.75)
    assert policy[0] == pytest.approx(0.25)
    assert policy[2:].sum() == pytest.approx(0.0)


def test_visit_policy_rejects_zero_visits() -> None:
    with pytest.raises(ValueError):
        visit_count_policy(np.zeros(MUZERO_NUM_ACTIONS), 0.0)


# --------------------------------------------------------------------------- #
# MinMaxStats
# --------------------------------------------------------------------------- #


def test_minmax_uninitialized_normalizes_to_zero() -> None:
    stats = MinMaxStats()
    assert stats.normalize(123.0) == 0.0
    assert stats.initialized is False


def test_minmax_single_value_has_zero_spread() -> None:
    stats = MinMaxStats()
    stats.update(5.0)
    assert stats.normalize(5.0) == 0.0
    assert np.isfinite(stats.normalize(1e9))


def test_minmax_equal_bounds_are_safe() -> None:
    stats = MinMaxStats()
    for _ in range(4):
        stats.update(2.0)
    assert stats.minimum == stats.maximum == 2.0
    assert stats.normalize(2.0) == 0.0


def test_minmax_positive_negative_and_mixed_values() -> None:
    positive = MinMaxStats()
    positive.update(1.0)
    positive.update(3.0)
    assert positive.normalize(1.0) == pytest.approx(0.0)
    assert positive.normalize(3.0) == pytest.approx(1.0)
    assert positive.normalize(2.0) == pytest.approx(0.5)

    negative = MinMaxStats()
    negative.update(-5.0)
    negative.update(-1.0)
    assert negative.normalize(-5.0) == pytest.approx(0.0)
    assert negative.normalize(-3.0) == pytest.approx(0.5)
    assert negative.normalize(-1.0) == pytest.approx(1.0)

    mixed = MinMaxStats()
    for value in (-2.0, 0.0, 4.0):
        mixed.update(value)
    assert all(np.isfinite(mixed.normalize(v)) for v in (-10.0, -2.0, 1.0, 4.0, 99.0))
    assert mixed.normalize(-2.0) == pytest.approx(0.0)
    assert mixed.normalize(4.0) == pytest.approx(1.0)
    assert mixed.normalize(1.0) == pytest.approx(0.5)


def test_minmax_reset_clears_state() -> None:
    stats = MinMaxStats()
    stats.update(3.0)
    stats.reset()
    assert stats.minimum is None and stats.maximum is None and stats.count == 0
    assert stats.normalize(3.0) == 0.0


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #


def test_search_is_reproducible_with_the_same_seed() -> None:
    model_a = _uniform_to_shared_state([0.0, -0.1, 0.3, -0.2, 0.1, 0.05], [1] * MUZERO_NUM_ACTIONS)
    model_b = _uniform_to_shared_state([0.0, -0.1, 0.3, -0.2, 0.1, 0.05], [1] * MUZERO_NUM_ACTIONS)
    first = _search(model_a, simulations=32)
    second = _search(model_b, simulations=32)
    assert np.array_equal(first.visit_counts, second.visit_counts)
    assert np.allclose(first.policy, second.policy)
    assert first.action == second.action


def test_repeated_searches_on_one_instance_are_deterministic_in_evaluation() -> None:
    model = _uniform_to_shared_state([0.0, -0.1, 0.3, -0.2, 0.1, 0.05], [1] * MUZERO_NUM_ACTIONS)
    search = MuZeroMCTS(model, SearchConfig(num_simulations=32, seed=0).evaluation())
    first = search.search(0, ALL_VALID)
    second = search.search(0, ALL_VALID)
    assert np.array_equal(first.visit_counts, second.visit_counts)
    assert first.action == second.action


def test_diagnostics_expose_root_level_debug_information() -> None:
    model = _uniform_to_shared_state([0.0, 0.4, 0.1, 0.2, 0.3, 0.05], [1] * MUZERO_NUM_ACTIONS)
    result = _search(model, simulations=32)
    assert result.diagnostics.q_values.shape == (MUZERO_NUM_ACTIONS,)
    assert result.diagnostics.puct_scores.shape == (MUZERO_NUM_ACTIONS,)
    assert result.diagnostics.predicted_rewards.shape == (MUZERO_NUM_ACTIONS,)
    assert np.isfinite(result.diagnostics.q_values).all()
    payload = result.to_dict()
    assert set(payload) >= {
        "action",
        "visit_counts",
        "policy",
        "root_value",
        "root_priors",
        "root_action_mask",
        "diagnostics",
    }
    assert len(payload["policy"]) == MUZERO_NUM_ACTIONS


def test_complexity_budgets_scale_with_simulations() -> None:
    model = _uniform_to_shared_state([0.0] * MUZERO_NUM_ACTIONS, [1] * MUZERO_NUM_ACTIONS)
    calls = {}
    for simulations in (16, 32, 64):
        result = _search(model, simulations=simulations)
        calls[simulations] = result.diagnostics.recurrent_inference_calls
        assert result.visit_counts.sum() == simulations
        assert result.diagnostics.expanded_nodes == simulations + 1
    assert calls == {16: 16, 32: 32, 64: 64}


# --------------------------------------------------------------------------- #
# Real-network integration
# --------------------------------------------------------------------------- #


def _small_network() -> Any:
    torch.manual_seed(0)
    config = MuZeroConfig(
        obs_dim=16,
        latent_dim=8,
        hidden_dim=16,
        num_layers=1,
        action_embedding_dim=4,
        value_support_size=11,
        reward_support_size=11,
    )
    model = build_muzero_network(config)
    model.eval()
    return model


def test_search_runs_against_the_real_network() -> None:
    model = _small_network()
    observation = torch.randn(16)
    search = MuZeroMCTS(model, SearchConfig(num_simulations=16, seed=0).evaluation())
    result = search.search(observation, ALL_VALID)
    assert result.policy.shape == (MUZERO_NUM_ACTIONS,)
    assert result.policy.sum() == pytest.approx(1.0)
    assert result.visit_counts.sum() == 16
    assert result.root_action_mask[result.action]
    assert np.isfinite(result.root_value)


def test_search_against_the_real_network_is_reproducible() -> None:
    model = _small_network()
    observation = torch.randn(16)
    search = MuZeroMCTS(model, SearchConfig(num_simulations=16, seed=0).evaluation())
    first = search.search(observation, ALL_VALID)
    second = search.search(observation, ALL_VALID)
    assert np.array_equal(first.visit_counts, second.visit_counts)
    assert first.action == second.action


def test_search_respects_a_masked_root_action_on_the_real_network() -> None:
    model = _small_network()
    observation = torch.randn(16)
    mask = ALL_VALID.copy()
    mask[FLAT] = False
    search = MuZeroMCTS(model, SearchConfig(num_simulations=16, seed=0).evaluation())
    result = search.search(observation, mask)
    assert result.visit_counts[FLAT] == 0
    assert result.root_priors[FLAT] == 0.0
    assert result.action != FLAT


def test_real_network_applies_action_masks_in_its_inference_layer() -> None:
    """The root latent history carries planning masks, not network-invented ones."""
    logits = torch.randn(1, MUZERO_NUM_ACTIONS)
    mask = PlanningState.flat().action_mask()
    masked = apply_action_mask(logits, mask)
    probs = torch.softmax(masked, dim=-1)
    assert probs[0, FLAT].item() == pytest.approx(0.0)
    assert probs.sum().item() == pytest.approx(1.0)


def test_mcts_rejects_a_model_with_the_wrong_action_space() -> None:
    torch.manual_seed(0)
    ten_action = build_muzero_network(MuZeroConfig(obs_dim=8, num_actions=10, latent_dim=4))
    with pytest.raises(ValueError, match="num_actions=6"):
        MuZeroMCTS(ten_action, SearchConfig(num_simulations=4))


def test_search_from_env_uses_the_environment_mask_and_state() -> None:
    from dataclasses import replace as dc_replace

    from forexmind.config import PositionSizingConfig
    from forexmind.environment import ForexEnvironment
    from forexmind.muzero.actions import env_action_index

    from tests.test_environment import _config, _dataset

    config = dc_replace(
        _config(close_at_episode_end=False), sizing=PositionSizingConfig(mode="equity_fraction")
    )
    env = ForexEnvironment(_dataset(instrument="EURUSD", price=1.1), config)
    env.reset(start_index=0, horizon=8)
    env.step(env_action_index(LONG_100))

    torch.manual_seed(0)
    model = build_muzero_network(
        MuZeroConfig(obs_dim=351, latent_dim=8, hidden_dim=16, num_layers=1, action_embedding_dim=4)
    )
    model.eval()
    search = MuZeroMCTS(model, SearchConfig(num_simulations=8, seed=0).evaluation())
    result = search.search_from_env(env, torch.randn(351))
    assert not result.root_action_mask[LONG_100]  # redundant at +100% exposure
    assert result.visit_counts[LONG_100] == 0
    assert result.policy.sum() == pytest.approx(1.0)
