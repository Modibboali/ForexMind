"""MuZero MCTS / PUCT search (Stage 4.2).

Given one real Forex observation, search builds a latent tree with
``initial_inference`` and ``recurrent_inference`` and returns an improved root
policy over MuZero's six actions::

    real observation
          ↓ initial_inference
    root latent state
          ↓ expand (real env action mask)
    repeated simulations
          ↓ selection      (PUCT over children, invalid actions have no child)
          ↓ evaluation     (recurrent_inference -> reward, value, next latent)
          ↓ expansion      (priors from masked logits; lazy children)
          ↓ backup         (discounted G along the visited path)
    root visit distribution

The environment is never consulted for imagined transitions: this module has no
reference to ``env.step()``.  Only the root mask/planning state come from the
environment, and they are passed in by the caller.

Three things are worth calling out explicitly:

1. **Single-agent backup.**  ForexMind is a single-agent environment, so the
   discounted return is accumulated straight up the path with no sign flipping
   and no player alternation: ``G_k = r_{k+1} + gamma * G_{k+1}``.
2. **Reward-inclusive Q.**  The reward of an edge is stored on the *child* node,
   so the PUCT value term is ``child.reward + discount * child.value()``, i.e.
   the discounted return of taking that action.  Evaluated only once the child
   has been visited.
3. **Deterministic planning masks.**  Imagined action masks come from
   :class:`forexmind.muzero.actions.PlanningState`, not from the network.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from forexmind.muzero.actions import MUZERO_NUM_ACTIONS, PlanningState, project_action_mask
from forexmind.muzero.config import SearchConfig
from forexmind.muzero.inference import apply_action_mask
from forexmind.muzero.minmax import MinMaxStats
from forexmind.muzero.node import Node

__all__ = [
    "MuZeroMCTS",
    "SearchConfig",
    "SearchDiagnostics",
    "SearchResult",
    "discounted_backup",
    "visit_count_policy",
]

_TEMPERATURE_EPS = 1e-8


def discounted_backup(rewards: Sequence[float], leaf_value: float, discount: float) -> list[float]:
    """Discounted returns ``G`` for every node on a search path.

    ``rewards[k]`` is the reward on the edge entering path node ``k + 1`` (the
    root has no incoming reward), and ``leaf_value`` is the predicted value of
    the last node.  Returns ``[G_0, ..., G_K]`` with::

        G_K     = leaf_value
        G_k     = rewards[k] + discount * G_{k+1}

    Single-agent semantics: rewards are *added*, never sign-flipped.
    """
    if not 0.0 <= discount <= 1.0:
        raise ValueError(f"discount must be in [0, 1], got {discount}")
    values = [0.0] * (len(rewards) + 1)
    value = float(leaf_value)
    values[-1] = value
    for index in range(len(rewards) - 1, -1, -1):
        value = float(rewards[index]) + discount * value
        values[index] = value
    return values


def visit_count_policy(counts: np.ndarray, temperature: float) -> np.ndarray:
    """``pi(a) ∝ N(a) ** (1 / temperature)``; argmax when temperature ~ 0."""
    counts = np.maximum(np.asarray(counts, dtype=np.float64), 0.0)
    total = float(counts.sum())
    if total <= 0.0:
        raise ValueError("visit counts sum to zero; run at least one simulation")
    if temperature <= _TEMPERATURE_EPS:
        policy = np.zeros_like(counts)
        policy[int(np.argmax(counts))] = 1.0
        return policy
    powered = np.power(counts, 1.0 / temperature)
    powered_sum = float(powered.sum())
    if powered_sum <= 0.0:
        raise ValueError("visit-count policy is degenerate")
    return powered / powered_sum


@dataclass(frozen=True, slots=True)
class SearchDiagnostics:
    """Per-search counters and root-level debug information."""

    num_simulations: int
    tree_depth: int
    expanded_nodes: int
    recurrent_inference_calls: int
    root_predicted_value: float
    q_values: np.ndarray
    puct_scores: np.ndarray
    predicted_rewards: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_simulations": self.num_simulations,
            "tree_depth": self.tree_depth,
            "expanded_nodes": self.expanded_nodes,
            "recurrent_inference_calls": self.recurrent_inference_calls,
            "root_predicted_value": self.root_predicted_value,
            "q_values": self.q_values.tolist(),
            "puct_scores": self.puct_scores.tolist(),
            "predicted_rewards": self.predicted_rewards.tolist(),
        }


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Improved root policy plus everything a later replay stage will need.

    ``visit_counts``, ``policy``, ``root_priors`` and ``root_action_mask`` are
    all six-wide.  Together with the observation the caller already holds, this
    is the information a future replay example requires (the real environment
    reward is only known after the environment is stepped).
    """

    action: int
    visit_counts: np.ndarray
    policy: np.ndarray
    root_value: float
    root_priors: np.ndarray
    root_action_mask: np.ndarray
    diagnostics: SearchDiagnostics

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": int(self.action),
            "visit_counts": self.visit_counts.tolist(),
            "policy": self.policy.tolist(),
            "root_value": float(self.root_value),
            "root_priors": self.root_priors.tolist(),
            "root_action_mask": self.root_action_mask.tolist(),
            "diagnostics": self.diagnostics.to_dict(),
        }


@dataclass(slots=True)
class _SearchStats:
    expanded_nodes: int = 0
    recurrent_inference_calls: int = 0
    tree_depth: int = 0


class MuZeroMCTS:
    """MuZero PUCT search over a learned latent dynamics model.

    ``model`` must expose the Stage 4.1 inference contract
    (``initial_inference`` / ``recurrent_inference`` returning ``NetworkOutput``)
    and a ``config`` with ``num_actions == 6``.
    """

    def __init__(
        self,
        model: Any,
        config: SearchConfig | None = None,
        *,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.model = model
        self.config = config or SearchConfig()
        self.rng = rng if rng is not None else np.random.default_rng(self.config.seed)
        num_actions = int(getattr(getattr(model, "config", None), "num_actions", 0))
        if num_actions != MUZERO_NUM_ACTIONS:
            raise ValueError(
                f"MuZeroMCTS requires a model with num_actions={MUZERO_NUM_ACTIONS}, "
                f"got {num_actions}"
            )
        self.num_actions = num_actions
        #: Root node of the most recent search, for debugging/inspection only.
        #: Overwritten on every call; never required by the algorithm.
        self.last_root: Node | None = None

    # -- public API -----------------------------------------------------------

    def search(
        self,
        observation: Any,
        action_mask: np.ndarray,
        *,
        planning_state: PlanningState | None = None,
        add_root_noise: bool | None = None,
    ) -> SearchResult:
        """Search from one real observation and return the improved root policy.

        Args:
            observation: Flat observation accepted by ``initial_inference``.
            action_mask: The real environment mask, ten-wide (projected onto
                MuZero's six actions) or already six-wide.
            planning_state: Optional explicit planning state.  When omitted it is
                reconstructed from ``action_mask`` (faithful for every state
                reachable by MuZero actions).
            add_root_noise: Overrides ``SearchConfig.add_root_noise``.  Leave
                ``False``/``None`` for deterministic validation/evaluation.
        """
        mask = project_action_mask(action_mask)
        self._validate_root_mask(mask)
        state = (
            planning_state if planning_state is not None else PlanningState.from_action_mask(mask)
        )
        if not np.array_equal(state.action_mask(), mask):
            raise ValueError("planning_state does not reproduce the supplied action mask")
        use_noise = self.config.add_root_noise if add_root_noise is None else bool(add_root_noise)

        stats = _SearchStats()
        minmax = MinMaxStats()

        with torch.no_grad():
            root_output = self.model.initial_inference(observation, mask)
        root = Node(prior=1.0, reward=0.0, planning_state=state)
        root.latent_state = root_output.latent_state[0]
        self._expand(root, root_output.policy_logits[0], mask, stats)
        if use_noise:
            self._apply_root_noise(root)

        for _ in range(self.config.num_simulations):
            self._simulate(root, minmax, stats)

        self.last_root = root
        assert root.children is not None
        counts = np.array(
            [child.visit_count if child is not None else 0 for child in root.children],
            dtype=np.float64,
        )
        policy = visit_count_policy(counts, self.config.temperature)
        action = self._select_action(counts, policy)
        priors = np.array(
            [child.prior if child is not None else 0.0 for child in root.children],
            dtype=np.float64,
        )
        diagnostics = self._diagnostics(root, minmax, stats, root_output)
        return SearchResult(
            action=action,
            visit_counts=counts,
            policy=policy,
            root_value=root.value(),
            root_priors=priors,
            root_action_mask=mask.copy(),
            diagnostics=diagnostics,
        )

    def search_from_env(
        self,
        env: Any,
        observation: Any,
        *,
        add_root_noise: bool | None = None,
    ) -> SearchResult:
        """Search using the live environment's mask and planning state."""
        mask = project_action_mask(np.asarray(env.action_masks(), dtype=bool))
        return self.search(
            observation,
            mask,
            planning_state=PlanningState.from_env(env),
            add_root_noise=add_root_noise,
        )

    # -- simulation -----------------------------------------------------------

    def _simulate(self, root: Node, minmax: MinMaxStats, stats: _SearchStats) -> None:
        node = root
        path: list[Node] = [root]
        actions: list[int] = []
        while node.is_expanded():
            action, child = self._select_child(node, minmax)
            actions.append(action)
            node = child
            path.append(node)

        parent = path[-2]
        action = actions[-1]
        assert parent.latent_state is not None
        output = self._recurrent_inference([parent.latent_state], [action], stats)

        leaf = path[-1]
        leaf.latent_state = output.latent_state[0]
        leaf.reward = float(output.reward.reshape(-1)[0])
        assert parent.planning_state is not None
        leaf.planning_state = parent.planning_state.after(action)
        self._expand(leaf, output.policy_logits[0], leaf.planning_state.action_mask(), stats)
        self._backup(path, float(output.value.reshape(-1)[0]), minmax)

        stats.tree_depth = max(stats.tree_depth, len(path) - 1)

    def _select_child(self, node: Node, minmax: MinMaxStats) -> tuple[int, Node]:
        assert node.children is not None
        best_action = -1
        best_child: Node | None = None
        best_score = -math.inf
        for action in node.valid_actions():
            child = node.children[action]
            if child is None:  # pragma: no cover - guarded by valid_actions
                continue
            score = self._puct_score(node, child, minmax)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        if best_child is None:
            raise ValueError("node has no selectable child")
        return best_action, best_child

    def _puct_score(self, parent: Node, child: Node, minmax: MinMaxStats) -> float:
        """Standard MuZero PUCT score ``Q(s,a) + U(s,a)``."""
        pb_c = math.log((parent.visit_count + self.config.pb_c_base + 1.0) / self.config.pb_c_base)
        pb_c += self.config.pb_c_init
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1.0)
        prior_score = pb_c * child.prior
        if child.visit_count > 0:
            q_value = child.reward + self.config.discount * child.value()
            value_score = minmax.normalize(q_value) if self.config.normalize_values else q_value
        else:
            value_score = 0.0
        return prior_score + value_score

    def _recurrent_inference(
        self, latents: list[torch.Tensor], actions: list[int], stats: _SearchStats
    ) -> Any:
        """One batched neural transition call.

        Sequential search passes a single element, but the call is already
        expressed as a batch so a later version can evaluate several leaves (or
        several environments) with one ``recurrent_inference`` invocation.
        """
        latent_batch = torch.stack(latents, dim=0)
        action_tensor = torch.tensor(actions, dtype=torch.long, device=latent_batch.device)
        stats.recurrent_inference_calls += 1
        with torch.no_grad():
            return self.model.recurrent_inference(latent_batch, action_tensor, None)

    def _expand(
        self,
        node: Node,
        policy_logits: torch.Tensor,
        mask: np.ndarray,
        stats: _SearchStats,
    ) -> None:
        masked = apply_action_mask(policy_logits.reshape(1, -1), mask)
        priors = torch.softmax(masked, dim=-1).reshape(-1).detach().cpu().numpy()
        priors = np.where(mask, priors.astype(np.float64), 0.0)
        total = float(priors.sum())
        if not total > 0.0:  # pragma: no cover - apply_action_mask validates
            raise ValueError("policy priors are degenerate after masking")
        node.expand((priors / total).tolist(), mask.tolist())
        stats.expanded_nodes += 1

    def _backup(self, path: list[Node], leaf_value: float, minmax: MinMaxStats) -> None:
        rewards = [node.reward for node in path[1:]]
        for node, value in zip(
            path, discounted_backup(rewards, leaf_value, self.config.discount), strict=True
        ):
            node.value_sum += value
            node.visit_count += 1
            minmax.update(value)

    # -- root utilities -------------------------------------------------------

    def _apply_root_noise(self, root: Node) -> None:
        assert root.children is not None
        valid = root.valid_actions()
        noise = self.rng.dirichlet([self.config.root_dirichlet_alpha] * len(valid))
        fraction = self.config.root_exploration_fraction
        for position, action in enumerate(valid):
            child = root.children[action]
            if child is None:  # pragma: no cover - valid_actions guarantees child
                continue
            child.prior = (1.0 - fraction) * child.prior + fraction * float(noise[position])

    def _select_action(self, counts: np.ndarray, policy: np.ndarray) -> int:
        if self.config.temperature <= _TEMPERATURE_EPS:
            return int(np.argmax(counts))
        return int(self.rng.choice(self.num_actions, p=policy))

    def _diagnostics(
        self,
        root: Node,
        minmax: MinMaxStats,
        stats: _SearchStats,
        root_output: Any,
    ) -> SearchDiagnostics:
        assert root.children is not None
        q_values = np.zeros(self.num_actions, dtype=np.float64)
        rewards = np.zeros(self.num_actions, dtype=np.float64)
        scores = np.zeros(self.num_actions, dtype=np.float64)
        for action in range(self.num_actions):
            child = root.children[action]
            if child is None:
                scores[action] = -math.inf
                continue
            q_values[action] = child.reward + self.config.discount * child.value()
            rewards[action] = child.reward
            scores[action] = self._puct_score(root, child, minmax)
        return SearchDiagnostics(
            num_simulations=self.config.num_simulations,
            tree_depth=stats.tree_depth,
            expanded_nodes=stats.expanded_nodes,
            recurrent_inference_calls=stats.recurrent_inference_calls,
            root_predicted_value=float(root_output.value.reshape(-1)[0]),
            q_values=q_values,
            puct_scores=scores,
            predicted_rewards=rewards,
        )

    @staticmethod
    def _validate_root_mask(mask: np.ndarray) -> None:
        if not bool(mask[0]):
            raise ValueError("root action mask must keep HOLD valid")
        if not bool(mask.any()):
            raise ValueError("root action mask must leave at least one valid action")
