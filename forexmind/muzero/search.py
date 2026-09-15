"""MuZero MCTS / PUCT search (Stage 4.2, batched driver added in Stage 4.6).

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

Stage 4.6 adds **batched search** without adding a second search algorithm.
:meth:`MuZeroMCTS.search_batch` drives ``B`` independent roots through the same
``num_simulations`` rounds, collecting every root's leaf request into one
recurrent-inference call per round::

    round k:  root A leaf -+
              root B leaf -+--> one recurrent_inference([B, ...]) --> expand/backup
              root C leaf -+

Each root keeps its own tree, ``MinMaxStats`` and RNG stream, and the RNG draws
happen in root order, so ``search_batch`` with a single root is *identical* to
the Stage 4.2 single-root search (:meth:`MuZeroMCTS.search` simply delegates to
it).  Batching only merges neural-network calls; visit counts, Q values,
children and planning state are never shared between roots (brief S15).

Inference itself goes through a pluggable backend
(:mod:`forexmind.muzero.inference_service`), so the same search code can run
against a local model (Stage 4.5 behaviour) or a central batching service.

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
from forexmind.muzero.inference_service import (
    InferenceBackend,
    LocalInferenceBackend,
    slice_network_output,
)
from forexmind.muzero.minmax import MinMaxStats
from forexmind.muzero.node import Node
from forexmind.muzero.profiling import PhaseTimer, phase

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
        backend: InferenceBackend | None = None,
        timer: PhaseTimer | None = None,
    ) -> None:
        self.model = model
        self.config = config or SearchConfig()
        self.rng = rng if rng is not None else np.random.default_rng(self.config.seed)
        #: Optional wall-time instrumentation (Stage 4.6 profiling, brief S2).
        self.timer = timer
        #: Where inference happens.  ``None`` keeps the Stage 4.5 behaviour of
        #: calling ``model`` directly in this process.
        self.backend: InferenceBackend = (
            backend if backend is not None else LocalInferenceBackend(model)
        )
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
        #: Roots of the most recent (possibly batched) search, same contract.
        self.last_roots: list[Node] = []

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

        This is exactly ``search_batch`` with one root; the single-root and
        multi-root paths therefore cannot drift apart.
        """
        return self.search_batch(
            [observation],
            [action_mask],
            planning_states=[planning_state],
            add_root_noise=add_root_noise,
        )[0]

    def search_batch(
        self,
        observations: Sequence[Any] | Any,
        action_masks: Sequence[Any],
        *,
        planning_states: Sequence[PlanningState | None] | None = None,
        add_root_noise: bool | None = None,
        rngs: Sequence[np.random.Generator] | None = None,
    ) -> list[SearchResult]:
        """Run ``B`` independent searches in lock-step, batching their inference.

        Each root owns its tree, its min-max value statistics and its share of
        the RNG stream; only the neural-network calls are merged (brief S14-S15).
        The returned list is in the same order as ``observations``.

        ``rngs`` optionally gives each root its own generator.  Parallel
        collectors use it so a root's root-noise and action sampling depend only
        on that root's ``(worker, episode, decision)`` seed and never on which
        other roots happened to share the batch.
        """
        observation_list = _normalise_observations(observations)
        mask_list = [np.asarray(mask) for mask in action_masks]
        num_roots = len(observation_list)
        if num_roots == 0:
            raise ValueError("search_batch requires at least one root")
        if len(mask_list) != num_roots:
            raise ValueError(
                f"got {num_roots} observations but {len(mask_list)} action masks"
            )
        if planning_states is None:
            state_list: list[PlanningState | None] = [None] * num_roots
        else:
            state_list = list(planning_states)
            if len(state_list) != num_roots:
                raise ValueError(
                    f"got {num_roots} observations but {len(state_list)} planning states"
                )
        if rngs is None:
            generators: list[np.random.Generator] = [self.rng] * num_roots
        else:
            generators = list(rngs)
            if len(generators) != num_roots:
                raise ValueError(f"got {num_roots} observations but {len(generators)} RNGs")

        masks = [project_action_mask(mask) for mask in mask_list]
        for mask in masks:
            self._validate_root_mask(mask)
        states = [
            state if state is not None else PlanningState.from_action_mask(mask)
            for state, mask in zip(state_list, masks, strict=True)
        ]
        for state, mask in zip(states, masks, strict=True):
            if not np.array_equal(state.action_mask(), mask):
                raise ValueError("planning_state does not reproduce the supplied action mask")
        use_noise = self.config.add_root_noise if add_root_noise is None else bool(add_root_noise)

        stats = [_SearchStats() for _ in range(num_roots)]
        minmax = [MinMaxStats() for _ in range(num_roots)]

        root_output = self._initial_inference(observation_list, masks)
        roots: list[Node] = []
        for index, mask in enumerate(masks):
            root = Node(prior=1.0, reward=0.0, planning_state=states[index])
            root.latent_state = root_output.latent_state[index]
            self._expand(root, root_output.policy_logits[index], mask, stats[index])
            if use_noise:
                self._apply_root_noise(root, generators[index])
            roots.append(root)

        for _ in range(self.config.num_simulations):
            self._simulate_round(roots, minmax, stats)

        self.last_roots = roots
        self.last_root = roots[0]
        return [
            self._result(
                root,
                mask,
                minmax[index],
                stats[index],
                slice_network_output(root_output, index),
                generators[index],
            )
            for index, (root, mask) in enumerate(zip(roots, masks, strict=True))
        ]

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

    def _initial_inference(
        self, observations: Sequence[Any], masks: Sequence[np.ndarray]
    ) -> Any:
        """One initial-inference call for every root (batched when B > 1).

        A single root keeps the Stage 4.2 call shape exactly (unconverted
        observation and 1-D mask), so the single-root path is bit-identical to
        Stage 4.5.
        """
        if len(observations) == 1:
            with phase(self.timer, "initial_inference"):
                return self.backend.initial_inference(observations[0], masks[0])
        with phase(self.timer, "initial_inference"):
            return self.backend.initial_inference(
                _stack_observations(observations), np.stack([np.asarray(m) for m in masks])
            )

    def _simulate_round(
        self,
        roots: Sequence[Node],
        minmax: Sequence[MinMaxStats],
        stats: Sequence[_SearchStats],
    ) -> None:
        """One simulation for every root, with a single shared recurrent batch.

        The traversal, expansion and backup performed for each root are exactly
        the operations the Stage 4.2 single-root simulation performed; the only
        change is that all roots' leaf transitions are evaluated by one
        ``recurrent_inference`` call (brief S9, S14, S15).
        """
        pending: list[tuple[int, list[Node], Node, int]] = []
        with phase(self.timer, "mcts_tree_logic"):
            for index, root in enumerate(roots):
                node = root
                path: list[Node] = [root]
                actions: list[int] = []
                while node.is_expanded():
                    action, child = self._select_child(node, minmax[index])
                    actions.append(action)
                    node = child
                    path.append(node)
                parent = path[-2]
                action = actions[-1]
                assert parent.latent_state is not None
                pending.append((index, path, parent, action))

        latents = [parent.latent_state for _, _, parent, _ in pending]
        actions_list = [action for _, _, _, action in pending]
        output = self._recurrent_inference(latents, actions_list)

        with phase(self.timer, "mcts_expand_backup"):
            for position, (index, path, parent, action) in enumerate(pending):
                leaf = path[-1]
                leaf.latent_state = output.latent_state[position]
                leaf.reward = float(output.reward.reshape(-1)[position])
                assert parent.planning_state is not None
                leaf.planning_state = parent.planning_state.after(action)
                self._expand(
                    leaf,
                    output.policy_logits[position],
                    leaf.planning_state.action_mask(),
                    stats[index],
                )
                self._backup(path, float(output.value.reshape(-1)[position]), minmax[index])
                stats[index].tree_depth = max(stats[index].tree_depth, len(path) - 1)
                # One leaf evaluation for this root; the call itself was shared
                # with the other roots (backend diagnostics count real calls).
                stats[index].recurrent_inference_calls += 1

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

    def _recurrent_inference(self, latents: list[torch.Tensor], actions: list[int]) -> Any:
        """One batched neural transition call.

        Single-root search passes one element; batched search passes one element
        per root.  The call shape is identical either way, so batching changes
        scheduling only, never the model computation.
        """
        latent_batch = torch.stack(latents, dim=0)
        action_tensor = torch.tensor(actions, dtype=torch.long, device=latent_batch.device)
        with phase(self.timer, "recurrent_inference"):
            return self.backend.recurrent_inference(latent_batch, action_tensor, None)

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

    def _apply_root_noise(self, root: Node, rng: np.random.Generator | None = None) -> None:
        assert root.children is not None
        valid = root.valid_actions()
        generator = self.rng if rng is None else rng
        noise = generator.dirichlet([self.config.root_dirichlet_alpha] * len(valid))
        fraction = self.config.root_exploration_fraction
        for position, action in enumerate(valid):
            child = root.children[action]
            if child is None:  # pragma: no cover - valid_actions guarantees child
                continue
            child.prior = (1.0 - fraction) * child.prior + fraction * float(noise[position])

    def _select_action(
        self,
        counts: np.ndarray,
        policy: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> int:
        if self.config.temperature <= _TEMPERATURE_EPS:
            return int(np.argmax(counts))
        generator = self.rng if rng is None else rng
        return int(generator.choice(self.num_actions, p=policy))

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

    def _result(
        self,
        root: Node,
        mask: np.ndarray,
        minmax: MinMaxStats,
        stats: _SearchStats,
        root_output: Any,
        rng: np.random.Generator | None = None,
    ) -> SearchResult:
        """Turn a finished root into the public :class:`SearchResult`."""
        assert root.children is not None
        counts = np.array(
            [child.visit_count if child is not None else 0 for child in root.children],
            dtype=np.float64,
        )
        policy = visit_count_policy(counts, self.config.temperature)
        action = self._select_action(counts, policy, rng)
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
            root_action_mask=np.asarray(mask, dtype=bool).copy(),
            diagnostics=diagnostics,
        )

    @staticmethod
    def _validate_root_mask(mask: np.ndarray) -> None:
        if not bool(mask[0]):
            raise ValueError("root action mask must keep HOLD valid")
        if not bool(mask.any()):
            raise ValueError("root action mask must leave at least one valid action")


def _normalise_observations(observations: Any) -> list[Any]:
    """Accept a single observation, a list of observations, or a ``[B, ...]`` block.

    A 2-D array/tensor is read as a stack of rows; anything else is read as a
    sequence of observations.  A single observation is never converted, so the
    one-root path passes the caller's object through untouched.
    """
    if isinstance(observations, np.ndarray) and observations.ndim == 2:
        return [observations[index] for index in range(observations.shape[0])]
    if isinstance(observations, torch.Tensor) and observations.ndim == 2:
        return [observations[index] for index in range(observations.shape[0])]
    return list(observations)


def _stack_observations(observations: Sequence[Any]) -> np.ndarray:
    """Stack root observations into ``[B, obs_dim]`` float32 for one model call."""
    rows: list[np.ndarray] = []
    for observation in observations:
        array = (
            observation.detach().cpu().numpy()
            if isinstance(observation, torch.Tensor)
            else np.asarray(observation)
        )
        rows.append(np.asarray(array, dtype=np.float32).reshape(-1))
    return np.stack(rows, axis=0)
