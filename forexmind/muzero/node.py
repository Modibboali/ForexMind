"""MuZero search-tree node (Stage 4.2).

A node stores the two pieces of state search needs:

* ``latent_state`` — the learned MuZero latent state (filled lazily, only when
  an edge is first traversed);
* ``planning_state`` — the deterministic discrete exposure state used to derive
  imagined action masks (:class:`forexmind.muzero.actions.PlanningState`).

Full observations are **not** duplicated per node; only the root ever receives
an observation, and it is consumed immediately by ``initial_inference``.

``reward`` is the predicted reward on the edge from the parent into this node
(the root's reward is ``0``).  It is unknown until the edge is first traversed,
so it defaults to ``0.0`` and is only read once the node has been visited.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from forexmind.muzero.actions import MUZERO_NUM_ACTIONS, PlanningState


@dataclass(slots=True)
class Node:
    """One node of the latent search tree."""

    prior: float
    reward: float = 0.0
    latent_state: torch.Tensor | None = None
    planning_state: PlanningState | None = None
    visit_count: int = 0
    value_sum: float = 0.0
    #: One entry per action: a child node for valid actions, ``None`` otherwise.
    #: ``children is None`` means "not expanded yet" (lazy expansion).
    children: list[Node | None] | None = field(default=None)

    # -- state ----------------------------------------------------------------

    def is_expanded(self) -> bool:
        """``True`` once the node's valid children and priors exist."""
        return self.children is not None

    def value(self) -> float:
        """Mean backed-up return of this node (``0.0`` when never visited)."""
        if self.visit_count <= 0:
            return 0.0
        return self.value_sum / self.visit_count

    def child(self, action: int) -> Node | None:
        if self.children is None:
            raise ValueError("node is not expanded; call expand() first")
        if not 0 <= action < MUZERO_NUM_ACTIONS:
            raise ValueError(f"action {action} out of range [0, {MUZERO_NUM_ACTIONS})")
        return self.children[action]

    def valid_actions(self) -> list[int]:
        """Indices of actions that were valid (and therefore have children)."""
        if self.children is None:
            return []
        return [i for i, child in enumerate(self.children) if child is not None]

    def total_visits(self) -> int:
        """Total visits distributed over the children (excludes this node)."""
        if self.children is None:
            return 0
        return sum(child.visit_count for child in self.children if child is not None)

    # -- expansion ------------------------------------------------------------

    def expand(self, priors: list[float], action_mask: list[bool] | tuple[bool, ...]) -> None:
        """Create one child per valid action and assign its prior.

        Invalid actions get no child at all, so they can never be selected or
        expanded.  No recurrent inference happens here — children are expanded
        lazily when an edge is first traversed.
        """
        if len(priors) != MUZERO_NUM_ACTIONS or len(action_mask) != MUZERO_NUM_ACTIONS:
            raise ValueError(f"priors and action_mask must both have {MUZERO_NUM_ACTIONS} entries")
        if self.children is not None:
            raise ValueError("node is already expanded")
        children: list[Node | None] = [
            Node(prior=float(prior)) if valid else None
            for prior, valid in zip(priors, action_mask, strict=True)
        ]
        if all(child is None for child in children):
            raise ValueError("cannot expand a node with no valid actions")
        self.children = children

    def set_planning_state(self, planning_state: PlanningState) -> None:
        self.planning_state = planning_state

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"Node(prior={self.prior:.4f}, visits={self.visit_count}, "
            f"value={self.value():.4f}, reward={self.reward:.4f}, "
            f"expanded={self.is_expanded()})"
        )
