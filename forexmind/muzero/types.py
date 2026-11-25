"""Shared result types for the MuZero inference API (Stage 4.1).

Both :meth:`MuZeroNetwork.initial_inference` and
:meth:`MuZeroNetwork.recurrent_inference` return a single :class:`NetworkOutput`
so MCTS can treat root and child expansions uniformly and never needs to know
about internal layers.

Field naming is deliberately unambiguous:

* ``value`` / ``reward`` are the *decoded scalars* ``[B, 1]``.
* ``value_logits`` / ``reward_logits`` are the *raw support logits*
  ``[B, value_support_size]`` / ``[B, reward_support_size]`` when support-based
  prediction is enabled, otherwise ``None``.

The two are never mixed silently: either the logits are present and consistent
with the scalar, or support is disabled and both logits are ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def assert_shape(tensor: torch.Tensor, expected: tuple[int, ...], *, name: str) -> None:
    """Raise ``ValueError`` when ``tensor.shape`` differs from ``expected``."""
    if tuple(tensor.shape) != tuple(expected):
        raise ValueError(f"{name}: expected shape {tuple(expected)}, got {tuple(tensor.shape)}")


@dataclass(frozen=True, slots=True)
class NetworkOutput:
    """Structured output shared by initial and recurrent inference.

    Shapes (batch size ``B``)::

        latent_state  [B, latent_dim]
        policy_logits [B, num_actions]
        value         [B, 1]
        reward        [B, 1]
        value_logits  [B, value_support_size]  or None
        reward_logits [B, reward_support_size] or None

    For *initial* inference ``reward`` is exactly zero (no dynamics transition
    has happened yet) and ``reward_logits`` is ``None``.
    """

    latent_state: torch.Tensor
    policy_logits: torch.Tensor
    value: torch.Tensor
    reward: torch.Tensor
    value_logits: torch.Tensor | None = None
    reward_logits: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.latent_state.ndim != 2:
            raise ValueError(
                f"latent_state must be 2-D [B, latent_dim], got {tuple(self.latent_state.shape)}"
            )
        batch = int(self.latent_state.shape[0])
        if self.policy_logits.ndim != 2:
            raise ValueError(
                f"policy_logits must be 2-D [B, num_actions], got {tuple(self.policy_logits.shape)}"
            )
        if self.policy_logits.shape[0] != batch:
            raise ValueError(
                "policy_logits batch mismatch: "
                f"{self.policy_logits.shape[0]} vs latent_state batch {batch}"
            )
        assert_shape(self.value, (batch, 1), name="value")
        assert_shape(self.reward, (batch, 1), name="reward")
        vl = self.value_logits
        if vl is not None and (vl.ndim != 2 or vl.shape[0] != batch):
            raise ValueError(
                f"value_logits must be 2-D [B, value_support_size], got {tuple(vl.shape)}"
            )
        rl = self.reward_logits
        if rl is not None and (rl.ndim != 2 or rl.shape[0] != batch):
            raise ValueError(
                f"reward_logits must be 2-D [B, reward_support_size], got {tuple(rl.shape)}"
            )

    @property
    def batch_size(self) -> int:
        return int(self.latent_state.shape[0])

    @property
    def latent_dim(self) -> int:
        return int(self.latent_state.shape[-1])

    @property
    def num_actions(self) -> int:
        return int(self.policy_logits.shape[-1])

    def policy_probs(self) -> torch.Tensor:
        """Softmax of ``policy_logits`` (external convenience only).

        The core networks never apply softmax themselves; this helper exists so
        callers/tests can inspect the induced categorical prior after masking.
        """
        return torch.softmax(self.policy_logits, dim=-1)
