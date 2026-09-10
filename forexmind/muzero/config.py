"""MuZero configuration (Stage 4.1).

Stage 4.1 only builds and validates the MuZero *neural architecture and
inference contracts*.  This module holds every architecture constant so the
networks never hard-code dimensions, and it derives the observation dimension
from the frozen Phase-2 observation encoder rather than assuming ``351``.

The defaults are deliberately compact: the goal is correctness and throughput
before scale.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from forexmind.observation.encoder import EncoderConfig

SUPPORTED_ACTIVATIONS: tuple[str, ...] = ("silu", "relu", "tanh", "gelu", "elu")


def observation_dim(encoder_config: EncoderConfig | None = None) -> int:
    """Return the flat observation dimension of the Phase-2 encoder.

    The value is *derived* from :class:`forexmind.observation.schema.ObservationSpec`
    (``context_length * n_market_features + n_account_features + n_time_features +
    n_instruments``) so it can never drift from the environment contract.  With
    the current defaults it evaluates to ``64 * 5 + 10 + 14 + 7 = 351``.
    """
    from forexmind.observation.encoder import EncoderConfig as _EncoderConfig

    config = encoder_config if isinstance(encoder_config, _EncoderConfig) else _EncoderConfig()
    return int(config.spec.encoded_shape[0])


@dataclass(frozen=True, slots=True)
class MuZeroConfig:
    """Architecture + support-transform configuration for the MuZero core.

    Attributes
    ----------
    obs_dim:
        Flat observation dimension (see :func:`observation_dim`).
    num_actions:
        Size of the categorical action space.  ForexMind defines exactly 10
        actions (``HOLD``, ``FLAT``, 4 shorts, 4 longs); this must not change.
    latent_dim:
        Dimension of the learned latent state produced by ``h_theta``.
    hidden_dim:
        Width of the internal MLP hidden layers.
    action_embedding_dim:
        Dimension of the learnable ``nn.Embedding`` used to encode a discrete
        action index before it is fed to the dynamics network.
    num_layers:
        Number of hidden MLP blocks in each sub-network.
    activation:
        One of :data:`SUPPORTED_ACTIVATIONS`.
    layer_norm:
        Apply ``LayerNorm`` after hidden projections (and to the latent state).
    residual_dynamics:
        Predict a latent *delta* and add it to the input latent
        (``next = normalize(s + delta)``) instead of mapping directly.
    use_support:
        When ``True`` reward/value are categorical-support predictions
        (MuZero-style).  When ``False`` they degrade to scalar regression heads
        while keeping the identical public inference API.
    value_support_size / reward_support_size:
        Number of support bins (must be odd so a symmetric centre exists).
    value_scale / reward_scale:
        Multiply the decoded scalar to recover the natural target scale.
    value_epsilon / reward_epsilon:
        Epsilon of the invertible support transform (0.001 matches MuZero;
        ``0.0`` reduces it to the identity for an exact round trip).
    """

    obs_dim: int
    num_actions: int = 10
    latent_dim: int = 128
    hidden_dim: int = 256
    action_embedding_dim: int = 16
    num_layers: int = 2
    activation: str = "silu"
    layer_norm: bool = True
    residual_dynamics: bool = True
    use_support: bool = True
    value_support_size: int = 21
    reward_support_size: int = 21
    value_scale: float = 1.0
    reward_scale: float = 1.0
    value_epsilon: float = 0.001
    reward_epsilon: float = 0.001

    def __post_init__(self) -> None:
        if self.obs_dim <= 0:
            raise ValueError(f"obs_dim must be > 0, got {self.obs_dim}")
        if self.num_actions < 2:
            raise ValueError(f"num_actions must be >= 2, got {self.num_actions}")
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be > 0, got {self.latent_dim}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {self.hidden_dim}")
        if self.action_embedding_dim <= 0:
            raise ValueError(f"action_embedding_dim must be > 0, got {self.action_embedding_dim}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")
        if self.activation not in SUPPORTED_ACTIVATIONS:
            raise ValueError(
                f"unsupported activation {self.activation!r}; "
                f"expected one of {SUPPORTED_ACTIVATIONS}"
            )
        if self.use_support:
            for name, size in (
                ("value_support_size", self.value_support_size),
                ("reward_support_size", self.reward_support_size),
            ):
                if size < 3 or size % 2 == 0:
                    raise ValueError(f"{name} must be an odd integer >= 3, got {size}")
        for name, scale in (("value_scale", self.value_scale), ("reward_scale", self.reward_scale)):
            if not scale > 0.0:
                raise ValueError(f"{name} must be > 0, got {scale}")
        for name, eps in (
            ("value_epsilon", self.value_epsilon),
            ("reward_epsilon", self.reward_epsilon),
        ):
            if eps < 0.0:
                raise ValueError(f"{name} must be >= 0, got {eps}")

    # -- derived --------------------------------------------------------------

    @property
    def value_output_dim(self) -> int:
        """Raw size of the value head output (support logits or a scalar)."""
        return self.value_support_size if self.use_support else 1

    @property
    def reward_output_dim(self) -> int:
        """Raw size of the reward head output (support logits or a scalar)."""
        return self.reward_support_size if self.use_support else 1

    @property
    def scalar_head(self) -> bool:
        """``True`` when reward/value are plain scalar regression heads."""
        return not self.use_support

    # -- constructors ---------------------------------------------------------

    @classmethod
    def from_encoder_config(
        cls, encoder_config: EncoderConfig | None = None, **overrides: Any
    ) -> MuZeroConfig:
        """Build a config whose ``obs_dim`` follows the Phase-2 encoder."""
        return cls(obs_dim=observation_dim(encoder_config), **overrides)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
