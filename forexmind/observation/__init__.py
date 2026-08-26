"""ForexMind observation layer: schema, window, normalization, encoder."""

from forexmind.observation.encoder import EncoderConfig, ObservationEncoder
from forexmind.observation.normalization import (
    IdentityNormalizer,
    Normalizer,
    NormalizerConfig,
    StandardNormalizer,
    make_normalizer,
)
from forexmind.observation.schema import (
    DEFAULT_MARKET_FEATURES,
    N_ACCOUNT_FEATURES,
    N_MARKET_FEATURES,
    N_TIME_FEATURES,
    EncodedObservation,
    ObservationSpec,
)
from forexmind.observation.window import (
    MarketWindow,
    MarketWindowBuilder,
    WindowConfig,
    WindowError,
)

__all__ = [
    "DEFAULT_MARKET_FEATURES",
    "N_ACCOUNT_FEATURES",
    "N_MARKET_FEATURES",
    "N_TIME_FEATURES",
    "EncodedObservation",
    "EncoderConfig",
    "IdentityNormalizer",
    "MarketWindow",
    "MarketWindowBuilder",
    "Normalizer",
    "NormalizerConfig",
    "ObservationEncoder",
    "ObservationSpec",
    "StandardNormalizer",
    "WindowConfig",
    "WindowError",
    "make_normalizer",
]
