"""Observation encoder (Phase 2).

Converts a Phase-1 :class:`Observation` (account + time + identity) plus a
causal :class:`MarketWindow` into an :class:`EncodedObservation`.

Market features (per bar, scale-aware, never raw prices):

* ``open_return``   = O_t / C_{t-1} - 1
* ``high_return``   = H_t / C_{t-1} - 1
* ``low_return``    = L_t / C_{t-1} - 1
* ``close_return``  = C_t / C_{t-1} - 1
* ``log_return``    = log(C_t / C_{t-1})

where ``C_{t-1}`` is the previous bar's close (``prior_close`` for the first
bar of the window).  Account and time features are normalized relative to the
initial balance / equity and are causally available.

No fitted statistics are used (analytical transformations only), so there is
no normalization leakage.  A ``Normalizer`` hook exists for future features.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal

import numpy as np

from forexmind.config import _dec
from forexmind.data.splits import DEFAULT_INSTRUMENT_ORDER
from forexmind.environment.state import Observation as Phase1Observation
from forexmind.environment.state import Session
from forexmind.observation.normalization import NormalizerConfig, make_normalizer
from forexmind.observation.schema import (
    DEFAULT_MARKET_FEATURES,
    EncodedObservation,
    ObservationSpec,
)
from forexmind.observation.window import MarketWindow

_MINUTES_PER_DAY = 1440.0
_SESSION_ORDER: tuple[Session, ...] = (
    Session.ASIA,
    Session.LONDON,
    Session.NEW_YORK,
    Session.OVERLAP,
    Session.QUIET,
    Session.UNKNOWN,
)


@dataclass(frozen=True)
class EncoderConfig:
    """Encoder configuration (serializable)."""

    context_length: int = 64
    market_features: tuple[str, ...] = DEFAULT_MARKET_FEATURES
    initial_balance: Decimal | float | str = Decimal("10000")
    instrument_order: tuple[str, ...] = DEFAULT_INSTRUMENT_ORDER
    dtype: str = "float32"
    max_leverage_feature: float = 100.0  # clamp for leverage/margin outliers
    normalizer: NormalizerConfig = field(default_factory=NormalizerConfig)

    def __post_init__(self) -> None:
        if self.context_length <= 0:
            raise ValueError("context_length must be > 0")
        for f in self.market_features:
            if f not in DEFAULT_MARKET_FEATURES:
                raise ValueError(f"unsupported market feature {f!r}")

    @property
    def spec(self) -> ObservationSpec:
        return ObservationSpec(
            context_length=self.context_length,
            n_market_features=len(self.market_features),
            n_instruments=len(self.instrument_order),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "context_length": self.context_length,
            "market_features": list(self.market_features),
            "initial_balance": str(self.initial_balance),
            "instrument_order": list(self.instrument_order),
            "dtype": self.dtype,
            "max_leverage_feature": self.max_leverage_feature,
            "normalizer": {
                "market": self.normalizer.market,
                "account": self.normalizer.account,
                "time": self.normalizer.time,
            },
        }


class ObservationEncoder:
    """Encodes Phase-1 observations + market windows into agent observations."""

    def __init__(self, config: EncoderConfig | None = None) -> None:
        self.config = config or EncoderConfig()
        self._initial = _dec(self.config.initial_balance)
        self._dtype = np.dtype(self.config.dtype)
        self._instrument_index = {
            name.upper(): i for i, name in enumerate(self.config.instrument_order)
        }
        self._session_index = {s: i for i, s in enumerate(_SESSION_ORDER)}
        self._market_normalizer = make_normalizer(self.config.normalizer.market)
        self._account_normalizer = make_normalizer(self.config.normalizer.account)
        self._time_normalizer = make_normalizer(self.config.normalizer.time)

    # -- market ---------------------------------------------------------------

    def encode_market(self, window: MarketWindow) -> np.ndarray:
        close = window.closes
        prior = np.float64(window.prior_close)
        base = np.empty_like(close)
        base[0] = prior
        base[1:] = close[:-1]
        base = np.maximum(base, 1e-12)  # guard; prices are > 0
        out = np.zeros((len(close), len(self.config.market_features)), dtype=self._dtype)
        for j, feat in enumerate(self.config.market_features):
            if feat == "open_return":
                out[:, j] = window.open / base - 1.0
            elif feat == "high_return":
                out[:, j] = window.high / base - 1.0
            elif feat == "low_return":
                out[:, j] = window.low / base - 1.0
            elif feat == "close_return":
                out[:, j] = close / base - 1.0
            elif feat == "log_return":
                out[:, j] = np.log(np.maximum(close / base, 1e-12))
        return self._market_normalizer.transform(out)

    # -- account --------------------------------------------------------------

    def encode_account(self, obs: Phase1Observation) -> np.ndarray:
        a = obs.account
        init = float(self._initial)
        equity = float(a.equity)
        units = float(a.position_units)

        if units == 0.0 or equity <= 0.0:
            position_exposure = 0.0
        else:
            position_exposure = math.copysign(float(a.gross_exposure) / equity, units)
        if units != 0.0 and float(a.entry_price) > 0.0:
            entry_distance = float(a.unrealized_pnl) / (units * float(a.entry_price))
        else:
            entry_distance = 0.0

        margin_util = float(a.margin_used) / equity if equity > 0.0 else 0.0
        free_margin_ratio = float(a.free_margin) / equity if equity > 0.0 else 0.0
        leverage_used = (
            float(a.gross_exposure) / equity
            if equity > 0.0 and float(a.gross_exposure) > 0.0
            else 0.0
        )
        leverage_used = float(np.clip(leverage_used, 0.0, self.config.max_leverage_feature))
        margin_util = float(np.clip(margin_util, 0.0, self.config.max_leverage_feature))

        vec = np.array(
            [
                position_exposure,  # signed exposure relative to equity
                units / init,  # position units per unit of initial capital
                entry_distance,  # (mid - entry)/entry
                float(a.unrealized_pnl) / init,
                float(a.realized_pnl) / init,
                equity / init - 1.0,  # equity return from initial
                float(a.drawdown) / init,
                margin_util,
                free_margin_ratio,
                leverage_used,
            ],
            dtype=self._dtype,
        )
        return self._account_normalizer.transform(vec)

    # -- time -----------------------------------------------------------------

    def encode_time(self, obs: Phase1Observation) -> np.ndarray:
        t = obs.time
        hour = t.hour / 24.0
        minute = t.minute / 60.0
        dow = t.day_of_week / 7.0
        session_vec = np.zeros(len(_SESSION_ORDER), dtype=self._dtype)
        session_vec[self._session_index[t.session]] = 1.0
        vec = np.concatenate(
            [
                np.array(
                    [
                        math.sin(2 * math.pi * hour),
                        math.cos(2 * math.pi * hour),
                        math.sin(2 * math.pi * minute),
                        math.cos(2 * math.pi * minute),
                        math.sin(2 * math.pi * dow),
                        math.cos(2 * math.pi * dow),
                        t.minutes_since_last_bar / _MINUTES_PER_DAY,
                        1.0 if t.is_weekend_gap else 0.0,
                    ],
                    dtype=self._dtype,
                ),
                session_vec,
            ]
        )
        return self._time_normalizer.transform(vec)

    # -- instrument -----------------------------------------------------------

    def encode_instrument(self, instrument: str) -> np.ndarray:
        vec = np.zeros(len(self.config.instrument_order), dtype=self._dtype)
        idx = self._instrument_index.get(instrument.upper())
        if idx is None:
            raise KeyError(
                f"instrument {instrument!r} not in encoder instrument_order "
                f"{self.config.instrument_order}"
            )
        vec[idx] = 1.0
        return vec

    # -- combined -------------------------------------------------------------

    def encode(self, obs: Phase1Observation, window: MarketWindow) -> EncodedObservation:
        return EncodedObservation(
            instrument=obs.instrument,
            step_index=obs.step_index,
            timestamp=obs.timestamp,
            spec=self.config.spec,
            market=self.encode_market(window),
            account=self.encode_account(obs),
            time=self.encode_time(obs),
            instrument_vec=self.encode_instrument(obs.instrument),
            closes=window.closes.astype(self._dtype),
            prior_close=float(window.prior_close),
        )
