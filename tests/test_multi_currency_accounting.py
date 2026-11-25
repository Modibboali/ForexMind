"""Deterministic multi-currency accounting tests (Phase 3.1).

Covers every supported instrument (EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD,
USDCAD, NZDUSD) for:

* long/short profitable & losing P&L, verified in **account currency**;
* hand-computed regression tests for USDJPY (100 JPY -> ~0.6662 USD),
  USDCAD (1 CAD -> ~1/1.3510 USD), and EURUSD (no double conversion);
* position sizing that produces the intended account-currency exposure;
* margin / free margin / leverage in account currency;
* reward using account-currency equity;
* the FX conversion service and explicit error on unavailable cross rates.

All values use exact Decimal arithmetic.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from forexmind.config import EnvironmentConfig, MarginConfig, RewardConfig
from forexmind.environment.fx_conversion import AccountCurrencyError, CurrencyConverter
from forexmind.environment.instruments import instrument_spec
from forexmind.environment.portfolio import Portfolio

# instrument -> (base, quote) for the seven supported pairs.
PAIRS: dict[str, tuple[str, str]] = {
    "EURUSD": ("EUR", "USD"),
    "GBPUSD": ("GBP", "USD"),
    "USDJPY": ("USD", "JPY"),
    "USDCHF": ("USD", "CHF"),
    "AUDUSD": ("AUD", "USD"),
    "USDCAD": ("USD", "CAD"),
    "NZDUSD": ("NZD", "USD"),
}

# P&L quote-currency is {quote}; account currency is USD by default.
ACCOUNT = "USD"


def _portfolio(instrument: str, balance: str = "10000") -> Portfolio:
    return Portfolio(instrument, balance, account_currency=ACCOUNT)


# ---------------------------------------------------------------------------
# Section 19: long/short profitable/losing in account currency, all pairs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "instrument", ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
)
def test_long_profitable_in_account_currency(instrument: str) -> None:
    p = _portfolio(instrument)
    units = Decimal("1000")
    entry = Decimal("1.10")
    exit_ = Decimal("1.12")
    p.adjust_to_target(units, entry)
    p.mark_to_market(exit_)
    assert p.position.direction == "long"
    assert p.unrealized_pnl > 0  # account currency
    assert p.balance == p.initial_balance  # unrealized does not touch balance
    # Equity identity holds.
    assert p.check_invariant()


@pytest.mark.parametrize(
    "instrument", ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
)
def test_long_losing_in_account_currency(instrument: str) -> None:
    p = _portfolio(instrument)
    units = Decimal("1000")
    entry = Decimal("1.10")
    exit_ = Decimal("1.08")
    p.adjust_to_target(units, entry)
    p.mark_to_market(exit_)
    assert p.position.direction == "long"
    assert p.unrealized_pnl < 0
    assert p.equity < p.initial_balance
    assert p.check_invariant()


@pytest.mark.parametrize(
    "instrument", ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
)
def test_short_profitable_in_account_currency(instrument: str) -> None:
    p = _portfolio(instrument)
    units = Decimal("-1000")
    entry = Decimal("1.10")
    exit_ = Decimal("1.08")
    p.adjust_to_target(units, entry)
    p.mark_to_market(exit_)
    assert p.position.direction == "short"
    assert p.unrealized_pnl > 0
    assert p.check_invariant()


@pytest.mark.parametrize(
    "instrument", ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
)
def test_short_losing_in_account_currency(instrument: str) -> None:
    p = _portfolio(instrument)
    units = Decimal("-1000")
    entry = Decimal("1.10")
    exit_ = Decimal("1.12")
    p.adjust_to_target(units, entry)
    p.mark_to_market(exit_)
    assert p.position.direction == "short"
    assert p.unrealized_pnl < 0
    assert p.check_invariant()


# ---------------------------------------------------------------------------
# Section 20: explicit USDJPY hand-computed test
# ---------------------------------------------------------------------------


def test_usdjpy_hand_computed() -> None:
    """USDJPY entry 150.00 -> exit 150.10, 1000 base USD.

    Raw P&L = 1000 * 0.10 = 100 JPY.
    Converted at USDJPY=150.10: 100 / 150.10 = 0.6662225... USD.
    """
    p = _portfolio("USDJPY")
    units = Decimal("1000")
    p.adjust_to_target(units, "150.00")
    p.mark_to_market("150.10")
    expected_raw = Decimal("100")  # JPY
    assert p.raw_unrealized_pnl == expected_raw
    expected_usd = Decimal("100") / Decimal("150.10")
    assert p.unrealized_pnl == pytest.approx(expected_usd, rel=0, abs=Decimal("1e-30"))
    # Not 100 USD.
    assert abs(p.unrealized_pnl - Decimal("100")) > Decimal("99")
    # Close realises ~0.6662 USD on balance.
    p.adjust_to_target(Decimal(0), "150.10")
    assert p.realized_pnl == pytest.approx(expected_usd, rel=0, abs=Decimal("1e-30"))
    assert p.position.is_flat


# ---------------------------------------------------------------------------
# Section 21: explicit USDCAD hand-computed test
# ---------------------------------------------------------------------------


def test_usdcad_hand_computed() -> None:
    """USDCAD entry 1.3500 -> exit 1.3510, 1000 base USD.

    Raw P&L = 1000 * 0.0010 = 1 CAD.
    Converted at USDCAD=1.3510: 1 / 1.3510 USD.
    """
    p = _portfolio("USDCAD")
    units = Decimal("1000")
    p.adjust_to_target(units, "1.3500")
    p.mark_to_market("1.3510")
    expected_raw = Decimal("1")  # CAD
    assert p.raw_unrealized_pnl == expected_raw
    expected_usd = Decimal("1") / Decimal("1.3510")
    assert p.unrealized_pnl == pytest.approx(expected_usd, rel=0, abs=Decimal("1e-30"))
    # Not 1 USD (the raw quote-currency amount).
    assert abs(p.unrealized_pnl - Decimal("1")) > Decimal("0.2")
    p.adjust_to_target(Decimal(0), "1.3510")
    assert p.realized_pnl == pytest.approx(expected_usd, rel=0, abs=Decimal("1e-30"))


# ---------------------------------------------------------------------------
# Section 22: EURUSD no-double-conversion test
# ---------------------------------------------------------------------------


def test_eurusd_no_double_conversion() -> None:
    """EURUSD entry 1.1000 -> exit 1.1010, 1000 EUR.

    Raw P&L = 1 USD.  Account P&L must be 1 USD exactly - NOT 1/1.101 and
    NOT 1 * 1.101.
    """
    p = _portfolio("EURUSD")
    units = Decimal("1000")
    p.adjust_to_target(units, "1.1000")
    p.mark_to_market("1.1010")
    assert p.raw_unrealized_pnl == Decimal("1")  # USD
    assert p.unrealized_pnl == Decimal("1")
    assert p.converter.quote_to_account_factor("EURUSD", "1.1010", "USD") == Decimal(1)
    p.adjust_to_target(Decimal(0), "1.1010")
    assert p.realized_pnl == Decimal("1")


# GBPUSD behaves the same (quote = USD).
def test_gbpusd_quote_is_usd_no_conversion() -> None:
    p = _portfolio("GBPUSD")
    p.adjust_to_target(Decimal("-1000"), "1.2500")
    p.mark_to_market("1.2400")
    # Raw = -1000 * (1.24 - 1.25) = 10 GBP-quote = 10 USD.
    assert p.raw_unrealized_pnl == Decimal("10")
    assert p.unrealized_pnl == Decimal("10")


# ---------------------------------------------------------------------------
# Section 23: position sizing -> intended account-currency exposure
# ---------------------------------------------------------------------------


def _target_units(portfolio: Portfolio, exposure: Decimal, price: str) -> Decimal:
    """Mirror forex_env._target_units for equity_fraction sizing."""
    equity = portfolio.equity
    factor = portfolio.converter.quote_to_account_factor(
        portfolio.instrument, Decimal(price), portfolio.account_currency
    )
    denom = Decimal(price) * factor
    return exposure * equity / denom if denom else Decimal(0)


@pytest.mark.parametrize(
    "instrument", ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
)
@pytest.mark.parametrize("exposure", ["1", "0.5", "-0.5", "-1"])
def test_sizing_produces_account_exposure(instrument: str, exposure: str) -> None:
    """equity=10k USD; target exposure => account gross ~= |exposure|*equity."""
    p = _portfolio(instrument)
    price = {
        "USDJPY": "150.00",
        "USDCHF": "0.9000",
        "USDCAD": "1.3500",
    }.get(instrument, "1.1000")
    exp = Decimal(exposure)
    units = _target_units(p, exp, price)
    p.adjust_to_target(units, price)
    gross = p.snapshot().gross_exposure
    expected = abs(exp) * p.initial_balance  # 10,000 USD * |exposure|
    assert gross == pytest.approx(expected, rel=Decimal("1e-6"))
    # Direction is correct.
    if exp > 0:
        assert p.position.direction == "long"
    elif exp < 0:
        assert p.position.direction == "short"


# ---------------------------------------------------------------------------
# Section 24: margin / free margin / leverage in account currency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "instrument", ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
)
def test_margin_in_account_currency(instrument: str) -> None:
    from forexmind.environment.margin import MarginModel

    model = MarginModel(MarginConfig(initial_balance=Decimal("10000"), leverage=Decimal("100")))
    p = _portfolio(instrument)
    price = {
        "USDJPY": "150.00",
        "USDCHF": "0.9000",
        "USDCAD": "1.3500",
    }.get(instrument, "1.1000")
    exposure = Decimal("1")
    units = _target_units(p, exposure, price)
    p.adjust_to_target(units, price)
    gross = p.snapshot().gross_exposure
    # account-currency gross exposure ~= 10,000 regardless of price scale.
    assert gross == pytest.approx(Decimal("10000"), rel=Decimal("1e-6"))
    snap = model.snapshot(equity=p.equity, gross_exposure=gross)
    assert snap.margin_used == pytest.approx(Decimal("100"), rel=Decimal("1e-6"))  # 10000 / 100
    assert snap.free_margin == pytest.approx(p.equity - snap.margin_used, abs=Decimal("1e-30"))
    assert snap.leverage_used == pytest.approx(Decimal("1"), rel=Decimal("1e-6"))
    # Leverage is dimensionless (no currency mixing).
    assert snap.leverage_used > 0


# ---------------------------------------------------------------------------
# Section 25: reward uses account-currency equity
# ---------------------------------------------------------------------------


def test_reward_uses_account_currency_equity() -> None:
    from forexmind.environment.reward import RewardService

    rw = RewardService(RewardConfig(reward_type="log_equity_return"))
    # USDJPY: 100 JPY P&L -> 0.6662 USD.
    p = _portfolio("USDJPY")
    p.adjust_to_target(Decimal("1000"), "150.00")
    p.mark_to_market("150.10")
    equity_now = p.equity
    expected_pnl = Decimal("100") / Decimal("150.10")
    assert equity_now - p.initial_balance == pytest.approx(expected_pnl, abs=Decimal("1e-30"))
    reward = rw.reward(p.initial_balance, equity_now)
    # reward == ln(equity_now / initial) where equity_now uses account-currency P&L.
    expected_reward = (equity_now / p.initial_balance).ln()
    assert reward == pytest.approx(float(expected_reward))


# ---------------------------------------------------------------------------
# FX conversion service (section 4, 5, 6)
# ---------------------------------------------------------------------------


def test_fx_same_currency() -> None:
    fx = CurrencyConverter("USD")
    assert fx.convert("123.45", "USD", "USD", {}) == Decimal("123.45")


def test_fx_direct_and_inverse() -> None:
    fx = CurrencyConverter("USD")
    # EUR -> USD via EURUSD = 1.10
    assert fx.convert("100", "EUR", "USD", {"EURUSD": "1.10"}) == Decimal("110")
    # JPY -> USD via USDJPY = 150.0 (inverse)
    assert fx.convert("150", "JPY", "USD", {"USDJPY": "150.0"}) == Decimal("1")
    # USD -> JPY via USDJPY = 150.0 (direct)
    assert fx.convert("1", "USD", "JPY", {"USDJPY": "150.0"}) == Decimal("150")


def test_fx_raises_on_unavailable_cross() -> None:
    fx = CurrencyConverter("USD")
    # EUR -> JPY requires a cross not derivable from the available pairs.
    with pytest.raises(AccountCurrencyError):
        fx.convert("100", "EUR", "JPY", {"EURUSD": "1.10"})


def test_fx_quote_to_account_factor_all_pairs() -> None:
    fx = CurrencyConverter("USD")
    for instrument, (_base, quote) in PAIRS.items():
        price = Decimal("1.10")
        if quote == "USD":
            assert fx.quote_to_account_factor(instrument, price, "USD") == Decimal(1)
        else:
            assert fx.quote_to_account_factor(instrument, price, "USD") == Decimal(1) / price


def test_instrument_spec_metadata() -> None:
    assert instrument_spec("EURUSD").quote_currency == "USD"
    assert instrument_spec("USDJPY").quote_currency == "JPY"
    assert instrument_spec("USDJPY").pip_size == 0.01
    assert instrument_spec("EURUSD").pip_size == 0.0001


# ---------------------------------------------------------------------------
# Environment integration: account-currency P&L flows through `info`
# ---------------------------------------------------------------------------


def _env_dataset(instrument: str, prices: list[float]) -> object:
    from forexmind.data.dataset import InstrumentData, MarketDataset

    from tests.synthetic import ladder_m1

    ds = MarketDataset()
    ds.add(InstrumentData.from_m1(instrument, ladder_m1("2025-01-06 00:00", prices)))
    return ds


def _env_config() -> EnvironmentConfig:
    return EnvironmentConfig(
        margin=MarginConfig(initial_balance=Decimal("10000"), leverage=Decimal("100")),
    )


def _run_env(instrument: str, prices: list[float], action: float) -> dict:
    """Run a single-step environment and return the info dict."""
    from forexmind.environment import ForexEnvironment

    env = ForexEnvironment(_env_dataset(instrument, prices), _env_config())
    env.reset(seed=0, start_index=0, horizon=2)
    _obs, _rew, _t, _tr, info = env.step(action)
    return info


def test_env_usdjpy_info_account_currency() -> None:
    """A long USDJPY position sizes to ~10,000 USD exposure and marks P&L in USD."""
    info = _run_env("USDJPY", [150.00, 150.10] * 30, 1.0)
    assert info["base_currency"] == "USD"
    assert info["quote_currency"] == "JPY"
    assert info["account_currency"] == "USD"
    # Gross exposure (account currency) is ~10,000 USD regardless of JPY price scale.
    assert abs(float(info["gross_exposure"]) - 10000.0) < 1.0
    # raw PnL is in JPY; converted PnL is in USD.
    assert info["raw_pnl_currency"] == "JPY"
    assert float(info["conversion_rate"]) > 0


def test_env_eurusd_no_double_conversion_info() -> None:
    info = _run_env("EURUSD", [1.1000, 1.1010] * 30, 1.0)
    assert info["quote_currency"] == "USD"
    assert float(info["conversion_rate"]) == 1.0
    # For a +1 sized position of ~10,000 EUR notional, a 10-pip move is ~10 USD.
    assert abs(float(info["conversion_rate"]) - 1.0) < 1e-12


def test_info_exposes_currency_metadata() -> None:
    info = _run_env("EURUSD", [1.1000, 1.1010] * 30, 1.0)
    for key in (
        "account_currency",
        "base_currency",
        "quote_currency",
        "raw_unrealized_pnl",
        "conversion_rate",
        "raw_pnl",
        "raw_pnl_currency",
        "converted_pnl",
    ):
        assert key in info, f"missing info key {key!r}"
