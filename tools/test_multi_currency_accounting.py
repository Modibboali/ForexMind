"""Synthetic multi-currency accounting diagnostic (Phase 3.1).

For each supported instrument this tool runs a deterministic long/close
scenario and prints the raw PnL (quote currency), the converted PnL (account
currency), and the account-currency gross exposure.  It is a *diagnostic*, not
a training benchmark.

Usage:
    python -m tools.test_multi_currency_accounting
"""

from __future__ import annotations

from decimal import Decimal

from forexmind.config import _dec
from forexmind.environment.instruments import instrument_spec, known_instruments
from forexmind.environment.portfolio import Portfolio

ACCOUNT = "USD"
BALANCE = "10000"
ENTRY: dict[str, str] = {
    "EURUSD": "1.1000",
    "GBPUSD": "1.2500",
    "USDJPY": "150.00",
    "USDCHF": "0.9000",
    "AUDUSD": "0.6600",
    "USDCAD": "1.3500",
    "NZDUSD": "0.6100",
}
# A small favorable +10-pip-equivalent move (in price units).
MOVE: dict[str, str] = {
    "EURUSD": "0.0010",
    "GBPUSD": "0.0010",
    "USDJPY": "0.10",
    "USDCHF": "0.0010",
    "AUDUSD": "0.0010",
    "USDCAD": "0.0010",
    "NZDUSD": "0.0010",
}
UNITS: Decimal = Decimal("1000")


def _scenario(instrument: str) -> tuple[str, str, str, str]:
    """Return (raw_pnl_ccy, raw_pnl_str, converted_str, gross_exp_str)."""
    spec = instrument_spec(instrument)
    entry = _dec(ENTRY[instrument])
    exit_ = entry + _dec(MOVE[instrument])

    # Long position of UNITS base units.
    p = Portfolio(instrument, BALANCE, account_currency=ACCOUNT)
    p.adjust_to_target(UNITS, entry)
    p.mark_to_market(exit_)
    raw = p.raw_unrealized_pnl
    converted = p.unrealized_pnl
    gross = p.snapshot().gross_exposure
    return spec.quote_currency, f"{raw:.6f}", f"{converted:.6f}", f"{gross:.2f}"


def main() -> int:
    print("=" * 60)
    print("FOREXMIND MULTI-CURRENCY ACCOUNTING")
    print("=" * 60)
    print()
    print(f"Account currency: {ACCOUNT}")
    print()
    header = f"{'Instrument':<10} {'Raw PnL CCY':<15} {'Converted PnL USD':<18} {'Gross Exp USD'}"
    print(header)
    print("-" * 63)
    for instrument in known_instruments():
        quote, raw, converted, gross = _scenario(instrument)
        raw_col = f"{raw} {quote}"
        print(f"{instrument:<10} {raw_col:<15} {converted:<18} {gross:<14}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
