"""Overnight financing-cost interface (Phase 3.1).

Phase 3.1 provides an *extensible* financing interface but no broker-specific
swap tables: there is no historical swap data in the dataset, so swapped rates
are deliberately **not fabricated**.  The initial implementation
(:class:`ZeroCostFinancing`) charges nothing and is the default.

The interface is architecturally capable of computing a financing cost from a
position, the elapsed holding time, and an instrument, so real swap tables can
be introduced later without redesigning the environment.
"""

from __future__ import annotations

from decimal import Decimal

import pandas as pd

from forexmind.environment.instruments import InstrumentSpec


class FinancingModel:
    """Base interface for computing an overnight-financing cost.

    Subclasses override :meth:`financing_cost`.  The cost is expressed in the
    **account currency** (the same currency as account equity), consistent with
    the rest of the accounting layer.
    """

    def financing_cost(
        self,
        *,
        instrument_spec: InstrumentSpec,
        units: Decimal,
        entry_price: Decimal,
        hold_seconds: float,
        now: pd.Timestamp,
        entry_time: pd.Timestamp,
    ) -> Decimal:
        """Return the financing cost (account currency) for an open position.

        A positive value is a *cost* (reduces equity); a negative value is a
        credit (increases equity).
        """
        raise NotImplementedError

    def as_dict(self) -> dict[str, object]:
        return {"model": self.__class__.__name__.lower()}


class ZeroCostFinancing(FinancingModel):
    """Charges no financing.  Default Phase 3.1 model (no historical swap data)."""

    def financing_cost(
        self,
        *,
        instrument_spec: InstrumentSpec,
        units: Decimal,
        entry_price: Decimal,
        hold_seconds: float,
        now: pd.Timestamp,
        entry_time: pd.Timestamp,
    ) -> Decimal:
        return Decimal(0)
