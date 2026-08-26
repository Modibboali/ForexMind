"""ForexMind baselines: flat, random, long/short, momentum, mean reversion, SMA.

Importing this package registers all baselines with
:func:`forexmind.baselines.base.make_agent`.
"""

from forexmind.baselines import (  # noqa: F401  (import for registration side effects)
    buy_hold,
    flat,
    mean_reversion,
    momentum,
    random,
    sma_crossover,
)
from forexmind.baselines.base import (
    TradingAgent,
    available_agents,
    make_agent,
    register_agent,
)

__all__ = [
    "TradingAgent",
    "available_agents",
    "make_agent",
    "register_agent",
]
