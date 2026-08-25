"""ForexMind: research-grade deterministic Forex market/data environment.

Phase 1 provides:
  * canonical M1 market data schema and loaders
  * dataset validation and temporal gap classification
  * deterministic M1 -> M5 resampling
  * instrument-aware dataset abstraction
  * execution-cost model (spread / slippage / commission) driven by config
  * portfolio / accounting engine (Decimal based)
  * configurable margin model with deterministic liquidation
  * target-exposure action model
  * a Gymnasium-style deterministic environment
"""

from forexmind import config  # noqa: F401  (import for side-effect-free config namespace)

__version__ = "0.1.0"
