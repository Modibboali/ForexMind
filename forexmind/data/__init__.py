"""ForexMind data layer: canonical schema, loaders, validator, resampler, dataset."""

from forexmind.data.dataset import InstrumentData, InstrumentMeta, MarketDataset
from forexmind.data.loaders import (
    LoadConfig,
    LoadResult,
    load_any,
    load_many_concat,
    load_parquet,
    load_tabular,
)
from forexmind.data.resampler import (
    CompletenessPolicy,
    ResampleConfig,
    resample_m1_to_m5,
)
from forexmind.data.schema import (
    CANONICAL_COLUMNS,
    CLOSE,
    HIGH,
    LOW,
    OPEN,
    TIMESTAMP,
    MarketBar,
    SchemaError,
    bar_from_row,
    validate_ohlc,
)
from forexmind.data.validator import (
    GapConfig,
    GapRecord,
    GapType,
    MarketDataValidator,
    ValidationIssue,
    ValidationResult,
    classify_gap,
    summarize_gaps,
)

__all__ = [
    "CANONICAL_COLUMNS",
    "CLOSE",
    "HIGH",
    "LOW",
    "OPEN",
    "TIMESTAMP",
    "CompletenessPolicy",
    "GapConfig",
    "GapRecord",
    "GapType",
    "InstrumentData",
    "InstrumentMeta",
    "LoadConfig",
    "LoadResult",
    "MarketBar",
    "MarketDataValidator",
    "MarketDataset",
    "ResampleConfig",
    "SchemaError",
    "ValidationIssue",
    "ValidationResult",
    "bar_from_row",
    "classify_gap",
    "load_any",
    "load_many_concat",
    "load_parquet",
    "load_tabular",
    "resample_m1_to_m5",
    "summarize_gaps",
    "validate_ohlc",
]
