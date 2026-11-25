"""Robust tabular market-data loaders.

Reads CSV / TSV / parquet / generic tabular sources and normalises arbitrary
column names into the canonical schema::

    timestamp, open, high, low, close

Rules (per the ForexMind spec):

* tolerate common column variations (``time``/``datetime``/``date``/``Time``/
  ``Open``/...), case-insensitively;
* never *silently* accept ambiguous mappings -- raise a loud error instead;
* fail loudly when required fields are missing;
* if source timestamps are timezone-aware, normalise them to UTC;
* if the timezone is ambiguous (e.g. MetaTrader server time), do NOT guess:
  timestamps are kept naive and ``source_timezone`` is reported as ``None``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from forexmind.data.schema import CLOSE, HIGH, LOW, OPEN, TIMESTAMP, SchemaError

# ---------------------------------------------------------------------------
# Column classification
# ---------------------------------------------------------------------------

_FULL_TIMESTAMP_ALIASES = {
    "timestamp",
    "datetime",
    "datetime_utc",
    "date_time",
    "datetimeutc",
    "time_stamp",
    "ts",
    "dt",
}
_DATE_ALIASES = {"date", "day", "d"}
_TIME_ALIASES = {"time", "tm", "time_local", "t"}
_OPEN_ALIASES = {"open", "o", "open_price", "price_open", "op"}
_HIGH_ALIASES = {"high", "h", "high_price", "price_high", "hi"}
_LOW_ALIASES = {"low", "l", "low_price", "price_low", "lo"}
_CLOSE_ALIASES = {"close", "c", "close_price", "price_close", "last", "cl"}

# MT5 "no header" export order: DATE,TIME,OPEN,HIGH,LOW,CLOSE[,VOLUME]
_MT5_POSITIONAL = ("date", "time", "open", "high", "low", "close", "volume")


def _norm(name: object) -> str:
    return str(name).strip().lower().replace(" ", "_")


def classify_column(name: object) -> str | None:
    """Map a source column name to a canonical role, or ``None`` if unknown."""
    n = _norm(name)
    if n in _FULL_TIMESTAMP_ALIASES:
        return "full_timestamp"
    if n in _DATE_ALIASES:
        return "date"
    if n in _TIME_ALIASES:
        return "time"
    if n in _OPEN_ALIASES:
        return OPEN
    if n in _HIGH_ALIASES:
        return HIGH
    if n in _LOW_ALIASES:
        return LOW
    if n in _CLOSE_ALIASES:
        return CLOSE
    return None


# ---------------------------------------------------------------------------
# Loading configuration & result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadConfig:
    """Configuration for :func:`load_tabular`.

    ``sep``: delimiter; ``None`` auto-detects (``,``/``\\t``/``;``/``|``).
    ``has_header``: ``None`` auto-detects whether the first row is a header.
    ``timestamp_format``: explicit ``strptime`` format for the timestamp
    column.  ``None`` auto-detects from a known set of formats.
    ``date_format`` / ``time_format``: formats used when a source stores the
    date and time in separate columns (the date+time pair is combined).
    ``column_map``: explicit source-name -> canonical mapping that bypasses
    alias detection entirely.
    """

    sep: str | None = None
    has_header: bool | None = None
    encoding: str = "utf-8"
    timestamp_format: str | None = None
    date_format: str | None = None
    time_format: str | None = None
    column_map: Mapping[str, str] | None = None


@dataclass(frozen=True)
class LoadResult:
    """Result of a load: canonical frame plus provenance metadata."""

    frame: pd.DataFrame
    source_timezone: str | None  # "UTC (explicit)" or None (unknown/server time)
    warnings: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Raw reading / sniffing
# ---------------------------------------------------------------------------

_CANDIDATE_SEPS = (",", "\t", ";", "|")


def _sniff_sep(path: Path, encoding: str, n_lines: int = 4) -> str:
    with path.open("r", encoding=encoding, errors="replace") as fh:
        lines = [fh.readline() for _ in range(n_lines)]
    lines = [ln.rstrip("\n\r") for ln in lines if ln.strip()]
    if not lines:
        return ","
    best, best_score = _CANDIDATE_SEPS[0], -1
    for sep in _CANDIDATE_SEPS:
        counts = [len(ln.split(sep)) for ln in lines]
        score = min(counts)  # consistent, wide rows win
        if score > best_score:
            best, best_score = sep, score
    return best


def _looks_like_header_row(cells: list[str]) -> bool:
    return any(classify_column(c) is not None for c in cells)


def _read_raw(path: Path, cfg: LoadConfig) -> tuple[pd.DataFrame, bool]:
    sep = cfg.sep if cfg.sep is not None else _sniff_sep(path, cfg.encoding)
    # Read a few lines to detect a header.
    with path.open("r", encoding=cfg.encoding, errors="replace") as fh:
        first = [fh.readline() for _ in range(3)]
    first = [ln for ln in first if ln.strip()]
    has_header = cfg.has_header
    if has_header is None:
        if first:
            cells = [c.strip() for c in first[0].rstrip("\n\r").split(sep)]
            has_header = _looks_like_header_row(cells)
        else:
            has_header = False

    df = pd.read_csv(
        path,
        sep=sep,
        header=0 if has_header else None,
        encoding=cfg.encoding,
        dtype="string",
        keep_default_na=True,
    )
    if df.shape[1] == 0:
        raise SchemaError(f"{path}: file contains no columns")
    return df, has_header


# ---------------------------------------------------------------------------
# Column resolution
# ---------------------------------------------------------------------------


def _resolve_columns(df: pd.DataFrame, has_header: bool, cfg: LoadConfig) -> dict[str, str]:
    """Return a mapping canonical-role -> source-column-name."""
    if cfg.column_map is not None:
        # Validate explicit map: all required OHLC + a timestamp source.
        mapped_roles = set(cfg.column_map.values())
        missing = {"open", "high", "low", "close", "timestamp"} - mapped_roles
        if missing:
            raise SchemaError(f"explicit column_map is missing required role(s): {sorted(missing)}")
        return {v: k for k, v in cfg.column_map.items()}

    cols = list(df.columns)
    if not has_header:
        # Unnamed columns: only the canonical MT5 positional layout is accepted.
        if len(cols) not in (6, 7):
            raise SchemaError(
                f"file has no header and {len(cols)} unnamed columns; only the standard "
                "MetaTrader layout (DATE,TIME,OPEN,HIGH,LOW,CLOSE[,VOLUME]) is supported "
                "for headerless files"
            )
        # Verify first column parses as a date and second as a time to avoid a
        # silent misread of some other headerless layout.
        date_sample = df.iloc[:5, 0].astype(str)
        time_sample = df.iloc[:5, 1].astype(str)
        if (
            date_sample.str.match(r"^\d{4}[./-]\d{1,2}[./-]\d{1,2}").all()
            and time_sample.str.match(r"^\d{1,2}:\d{2}").all()
        ):
            mapping = dict(zip(_MT5_POSITIONAL, cols, strict=False))
            # Drop volume if present; keep date+time -> combine into timestamp later.
            return {
                "date": mapping["date"],
                "time": mapping["time"],
                "open": mapping["open"],
                "high": mapping["high"],
                "low": mapping["low"],
                "close": mapping["close"],
            }
        raise SchemaError(
            "file has no header and its first columns do not match the standard "
            "MetaTrader DATE/TIME layout; refusing to guess the layout"
        )

    # Header present: classify each named column.
    roles: dict[str, str] = {}
    unknown: list[object] = []
    for col in cols:
        role = classify_column(col)
        if role is None:
            unknown.append(col)
            continue
        if role in roles:
            raise SchemaError(
                f"ambiguous mapping: multiple columns map to role {role!r} "
                f"({roles[role]!r} and {col!r})"
            )
        roles[role] = col

    required = {OPEN, HIGH, LOW, CLOSE}
    missing = required - set(roles)
    if missing:
        raise SchemaError(
            f"missing required column(s): {sorted(missing)}; "
            f"recognised columns: {sorted(roles)}; unknown/ignored: {unknown}"
        )
    ts_sources = set(roles) & {"full_timestamp", "date", "time"}
    if not ts_sources:
        raise SchemaError(
            "no timestamp source column found (expected a full timestamp column "
            "or a date+time pair)"
        )
    return roles


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------

_TIMESTAMP_FORMATS: tuple[str, ...] = (
    "%Y.%m.%d %H:%M",
    "%Y.%m.%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y/%m/%d %H:%M",
    "%Y/%m/%d %H:%M:%S",
    "%d.%m.%Y %H:%M",
    "%d.%m.%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%m/%d/%Y %H:%M:%S",
    "%Y%m%d %H:%M:%S",
    "%Y%m%d%H%M%S",
)


def _parse_timestamp_series(values: pd.Series, fmt: str | None) -> pd.Series:
    """Parse a string series into naive ``datetime64[ns]``."""
    s = values.astype("string").str.strip()
    if s.str.len().eq(0).all():
        raise SchemaError("timestamp column is empty")
    if fmt is not None:
        parsed = pd.to_datetime(s, format=fmt, errors="coerce")
        if parsed.isna().all():
            raise SchemaError(f"timestamp column does not parse with format {fmt!r}")
        _raise_unparsed(s, parsed, fmt)
        return parsed
    for candidate in _TIMESTAMP_FORMATS:
        parsed = pd.to_datetime(s, format=candidate, errors="coerce")
        if not parsed.isna().any():
            return parsed
    # Final lenient attempt (ISO 8601 etc.).
    parsed = pd.to_datetime(s, errors="coerce")
    _raise_unparsed(s, parsed, "auto-detected")
    return parsed


def _raise_unparsed(original: pd.Series, parsed: pd.Series, fmt: str) -> None:
    bad = original[parsed.isna()]
    if len(bad) > 0:
        sample = ", ".join(repr(v) for v in bad.head(5))
        raise SchemaError(f"{len(bad)} timestamp(s) failed to parse with {fmt}: {sample}")


def _combine_date_time(
    df: pd.DataFrame, date_col: object, time_col: object, cfg: LoadConfig
) -> pd.Series:
    date_s = df[date_col].astype("string").str.strip()
    time_s = df[time_col].astype("string").str.strip()
    # If the "date" column already holds a full timestamp, that is ambiguous
    # with a separate time column -> fail loudly rather than guess.
    if date_s.str.match(r"^\d{4}[./-]\d{1,2}[./-]\d{1,2}[ T]\d{1,2}:\d{2}").any():
        raise SchemaError(
            f"ambiguous timestamp mapping: column {date_col!r} contains full timestamps "
            "while a separate time column exists; refusing to guess"
        )
    combined = date_s + " " + time_s
    return _parse_timestamp_series(
        combined,
        cfg.timestamp_format,
    )


def _parse_timestamps(
    df: pd.DataFrame, roles: dict[str, str], cfg: LoadConfig
) -> tuple[pd.Series, str | None]:
    """Return (naive UTC-normalised timestamp series, source_timezone)."""
    if "full_timestamp" in roles:
        if {"date", "time"} & set(roles):
            raise SchemaError(
                "ambiguous timestamp mapping: full timestamp column present together "
                "with separate date/time columns"
            )
        ts = _parse_timestamp_series(df[roles["full_timestamp"]], cfg.timestamp_format)
    elif {"date", "time"} <= set(roles):
        ts = _combine_date_time(df, roles["date"], roles["time"], cfg)
    elif "date" in roles:
        # A single date column: accept only if it actually holds full timestamps.
        ts = _parse_timestamp_series(df[roles["date"]], cfg.timestamp_format)
        has_time = ts.dt.hour.ne(0) | ts.dt.minute.ne(0) | ts.dt.second.ne(0)
        if not has_time.any():
            raise SchemaError(
                f"column {roles['date']!r} contains dates only with no time component; "
                "a minute-resolution timestamp cannot be reconstructed"
            )
    elif "time" in roles:
        # A "time" column with no separate date column: accept it only if it
        # actually holds full timestamps (e.g. "2025-01-06 00:00:00").
        col = roles["time"]
        sample = df[col].astype("string").str.strip()
        if sample.str.match(r"^\d{4}[./-]\d{1,2}[./-]\d{1,2}[ T]\d{1,2}:\d{2}").any():
            ts = _parse_timestamp_series(sample, cfg.timestamp_format)
        else:
            raise SchemaError(
                f"column {col!r} provides times only with no date column and no "
                "full-timestamp content; cannot reconstruct minute-resolution "
                "timestamps"
            )
    else:  # pragma: no cover - guarded earlier
        raise SchemaError("no timestamp source column")

    source_timezone: str | None = None
    if hasattr(ts.dt, "tz") and ts.dt.tz is not None:
        # Explicit timezone info: normalise to UTC, then store naive UTC.
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
        source_timezone = "UTC (explicit)"
    else:
        source_timezone = None  # ambiguous (e.g. MT5 server time); not guessed
    return ts.rename(TIMESTAMP), source_timezone


# ---------------------------------------------------------------------------
# Public loading API
# ---------------------------------------------------------------------------


def load_tabular(path: str | Path, config: LoadConfig | None = None) -> LoadResult:
    """Load a CSV/TSV/tabular file into the canonical schema.

    Returns a :class:`LoadResult` with a canonical frame
    (``timestamp, open, high, low, close``) and source-timezone metadata.
    """
    cfg = config or LoadConfig()
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"data file not found: {path}")
    df, has_header = _read_raw(path, cfg)
    roles = _resolve_columns(df, has_header, cfg)
    ts, source_tz = _parse_timestamps(df, roles, cfg)

    ohlc: dict[str, pd.Series] = {}
    warnings: list[str] = []
    for role in (OPEN, HIGH, LOW, CLOSE):
        col = roles[role]
        num = pd.to_numeric(df[col], errors="coerce")
        n_bad = int(num.isna().sum())
        if n_bad:
            warnings.append(f"column {col!r}: {n_bad} non-numeric value(s) coerced to NaN")
        ohlc[role] = num.astype("float64")

    out = pd.DataFrame({"timestamp": ts, **ohlc}, columns=[TIMESTAMP, OPEN, HIGH, LOW, CLOSE])
    return LoadResult(frame=out, source_timezone=source_tz, warnings=tuple(warnings))


def load_parquet(path: str | Path, config: LoadConfig | None = None) -> LoadResult:
    """Load a parquet file into the canonical schema."""
    cfg = config or LoadConfig()
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"data file not found: {path}")
    df = pd.read_parquet(path)
    roles = _resolve_columns(df, has_header=True, cfg=cfg)
    ts, source_tz = _parse_timestamps(df, roles, cfg)
    ohlc = {
        role: pd.to_numeric(df[roles[role]], errors="coerce").astype("float64")
        for role in (OPEN, HIGH, LOW, CLOSE)
    }
    out = pd.DataFrame({"timestamp": ts, **ohlc}, columns=[TIMESTAMP, OPEN, HIGH, LOW, CLOSE])
    return LoadResult(frame=out, source_timezone=source_tz)


def load_any(path: str | Path, config: LoadConfig | None = None) -> LoadResult:
    """Dispatch to the right loader based on file extension."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        return load_parquet(path, config)
    if suffix in (".csv", ".tsv", ".txt", ".dat"):
        return load_tabular(path, config)
    raise SchemaError(f"unsupported data file extension: {suffix!r} for {path}")


def load_many_concat(files: Sequence[str | Path], config: LoadConfig | None = None) -> LoadResult:
    """Load several files (e.g. one per year) and concatenate in time order."""
    if not files:
        raise ValueError("load_many_concat requires at least one file")
    parts: list[pd.DataFrame] = []
    tz: set[str | None] = set()
    warnings: list[str] = []
    for f in files:
        res = load_any(f, config)
        parts.append(res.frame)
        tz.add(res.source_timezone)
        warnings.extend(res.warnings)
    if len(tz) > 1:
        raise SchemaError(f"inconsistent timezone metadata across files: {sorted(tz, key=str)}")
    frame = pd.concat(parts, ignore_index=True)
    frame = (
        frame.drop_duplicates(subset=[TIMESTAMP], keep="first")
        .sort_values(TIMESTAMP, kind="stable")
        .reset_index(drop=True)
    )
    return LoadResult(frame=frame, source_timezone=next(iter(tz)), warnings=tuple(warnings))
