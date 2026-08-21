"""
agents/temporal.py
───────────────────
Shared time-axis detection.

Lifted verbatim out of ``VisualizationAgent._datetime_series`` when MiningAgent
and OrchestratorAgent came to need the same answer: Mining to decide whether a
seasonal decomposition is even possible, and the orchestrator to decide whether
to *ask* for one. Three copies of "is there a date column here" would drift, and
the drift would be silent — a chart drawn on an axis the decomposition declined
to use, or the reverse.

The behaviour is deliberately unchanged from the original, including its two
judgement calls:

* **Text columns count.** A CSV upload does not arrive with date dtypes — the
  ingestion layer does not infer them — so a `datetime64`-only check would mean
  the time axis is never found outside hand-built test frames.
* **A minimum string length guards against category codes.** "2026-01-04" and
  "04/01/2026" clear it; "3" does not, so an integer-coded category is not
  mistaken for a year.
"""

from __future__ import annotations

import pandas as pd

# How much of a text column must parse as a date, and how long its values must
# typically be before a parse is even attempted.
MIN_DATE_PARSE_RATIO = 0.9
MIN_DATE_TEXT_LENGTH = 6
# Below three distinct timestamps there is no axis worth speaking of.
MIN_DISTINCT_TIMESTAMPS = 3


def detect_time_axis(df: pd.DataFrame) -> tuple[str, pd.Series] | None:
    """The dataset's time axis as ``(column_name, parsed_series)``, or None.

    Parsed datetime columns are taken as-is and win over text candidates.
    """
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col, df[col]

    for col in df.columns:
        series = df[col].dropna()
        if series.empty or not (
            pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])
        ):
            continue
        text = series.astype(str)
        if float(text.str.len().median()) < MIN_DATE_TEXT_LENGTH:
            continue
        try:
            parsed = pd.to_datetime(text, errors="coerce", format="mixed")
        except Exception:  # noqa: BLE001 - an unparseable column is simply not the axis
            continue
        if (
            parsed.notna().mean() < MIN_DATE_PARSE_RATIO
            or parsed.nunique() < MIN_DISTINCT_TIMESTAMPS
        ):
            continue
        return col, parsed.reindex(df.index)
    return None
