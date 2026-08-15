---
title: Time Series Trend and Seasonality Analysis
doc_type: methodology
section: temporal_analysis
---

# Time Series Analysis

## Recognising a time series

A dataset is temporal when rows carry a date or timestamp and their order
is meaningful. Once that is true, most default EDA choices change: rows
are no longer independent, random splits become invalid, and summary
statistics computed over the whole span can conceal drift entirely.

## Decomposition

Seasonal decomposition separates an observed series into three parts:

- **Trend** — the long-run direction, after short-term fluctuation is
  smoothed away.
- **Seasonality** — a repeating cycle of fixed period (day of week, month
  of year).
- **Residual** — what neither explains, which is where anomalies show up.

Use an **additive** model when the seasonal swing is roughly constant in
absolute size, and a **multiplicative** model when the swing grows with
the level of the series. Decomposition requires a regular frequency, so
resample or interpolate gaps first.

## Stationarity

Many time-series methods assume a stationary series — constant mean and
variance over time. Test with the Augmented Dickey-Fuller test. If it is
non-stationary, difference the series (subtract the previous value), or
remove the trend explicitly. A log transform first will stabilise
variance that grows with the level.

## Charts for temporal goals

| Goal | Chart |
|---|---|
| Show the trend over time | Line chart, with a rolling average for noisy series |
| Expose a repeating cycle | Seasonal subseries plot, or period-over-period overlay |
| Locate temporal anomalies | Line chart of the decomposition residual |
| Compare groups over time | Multi-series line chart, one line per group |

Bar charts and histograms discard the ordering that makes the data a time
series, and should not be the primary view for a temporal goal.

## Autocorrelation

Plot the autocorrelation function to see how strongly a value depends on
earlier ones. A large spike at lag 7 on daily data indicates weekly
seasonality; slow decay across many lags indicates a trend rather than a
cycle.

## Reporting requirement

State the frequency, the span covered, and how gaps were handled.
Never validate a temporal model on a random split — split chronologically
so the evaluation reflects predicting forward, which is the task that
will actually be asked of it.
