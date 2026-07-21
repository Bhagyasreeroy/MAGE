---
title: Outlier Detection Methods for Tabular Data
doc_type: methodology
section: statistical_analysis
---

# Outlier Detection

## IQR (Interquartile Range) method

For a single numeric column, compute Q1 (25th percentile) and Q3 (75th
percentile). The IQR is `Q3 - Q1`. A value is flagged as an outlier if it
falls outside:

```
[Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]
```

This is a good default for **univariate** outlier screening on
approximately unimodal distributions. It is robust to non-normality but
can over-flag values in heavily skewed distributions — consider a
log-transform first for right-skewed data (revenue, counts, durations).

## Z-score method

Flag values where `|x - mean| / std > 3`. Only appropriate for
approximately normal distributions; sensitive to the outliers it is trying
to detect (the mean/std themselves get pulled by extreme values), so IQR is
generally preferred for exploratory work.

## Isolation Forest

For **multivariate** outlier detection (an observation that is normal on
every individual feature but anomalous in combination), use Isolation
Forest. It isolates observations by randomly partitioning feature space;
anomalies require fewer partitions to isolate and get a higher anomaly
score. Preferred when:

- There are more than ~3 numeric features and interactions between them
  matter.
- The dataset is large enough (Isolation Forest scales well, O(n log n)).
- You cannot assume any particular distribution shape.

## Reporting requirement

Outlier detection should always report *how many* points were flagged and
*which method* was used — never silently drop outlier rows without
surfacing the decision, since legitimate rare events (fraud, peak demand)
are frequently the most analytically interesting rows in the dataset.
