---
title: Distribution Profiling and Visualization Selection
doc_type: methodology
section: visualization
---

# Distribution Profiling

## Shape diagnostics

For every numeric column, compute skewness and kurtosis before choosing a
chart or a downstream statistical test:

- **Skewness near 0**: roughly symmetric — mean/std and Pearson correlation
  are reasonable summary statistics.
- **|Skewness| > 1**: strongly skewed — prefer median/IQR over mean/std,
  and consider a log or Box-Cox transform before applying methods that
  assume normality (e.g. Pearson correlation, z-score outlier detection).
- **High kurtosis (leptokurtic)**: heavy tails — expect more extreme values
  than a normal distribution would predict; IQR-based outlier detection is
  safer than z-score here.

## Chart selection by goal and data shape

| Goal | Data shape | Recommended chart |
|---|---|---|
| Understand single-column distribution | Numeric | Histogram + KDE overlay |
| Understand single-column distribution | Categorical | Bar chart (sorted by frequency) |
| Compare distributions across a group | Numeric x categorical | Box plot or violin plot |
| Relationship between two numeric columns | Numeric x numeric | Scatter plot (add trendline if goal is correlation) |
| Relationship across many numeric columns | Numeric x numeric x N | Correlation heatmap |
| Trend over time | Time-indexed numeric | Line chart, with rolling average for noisy series |
| Geographic goal | Numeric x region | Choropleth |

## Goal-conditioning note

The same dataset should not always produce the same chart set: a goal like
"investigate churn drivers" should prioritize box plots of numeric features
split by the churn flag and a correlation heatmap restricted to
churn-relevant columns, while a goal like "summarize sales trends" should
prioritize time-series decomposition and rolling-average line charts over
the same underlying columns.
