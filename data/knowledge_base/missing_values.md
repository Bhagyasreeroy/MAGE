---
title: Handling Missing Values in Exploratory Data Analysis
doc_type: methodology
section: data_quality
---

# Handling Missing Values

## Diagnosing missingness

Before imputing or dropping anything, classify *why* data is missing:

- **MCAR (Missing Completely At Random)** — the fact that a value is missing
  is unrelated to any observed or unobserved data. Safe to drop or impute
  with simple statistics.
- **MAR (Missing At Random)** — missingness depends on other *observed*
  columns (e.g. income is missing more often for younger respondents).
  Conditional imputation (grouped mean/median, regression imputation) is
  preferred over dropping rows.
- **MNAR (Missing Not At Random)** — missingness depends on the unobserved
  value itself (e.g. high earners refusing to report income). Dropping or
  naive imputation biases the analysis; flag this explicitly in the report.

## Practical thresholds

- Columns with **>60% missing** are usually better dropped than imputed,
  unless the column is analytically critical to the stated goal.
- Columns with **5–40% missing** are good imputation candidates.
- Rows with missingness across **most feature columns** should be dropped
  rather than imputed row-wise.

## Imputation strategies by column type

| Column type | Recommended strategy |
|---|---|
| Numeric, roughly symmetric | Mean imputation |
| Numeric, skewed or has outliers | Median imputation |
| Categorical | Mode imputation, or an explicit "Unknown" category |
| Time series | Forward-fill / interpolation, not global mean |

## Reporting requirement

Any automated pipeline should always report, per column: missing count,
missing percentage, and the imputation strategy applied (if any) — never
silently impute without logging the decision, since it changes the
underlying distribution.
