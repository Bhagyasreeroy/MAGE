---
title: Correlation and Association Analysis
doc_type: methodology
section: statistical_analysis
---

# Correlation Analysis

## Choosing a correlation coefficient

| Situation | Recommended measure |
|---|---|
| Two numeric, roughly linear relationship | Pearson correlation |
| Two numeric, monotonic but non-linear | Spearman rank correlation |
| Numeric vs. ordinal categorical | Spearman or Kendall's tau |
| Two nominal categorical | Cramér's V (based on chi-squared) |
| Numeric vs. nominal categorical | Point-biserial (binary) or ANOVA F-stat (multi-class) |

## Interpreting strength

As a rough, goal-agnostic guide for |r|:

- **0.0–0.1**: negligible
- **0.1–0.3**: weak
- **0.3–0.5**: moderate
- **0.5–0.7**: strong
- **0.7–1.0**: very strong

These bands are heuristic — in noisy domains (social science, behavioral
data) a 0.3 correlation may be practically important, while in physical
measurement data anything below 0.9 may be considered weak.

## Correlation is not causation — and not even always relevance

Before recommending a "top correlated feature" to a user:

1. Check for **multicollinearity** among the candidate predictors
   themselves (e.g. via a correlation matrix or VIF) — two highly
   correlated predictors will both look important but carry redundant
   information.
2. Watch for **confounding**: a third variable driving both X and Y
   independently (e.g. store size correlates with both revenue and staff
   count).
3. Watch for **Simpson's paradox**: a correlation can reverse sign when
   the data is split by a categorical group. Always check correlation
   within key subgroups before reporting an aggregate correlation as a
   recommendation.

## Reporting requirement

Report the coefficient value, the method used, and the sample size behind
it — correlations computed on fewer than ~30 paired observations should be
flagged as low-confidence.
