---
title: Feature Engineering and Scaling
doc_type: methodology
section: data_preparation
---

# Feature Engineering

## Scaling and standardisation

| Method | What it does | Use when |
|---|---|---|
| Standardisation (z-score) | Centres at 0, unit variance | Default for most models |
| Min-max normalisation | Rescales to a fixed range | Bounded inputs are required |
| Robust scaling | Uses median and IQR | Outliers are present and retained |
| No scaling | Leaves features as they are | Tree-based models |

Scaling is required for anything distance- or gradient-based: k-means,
DBSCAN, k-NN, SVM, PCA, logistic regression, neural networks. Without it,
a column measured in thousands overwhelms one measured in units, and the
model is effectively fitted to whichever variable has the largest scale.
Tree-based models split on thresholds and are unaffected.

## Encoding categorical variables

- **One-hot encoding** for nominal categories with low cardinality.
- **Ordinal encoding** only where the categories have a genuine order
  (small/medium/large), since it imposes a distance between levels.
- **Target or frequency encoding** for high-cardinality columns, fitted
  on the training fold only — target encoding leaks the outcome directly
  if fitted before splitting.

## Transformations for skew

Right-skewed positive quantities — revenue, counts, durations — often
behave better after a log or square-root transform: it stabilises
variance, makes relationships more linear, and reduces the influence of
extreme values without discarding them. Use `log1p` when the column
contains zeros. Report any transformation, since it changes how
coefficients and summary statistics must be read.

## Derived features worth trying

Ratios and rates (revenue per unit, error rate) often carry more signal
than either component alone. Date columns should be decomposed into
day-of-week, month, and is-weekend rather than used as raw timestamps.
Interaction terms are usually unnecessary for tree ensembles, which learn
them implicitly, but can help linear models materially.

## Reporting requirement

Every fitted transformation is part of the model. Fit it on the training
fold, apply it to the others, and record which transformations were
applied — an unrecorded transformation makes the analysis irreproducible
and can silently leak the target.
