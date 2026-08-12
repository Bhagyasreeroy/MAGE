---
title: Data Leakage and How to Avoid It
doc_type: methodology
section: methodology_pitfalls
---

# Data Leakage

Data leakage is the use of information during training that would not be
available at prediction time. It is the single most common cause of a
model that scores excellently in evaluation and fails in deployment, and
it is dangerous precisely because its symptom is a *good* result.

## The three kinds

| Kind | What happens | Typical symptom |
|---|---|---|
| Target leakage | A feature encodes the outcome | One feature dominates; near-perfect accuracy |
| Train-test contamination | Test rows influence training | Test score far above real-world performance |
| Temporal leakage | Future information predicts the past | Excellent backtest, poor live performance |

## Target leakage

A feature is computed after, or as a consequence of, the outcome. Classic
examples: `days_since_cancellation` when predicting cancellation,
`total_paid` when predicting default, a `case_closed_reason` field when
predicting case resolution.

The tell is a single feature with implausibly high attribution combined
with near-perfect accuracy. Ask of every top-ranked feature: *would this
value actually be known at the moment the prediction is needed?* If not,
drop it.

## Train-test contamination

Any statistic computed over the full dataset before splitting leaks the
test set into training. The rule is that **every fitted transformation
must be fitted on the training fold only** and then applied to the test
fold — scaling parameters, imputation values, encoding categories,
feature selection, and resampling all qualify. Fitting a scaler on all
the data before splitting is the most frequent instance.

Duplicate rows split across train and test cause the same problem: check
for and remove duplicates before splitting.

## Temporal leakage

With time-ordered data, a random split lets the model train on the future
and predict the past. Split by time instead, and ensure any rolling or
aggregate feature uses only a backward-looking window.

## Detection heuristics

- Accuracy above ~99% on a non-trivial problem is a leakage hypothesis,
  not a success.
- A single feature carrying most of the attribution warrants inspection.
- A large gap between cross-validation and held-out performance suggests
  contamination in the fitting pipeline.
