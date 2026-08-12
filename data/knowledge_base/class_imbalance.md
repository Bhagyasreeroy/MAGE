---
title: Handling Imbalanced Target Classes
doc_type: methodology
section: supervised_learning
---

# Class Imbalance

## Diagnosing imbalance

Compute the class distribution of the target column and the **imbalance
ratio** — the count of the majority class divided by the count of the
minority class. Interpretation:

| Imbalance ratio | Reading | Action |
|---|---|---|
| Below 1.5:1 | Effectively balanced | No special handling needed |
| 1.5:1 to 4:1 | Mild imbalance | Prefer balanced metrics; usually no resampling |
| 4:1 to 20:1 | Substantial imbalance | Class weights, and stratify every split |
| Above 20:1 | Severe imbalance | Treat as anomaly detection, not classification |

## Why accuracy is the wrong metric here

With a 99:1 imbalance, a model that always predicts the majority class
scores 99% accuracy while being useless. Report **precision, recall, and
F1 for the minority class**, and prefer the area under the
precision-recall curve to ROC-AUC when the positive class is rare —
ROC-AUC is optimistic under heavy imbalance because the large number of
true negatives dominates the false-positive rate.

## Correcting imbalance

Three families, in rough order of preference:

- **Class weights.** Most estimators accept `class_weight="balanced"`,
  which reweights the loss inversely to class frequency. Cheapest option,
  changes no data, and is usually enough up to about 10:1.
- **Resampling.** Random undersampling of the majority class discards
  data; random oversampling of the minority class risks overfitting to
  duplicated rows. SMOTE synthesises new minority examples by
  interpolation and is generally preferable to naive oversampling.
- **Threshold adjustment.** Leave the model alone and move the decision
  threshold away from 0.5, chosen on a validation set against whichever
  metric reflects the real cost of each error type.

## Reporting requirement

Resampling must be applied **inside** the training fold only, never
before splitting — oversampling first leaks synthetic copies of training
rows into the test set and inflates every score. Always report the
imbalance ratio alongside model performance, so a reader can judge
whether the headline figure is meaningful.
