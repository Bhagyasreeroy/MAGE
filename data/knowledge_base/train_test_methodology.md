---
title: Train, Validation, and Test Splitting Methodology
doc_type: methodology
section: methodology_pitfalls
---

# Splitting Data for Evaluation

## Choosing a split strategy

| Data shape | Strategy | Note |
|---|---|---|
| Independent rows, ample data | Random 70/15/15 split | Simple and sufficient |
| Independent rows, limited data | k-fold cross-validation | k = 5 or 10 is standard |
| Imbalanced target | Stratified split or stratified k-fold | Preserves class proportions |
| Time-ordered rows | Chronological split | Never shuffle time series |
| Grouped rows (repeat subjects) | Group k-fold | Keeps a group wholly on one side |

## Why three sets, not two

The validation set is used to choose hyperparameters and compare models;
the test set is touched **once**, at the end, to estimate performance.
Selecting a model on the test set makes that score optimistic, because
the choice has been fitted to it. With limited data, use
cross-validation within the training portion and still hold out a
separate final test set.

## Stratification

When the target is imbalanced, a random split can leave the minority
class badly represented — or absent — in a fold. Stratified splitting
preserves the class distribution in every partition, and should be the
default for any classification problem with a ratio beyond about 4:1.

## Grouped data

If several rows describe the same underlying entity — repeat customers,
multiple readings from one sensor, several visits by one patient — a
random split places related rows on both sides and the model appears to
generalise when it has effectively memorised the entity. Split by group.

## Fitting transformations

Every transformation with learned parameters is fitted on training data
only, then applied to validation and test. That covers scaling,
imputation, encoding, feature selection, and any resampling. A pipeline
object that bundles the transformations with the estimator is the
reliable way to enforce this, because it makes the correct order
structural rather than a matter of discipline.

## Reporting requirement

State the split strategy, the proportions, and the random seed. An
evaluation number without its protocol is not reproducible, and small
test sets produce wide confidence intervals that a bare point estimate
conceals.
