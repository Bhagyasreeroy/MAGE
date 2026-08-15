---
title: Classification Model Selection
doc_type: methodology
section: supervised_learning
---

# Choosing a Classification Model

## Selection by situation

| Situation | Recommended model | Why |
|---|---|---|
| Interpretability is the priority | Logistic regression | Coefficients read as log-odds |
| Tabular data, mixed feature types | Gradient-boosted trees | Strongest default for tabular problems |
| Small dataset, many features | Regularised logistic regression | Controls variance |
| Non-linear boundary, moderate size | Random forest | Robust, few assumptions |
| A quick baseline is needed | Decision tree or logistic regression | Fast, easy to explain |
| Text or high-dimensional sparse data | Linear SVM or logistic regression | Handles sparsity well |

Start with a simple, interpretable baseline. A gradient-boosted model
that beats logistic regression by two points may not justify the loss of
interpretability; one that beats it by twenty probably does.

## Metrics beyond accuracy

Choose the metric from the cost of the errors, not from convention:

- **Precision** when a false positive is expensive (flagging a legitimate
  transaction as fraud).
- **Recall** when a false negative is expensive (missing a disease).
- **F1** when both matter and classes are imbalanced.
- **ROC-AUC** for ranking quality independent of threshold — but see the
  class-imbalance guidance, since it flatters models on rare positives.

Always inspect the confusion matrix. Two models with identical accuracy
can fail in entirely different, and differently costly, ways.

## Preprocessing that matters

Tree-based models need no feature scaling and handle monotonic
transformations natively. Distance- and gradient-based models —
logistic regression, SVM, k-NN, neural networks — require standardised
features or they will be dominated by whichever column happens to have
the largest units.

Categorical predictors need encoding: one-hot for low-cardinality
nominal columns, ordinal encoding only where the categories genuinely
have an order.

## Reporting requirement

Report the baseline a model is being compared against. "87% accuracy"
means nothing without knowing that always predicting the majority class
would have scored 85%.
