---
title: Feature Attribution and Model Explainability
doc_type: methodology
section: explainability
---

# Feature Attribution

## What SHAP values mean

SHAP (SHapley Additive exPlanations) assigns each feature a contribution
to an individual prediction, derived from cooperative game theory: the
value is the feature's average marginal contribution across all possible
orderings of the features. Summing a row's SHAP values and the model's
base value reconstructs that row's prediction exactly, which is the
property that makes SHAP *additive* and auditable.

To rank features globally, take the **mean absolute SHAP value** per
feature across rows. Normalising those to sum to 1 expresses each
feature's share of the total explanation.

## Choosing an attribution method

| Method | Use when | Cost |
|---|---|---|
| SHAP TreeExplainer | The model is a tree ensemble | Fast, exact for trees |
| SHAP KernelExplainer | Any model, no structure assumed | Very slow |
| Permutation importance | Any model; a robust fallback | Moderate |
| LIME | A single local explanation is needed | Moderate, approximate |
| PCA loadings | No target exists at all | Cheap, unsupervised |

## Supervised attribution is not variance ranking

PCA loadings describe which features carry the most **variance**, which
is a statement about the data alone. Feature attribution describes which
features drive **the target**, which is a statement about a fitted model.
The two frequently disagree, and reporting one as the other is a common
and consequential error: a high-variance column that is irrelevant to the
outcome will top a PCA ranking and sit near zero in a SHAP ranking.

## Interpreting attributions honestly

An attribution explains the model, not the world. Three caveats belong
next to any reported ranking:

- **Attributions inherit the model's quality.** If the fitted model
  cannot predict the target, its explanations describe noise. Always
  report the model's score alongside the ranking.
- **Correlated features share credit.** Two nearly collinear predictors
  will split the attribution between them, making both look weaker than
  the underlying signal is.
- **Attribution is not causation.** A feature can carry high attribution
  because it is a downstream proxy for the outcome — which is also how
  data leakage first shows itself.
