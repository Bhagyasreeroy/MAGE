---
title: Regression Assumptions and Residual Diagnostics
doc_type: methodology
section: supervised_learning
---

# Regression Diagnostics

## Checking linearity before fitting

A linear model assumes the relationship between each predictor and the
target is approximately linear. Screen for this by computing the Pearson
correlation between every numeric feature and the target, and by plotting
the strongest few as scatter plots. A near-zero Pearson coefficient does
not mean "no relationship" — it means no *linear* relationship, and a
U-shaped or thresholded relationship will read as r ≈ 0 while being
strongly predictive.

## The four residual checks

After fitting, plot residuals (actual minus predicted) and check:

| Assumption | Diagnostic | Failure looks like |
|---|---|---|
| Linearity | Residuals vs. fitted values | Curvature, a visible arc |
| Homoscedasticity | Residuals vs. fitted values | A fan or cone widening to the right |
| Normality of residuals | Q-Q plot of residuals | Systematic departure from the diagonal |
| Independence | Residuals vs. row or time order | Runs, drift, or cyclical structure |

Normality applies to the *residuals*, not to the predictors or the
target. Predictors are never required to be normally distributed.

## Common remedies

- **Curvature** — add polynomial terms, or log-transform a right-skewed
  predictor or target.
- **Heteroscedasticity** — log-transform the target, or move to a model
  that does not assume constant variance.
- **Non-independent residuals** — the data is likely time-ordered or
  grouped; use a time-series or mixed-effects approach instead.

## Multicollinearity

Strongly correlated predictors do not harm predictive accuracy but make
individual coefficients unstable and uninterpretable — signs can flip
with a small change to the sample. Check the correlation matrix for
|r| > 0.8 between predictors, or compute the variance inflation factor.
This matters whenever coefficients are being interpreted rather than
merely used to predict.

## Reporting requirement

R² alone is not a diagnostic. Report it alongside the residual checks,
because a high R² with a visibly patterned residual plot means the model
fits well *in aggregate* while being systematically wrong in specific
regions of the input space.
