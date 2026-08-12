---
title: Dimensionality Reduction and PCA
doc_type: methodology
section: unsupervised_learning
---

# Dimensionality Reduction

## When it helps

Reducing dimensionality is worth doing when there are many correlated
numeric features, when a distance-based method is degrading under the
curse of dimensionality, or when a high-dimensional dataset needs to be
visualised in two dimensions. It is *not* a substitute for feature
selection when the goal is interpretability, since components are
combinations of the originals and rarely have a clean meaning.

## PCA

Principal Component Analysis projects the data onto orthogonal
directions of maximum variance. The essentials:

- **Standardise first.** PCA maximises variance, so on unscaled data the
  column with the largest units dominates every component. This is the
  most common PCA mistake.
- **Choose the number of components** from the explained-variance ratio:
  keep enough components to reach a stated threshold, commonly 80–95%,
  or use the elbow of the scree plot.
- **Loadings are not importance.** The loading of a feature on the first
  component describes its contribution to variance, not to any target.

## Choosing a method

| Situation | Method |
|---|---|
| Linear structure, need reproducibility | PCA |
| Visualising clusters in 2D | UMAP or t-SNE |
| Sparse, high-dimensional counts | Truncated SVD |
| Interpretability required | Feature selection, not projection |

t-SNE and UMAP are for visualisation only. Distances between clusters in
their output are not meaningful, and cluster sizes carry no information —
neither should be used as input to a downstream model.

## Identifier columns

Row identifiers, primary keys, and sequential index columns must be
excluded before any variance- or distance-based method. They are
near-unique by construction, so they carry apparent variance while
holding no information, and they distort PCA, k-means, and every
distance-based outlier method.

## Reporting requirement

Report the cumulative explained variance for the components retained. A
two-component projection that captures 35% of the variance is a picture,
not a summary, and any structure it shows should be verified in the
original feature space.
