---
title: Clustering Method Selection
doc_type: methodology
section: unsupervised_learning
---

# Clustering Method Selection

## K-Means

Best when clusters are expected to be roughly spherical, similarly sized,
and the number of clusters *k* can be reasonably estimated (elbow method
on inertia, or silhouette score). Sensitive to feature scale — always
standardize numeric features before fitting. Sensitive to outliers, since
a single extreme point can pull a centroid.

Use the **elbow method** (plot inertia vs. k, look for the bend) combined
with **silhouette score** (closer to 1 is better-separated clusters) to
choose k rather than guessing.

## DBSCAN

Best when:

- The number of clusters is unknown in advance.
- Clusters have irregular (non-spherical) shapes.
- The dataset contains noise/outliers that should be excluded from any
  cluster rather than forced into one (DBSCAN labels them as noise, `-1`).

Requires tuning `eps` (neighborhood radius) and `min_samples`. A k-distance
plot (distance to the k-th nearest neighbor, sorted) helps choose `eps` —
look for the "knee" in the curve.

## Hierarchical (Agglomerative) Clustering

Best for smaller datasets where a dendrogram is useful for exploratory
insight into nested cluster structure, or when the "right" number of
clusters is itself an open question the user wants to explore visually.
Does not scale well beyond a few thousand rows without approximations.

## Decision guide

| Signal in the data | Preferred method |
|---|---|
| Unknown k, noisy data, irregular shapes | DBSCAN |
| Roughly spherical, known/estimable k | K-Means |
| Want a dendrogram / hierarchy of groupings | Agglomerative |

## Reporting requirement

Always report the chosen k (or eps/min_samples), the resulting cluster
sizes, and a separation quality metric (silhouette score) — a clustering
result without a quality metric attached is not a citable recommendation.
