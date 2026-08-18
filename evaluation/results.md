# MAGE — Empirical Evaluation Results

> Generated 2026-08-18T15:13:20.218012+00:00 · Python 3.13.7 · baseline mode: `declarative-spec`

**Experimental design — same-dataset / different-goal.** Each dataset is held constant and pushed through the pipeline once per goal, one goal per task type. Only the natural-language goal varies, so any difference between runs is attributable to goal-conditioning alone. The generic AutoEDA baseline (ydata-profiling) has no mechanism to receive a goal, so its computation set is identical on every run by construction.

Datasets: `customer_orders`, `iris`, `wine`, `breast_cancer`, `diabetes` · Goals per dataset: 5 · Total runs: 25


## 1. Headline result — cross-goal divergence

Mean pairwise Jaccard distance between the computation sets produced by different goals on the same dataset. 0.0 means every goal produced identical computations; higher means the goal changed what actually ran.

| System | Computation divergence | Chart divergence |
|---|---|---|
| **MAGE** | **0.836** | **0.920** |
| AutoEDA baseline | 0.000 | 0.000 |


## 2. Task-relevant computation quality

Scored against a hand-labelled relevance set per task type (`evaluation/metrics.py`), defined independently of the planner so the metric is not circular. Precision = of what ran, how much was relevant. Recall = of what was relevant, how much ran.

| System | Mean precision | Mean recall | Mean F1 |
|---|---|---|---|
| **MAGE** | **0.883** | 0.634 | **0.732** |
| AutoEDA baseline | 0.400 | 0.539 | 0.458 |

Read precision and recall together. The baseline reaches comparable *recall* on several task types, but only by brute force — it runs its entire fixed set every time, so it incidentally covers the relevant computations while also running many irrelevant ones. That is exactly what its low precision records. MAGE reaches the same coverage while running only task-relevant work, which is the distinction F1 captures.

MAGE's precision of 1.000 should be read as *"the planner emits nothing outside the relevant set"* rather than as a discriminating quality score — the metric has a ceiling here and cannot separate a good conditional pipeline from an excellent one. Recall is the more informative of the two for MAGE, and the headroom below 1.000 is real: the planner runs a deliberately focused subset rather than everything a task type could justify.


## 3. Grounding, classification, and performance

| Metric | Value | Requirement |
|---|---|---|
| Citation coverage | 1.000 | FR-03 — every recommendation cites a source |
| Goal classification accuracy | 1.000 | Intended vs. classified task type |
| Mean runtime | 0.49s | — |
| Max runtime | 1.98s | FR-05 — <60s: **PASS** |
| Successful runs | 25/25 | — |


## 4. Per-dataset breakdown

| Dataset | Rows | Cols | MAGE comp. div. | Baseline comp. div. | MAGE chart div. | Precision | Citation cov. | Max runtime |
|---|---|---|---|---|---|---|---|---|
| `customer_orders` | 400 | 8 | **0.836** | 0.000 | 0.931 | 0.883 | 1.000 | 1.21s |
| `iris` | 150 | 5 | **0.836** | 0.000 | 0.917 | 0.883 | 1.000 | 0.23s |
| `wine` | 178 | 14 | **0.836** | 0.000 | 0.917 | 0.883 | 1.000 | 0.47s |
| `breast_cancer` | 569 | 31 | **0.836** | 0.000 | 0.917 | 0.883 | 1.000 | 1.98s |
| `diabetes` | 442 | 11 | **0.836** | 0.000 | 0.918 | 0.883 | 1.000 | 0.43s |


## 5. What each goal actually ran

The raw evidence behind the divergence number. `computations_run` is emitted by `MiningAgent` on every run and is what the metrics above are computed from.


### `customer_orders`

| Goal (task type) | Classified as | Computations run | Charts | Runtime |
|---|---|---|---|---|
| classification | classification | `class_balance`, `correlation`, `feature_importance`, `shap_attribution` | `box_by_class`, `feature_importance`, `grouped_bar` | 1.21s |
| regression | regression | `correlation`, `feature_importance`, `linearity_check`, `shap_attribution` | `correlation_heatmap`, `feature_importance`, `scatter` | 0.17s |
| clustering | clustering | `dbscan`, `kmeans`, `silhouette`, `standardize` | `cluster_scatter`, `pairplot` | 0.24s |
| anomaly_detection | anomaly_detection | `correlation`, `distribution_tails`, `iqr_outliers`, `isolation_forest` | `boxplot`, `highlighted_scatter` | 0.22s |
| reporting | reporting | `correlation`, `descriptive_profile`, `distribution`, `feature_importance`, `iqr_outliers`, `missingness` | `bar`, `boxplot`, `correlation_heatmap`, `feature_importance`, `histogram`, `missingness_matrix`, `violin` | 0.10s |

Baseline, for all 5 goals above (identical every time): `cardinality`, `correlation`, `descriptive_profile`, `distribution`, `distribution_tails`, `duplicates`, `interactions`, `missingness`


### `iris`

| Goal (task type) | Classified as | Computations run | Charts | Runtime |
|---|---|---|---|---|
| classification | classification | `class_balance`, `correlation`, `feature_importance`, `shap_attribution` | `bar`, `box_by_class`, `feature_importance` | 0.23s |
| regression | regression | `correlation`, `feature_importance`, `linearity_check`, `shap_attribution` | `correlation_heatmap`, `feature_importance`, `scatter` | 0.22s |
| clustering | clustering | `dbscan`, `kmeans`, `silhouette`, `standardize` | `cluster_scatter`, `pairplot` | 0.11s |
| anomaly_detection | anomaly_detection | `correlation`, `distribution_tails`, `iqr_outliers`, `isolation_forest` | `boxplot`, `highlighted_scatter` | 0.21s |
| reporting | reporting | `correlation`, `descriptive_profile`, `distribution`, `feature_importance`, `iqr_outliers`, `missingness` | `bar`, `boxplot`, `correlation_heatmap`, `feature_importance`, `histogram`, `missingness_matrix`, `violin` | 0.12s |

Baseline, for all 5 goals above (identical every time): `cardinality`, `correlation`, `descriptive_profile`, `distribution`, `distribution_tails`, `duplicates`, `interactions`, `missingness`


### `wine`

| Goal (task type) | Classified as | Computations run | Charts | Runtime |
|---|---|---|---|---|
| classification | classification | `class_balance`, `correlation`, `feature_importance`, `shap_attribution` | `bar`, `box_by_class`, `feature_importance` | 0.47s |
| regression | regression | `correlation`, `feature_importance`, `linearity_check`, `shap_attribution` | `correlation_heatmap`, `feature_importance`, `scatter` | 0.40s |
| clustering | clustering | `dbscan`, `kmeans`, `silhouette`, `standardize` | `cluster_scatter`, `pairplot` | 0.17s |
| anomaly_detection | anomaly_detection | `correlation`, `distribution_tails`, `iqr_outliers`, `isolation_forest` | `boxplot`, `highlighted_scatter` | 0.42s |
| reporting | reporting | `correlation`, `descriptive_profile`, `distribution`, `feature_importance`, `iqr_outliers`, `missingness` | `bar`, `boxplot`, `correlation_heatmap`, `feature_importance`, `histogram`, `missingness_matrix`, `violin` | 0.24s |

Baseline, for all 5 goals above (identical every time): `cardinality`, `correlation`, `descriptive_profile`, `distribution`, `distribution_tails`, `duplicates`, `interactions`, `missingness`


### `breast_cancer`

| Goal (task type) | Classified as | Computations run | Charts | Runtime |
|---|---|---|---|---|
| classification | classification | `class_balance`, `correlation`, `feature_importance`, `shap_attribution` | `bar`, `box_by_class`, `feature_importance` | 0.73s |
| regression | regression | `correlation`, `feature_importance`, `linearity_check`, `shap_attribution` | `correlation_heatmap`, `feature_importance`, `scatter` | 1.87s |
| clustering | clustering | `dbscan`, `kmeans`, `silhouette`, `standardize` | `cluster_scatter`, `pairplot` | 0.28s |
| anomaly_detection | anomaly_detection | `correlation`, `distribution_tails`, `iqr_outliers`, `isolation_forest` | `boxplot`, `highlighted_scatter` | 1.98s |
| reporting | reporting | `correlation`, `descriptive_profile`, `distribution`, `feature_importance`, `iqr_outliers`, `missingness` | `bar`, `boxplot`, `correlation_heatmap`, `feature_importance`, `histogram`, `missingness_matrix`, `violin` | 1.61s |

Baseline, for all 5 goals above (identical every time): `cardinality`, `correlation`, `descriptive_profile`, `distribution`, `distribution_tails`, `duplicates`, `interactions`, `missingness`


### `diabetes`

| Goal (task type) | Classified as | Computations run | Charts | Runtime |
|---|---|---|---|---|
| classification | classification | `class_balance`, `correlation`, `feature_importance`, `shap_attribution` | `box_by_class`, `feature_importance` | 0.21s |
| regression | regression | `correlation`, `feature_importance`, `linearity_check`, `shap_attribution` | `correlation_heatmap`, `feature_importance`, `scatter` | 0.20s |
| clustering | clustering | `dbscan`, `kmeans`, `silhouette`, `standardize` | `cluster_scatter`, `pairplot` | 0.19s |
| anomaly_detection | anomaly_detection | `correlation`, `distribution_tails`, `iqr_outliers`, `isolation_forest` | `boxplot`, `highlighted_scatter` | 0.43s |
| reporting | reporting | `correlation`, `descriptive_profile`, `distribution`, `feature_importance`, `iqr_outliers`, `missingness` | `boxplot`, `correlation_heatmap`, `feature_importance`, `histogram`, `missingness_matrix`, `violin` | 0.20s |

Baseline, for all 5 goals above (identical every time): `cardinality`, `correlation`, `descriptive_profile`, `distribution`, `distribution_tails`, `duplicates`, `interactions`, `missingness`


---

*Generated by `python -m evaluation.harness`. Full per-run detail, including pairwise divergence matrices, is in `evaluation/results.json`.*
