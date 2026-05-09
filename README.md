# Meta-Learning for Pseudo-Label Utility Prediction

This repository studies whether dataset properties can predict which clustering
algorithm will produce the most useful pseudo-labels for downstream tabular
classification before the clustering algorithms are run.

The core metric is **Label Substitution Efficiency (LSE)**:

```text
LSE = balanced_accuracy(RF trained on pseudo-labels) /
      balanced_accuracy(RF trained on true labels)
```

An LSE near 1 means pseudo-labels support downstream classification nearly as
well as true labels under the fixed evaluation protocol. Values above 1 can
occur from evaluation noise and are kept unclipped.

## Project Layout

```text
rq1_benchmark/
  01_data_pull.ipynb          Build and validate the OpenML dataset manifest
  02_lse_computation.ipynb    Compute LSE for six pseudo-labeling methods
  03_method_redundancy.ipynb  Analyze method correlation, wins, and oracle contribution

rq2_meta_features/
  04_metafeatures.ipynb       Extract meta-feature representations and ablations

rq3_meta_learner/
  05_meta_learner.ipynb       Train/evaluate classification and regression meta-learners
  06_shap_analysis.ipynb      Explain the saved classifier with SHAP

rq4_generalization/
  07_showcase_eval.ipynb      Evaluate on held-out showcase datasets and calibrate LSE

src/
  clustering.py               Six pseudo-label generators
  hungarian.py                Cluster-to-class Hungarian alignment
  lse.py                      LSE evaluation helpers
  metafeatures.py             Meta-feature extractors
  meta_learner.py             LOO-CV, ranking metrics, and baselines

data/meta_table/              Generated manifests, checkpoints, meta-training tables
outputs/figures/              Generated plots and showcase CSV
outputs/models/               Saved meta-learner models
```

## Research Workflow

Run the notebooks in order:

```text
1. rq1_benchmark/01_data_pull.ipynb
2. rq1_benchmark/02_lse_computation.ipynb
3. rq1_benchmark/03_method_redundancy.ipynb
4. rq2_meta_features/04_metafeatures.ipynb
5. rq3_meta_learner/05_meta_learner.ipynb
6. rq3_meta_learner/06_shap_analysis.ipynb
7. rq4_generalization/07_showcase_eval.ipynb
```

The expensive stages are checkpointed. Delete the relevant checkpoint CSVs in
`data/meta_table/` only when you intentionally want a full recomputation after
changing the manifest, preprocessing, LSE formula, clustering code, or
meta-feature code.

## Current Run Summary

The current pipeline uses a verified OpenML manifest with **113 datasets**:

- 30 binary datasets
- 26 datasets with 3-4 classes
- 30 datasets with 5-7 classes
- 27 datasets with 8-10 classes

After LSE computation, the usable meta-training table contains **94 datasets**.
The best-method distribution is:

| Method | Wins |
|--------|------|
| k-means | 28 |
| GMM | 21 |
| Agglomerative | 17 |
| Autoencoder | 13 |
| Dictionary Learning | 10 |
| DBSCAN | 5 |

Method redundancy analysis retains all six methods. The all-method oracle mean
LSE is **0.7307**, and no method fails both retention thresholds.

## Methods Compared

The benchmark evaluates six pseudo-label generators:

- k-means
- DBSCAN with automatic `eps` search and noise reassignment
- Agglomerative clustering with Ward linkage
- Gaussian Mixture Model
- Autoencoder bottleneck clustering
- Dictionary Learning sparse-code clustering

All methods are converted to hard pseudo-labels and evaluated through the same
Hungarian alignment and fixed Random Forest pipeline.

## Meta-Feature Options

Notebook 04 generates the following feature tables:

| Option | Description | Meta-feature dims |
|--------|-------------|-------------------|
| A | Hand-crafted label-free dataset features | 20 |
| B | Autoencoder bottleneck summary | 8 |
| C | Dictionary Learning sparse-code summary | 8 |
| C2 | Multi-scale order-invariant Dictionary Learning features | 27 |
| D | Distance-based features from Ferrari & de Castro-style statistics | 19 |
| A+B | Concatenated A and B | 28 |
| A+C | Concatenated A and C | 28 |
| A+C2 | Concatenated A and C2 | 47 |
| A+D | Concatenated A and D | 39 |
| C+D | Concatenated C and D | 27 |
| Rand | Random sanity baseline | 8 |

Option A is the main deployment-style representation. It is label-free except
for `n_classes`, which is assumed to be supplied as the requested number of
clusters/classes.

## Current Meta-Learner Results

Notebook 05 evaluates both classification and regression formulations with
leave-one-out cross-validation.

Baselines:

| Baseline | Top-1 accuracy | Expected LSE |
|----------|----------------|--------------|
| Always k-means | 0.298 | 0.640 |
| de Souto default ranking (always GMM) | 0.223 | 0.664 |
| Oracle | 1.000 | 0.731 |

Current highlights:

- Best classification result: **C2 + Logistic Regression**, Top-1 = **0.436**
- Saved classifier: **Random Forest on Option A**, Top-1 = **0.426**
- Best primary regression model: **ExtraTrees multi-output on Option A**
  - MAE = **0.1279**
  - Spearman rank correlation = **0.486**
  - Top-1 = **0.457**
  - Expected LSE = **0.682**
- Best regression ablation by MAE: **A+C**, MAE = **0.1233**
- Best regression ablations by Top-1: **A+B** and **A+C2**, Top-1 = **0.468**

Saved models:

```text
outputs/models/meta_clf_optA.pkl
outputs/models/meta_reg_optA.pkl
```

## SHAP Explanation

Notebook 06 uses `shap.TreeExplainer` on the saved Option A Random Forest
classifier. The current global top features are:

1. `knn5_dist_p90`
2. `pca_top3_var`
3. `feature_sparsity`
4. `pairwise_dist_cv`
5. `corr_dispersion`

The SHAP outputs are saved under `outputs/figures/` and
`data/meta_table/shap_values.csv`.

## Showcase Evaluation

Notebook 07 evaluates the saved Option A models on held-out showcase datasets
excluded from meta-training.

Current showcase run:

- Completed datasets: **8/9**
- Top-1 accuracy: **37.5%** (3/8)
- Mean gap vs oracle: **0.051 LSE**
- Mean gain vs k-means: **-0.013 LSE**
- Overall calibration MAE: **0.1289**
- Suggested empirical confidence floor: predicted LSE > **0.52**

Outputs:

```text
outputs/figures/showcase_comparison.csv
outputs/figures/lse_calibration.png
```

## Environment

The notebooks were run with Python 3.11. The main packages used by the code are:

- numpy
- pandas
- scipy
- scikit-learn
- torch
- openml
- matplotlib
- shap
- jupyter / ipykernel

If you recreate the environment, install the dependencies above and run the
notebooks from the repository root or from their existing notebook directories,
as written.

## Reproducibility Notes

- `SEED = 42` is used throughout the shared source modules and notebooks.
- OpenML data is cached under `data/raw/`.
- `data/meta_table/` contains generated CSVs and checkpoints.
- Long-running notebooks should be resumed from their checkpoint files unless
  the underlying methodology has changed.
- The benchmark excludes held-out showcase datasets from the manifest before
  sampling to avoid train/test leakage.
