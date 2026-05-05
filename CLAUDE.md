# CLAUDE.md — Meta-Learning for Pseudo-Label Utility Prediction

## Project Overview

This is a **research project** building a meta-learning system that predicts, from dataset
properties alone, which unsupervised clustering algorithm will produce the best pseudo-labels
for downstream supervised classification on tabular data — before any clustering is run.

The system is validated on 8–10 held-out datasets and deployed as an interactive Streamlit
platform (Upload → Predict → Run → Explain).

---

## Core Concept

**Label Substitution Efficiency (LSE)** is our novel target metric:

```
LSE = balanced_accuracy(RF on pseudo-labels) / balanced_accuracy(RF on true labels)
```

Balanced accuracy is used throughout to handle class-imbalanced datasets correctly.
The fixed RF uses `class_weight='balanced'` to prevent majority-class collapse.

LSE = 1.0 means pseudo-labels are as good as human annotation.
LSE = 0.7 means we recover 70% of supervised accuracy at zero annotation cost.

The meta-learner predicts a **vector of 6 LSE values** for a new dataset;
the recommended method is the argmax of that vector. This regression-first framing
is more informative than direct classification: it reveals expected quality, not
just a label.

---

## Project Scope & Constraints

- **Tabular data only** — no images, no text, no time series
- **Classification tasks only** — LSE is only meaningful when cluster structure should
  align with predefined class boundaries. This is not general-purpose exploratory clustering.
- **Supervised classification of a known target concept** — true labels exist but are hidden
  during clustering and only used for Hungarian alignment and LSE evaluation
- **Scope boundary**: LSE is the wrong metric for open-ended discovery tasks. Do not
  generalise findings beyond pseudo-labeling for classification.

---

## Repository Structure

Notebooks are organised by research question — each folder answers one RQ.

```
project/
├── CLAUDE.md
├── README.md
├── rq1_benchmark/               ← RQ1: Which method generates best pseudo-labels?
│   ├── 01_data_pull.ipynb
│   ├── 02_lse_computation.ipynb
│   └── 03_method_redundancy.ipynb
├── rq2_meta_features/           ← RQ2: What dataset properties predict method utility?
│   └── 04_metafeatures.ipynb
├── rq3_meta_learner/            ← RQ3: Can a meta-learner predict the best method?
│   ├── 05_meta_learner.ipynb
│   └── 06_shap_analysis.ipynb
├── rq4_generalization/          ← RQ4: Does it generalize to unseen datasets?
│   └── 07_showcase_eval.ipynb
├── data/
│   ├── raw/                     ← OpenML cache (never commit)
│   └── meta_table/              ← generated CSVs and checkpoints
├── src/
│   ├── lse.py                   ← LSE computation (balanced accuracy)
│   ├── clustering.py            ← all 6 pseudo-label generation methods
│   ├── metafeatures.py          ← Option A/B/C/D feature extraction
│   └── hungarian.py             ← scipy linear_sum_assignment wrapper
├── outputs/
│   ├── figures/                 ← SHAP plots, LSE distributions, ablation charts
│   └── models/                  ← saved meta-learner .pkl files
└── app/
    └── streamlit_app.py         ← Upload → Predict → Run → Explain UI
```

---

## The Meta-Training Table

The central artifact of the project. Located at `data/meta_table/meta_training.csv`.

**Shape**: ~94 rows (one per OpenML dataset, after LSE filtering) × (~20 meta-feature columns + 6 LSE target columns + 1 best_method column)

**Structure**:
```
[dataset_id | meta-features ... | LSE_kmeans | LSE_dbscan | LSE_agg | LSE_gmm | LSE_autoenc | LSE_dictlearn | best_method]
```

**Rules**:
- `best_method` = argmax of the 6 LSE columns (used as classification target)
- LSE targets are also used for regression (predict raw LSE value per method)
- Showcase datasets (8–10) are NEVER included in this table — strict separation
- This table is the entire project — validate it before building anything else

---

## The 6 Pseudo-Label Generation Methods

| Method | Inductive Bias | Label Type |
|--------|---------------|------------|
| k-means | Geometric (spherical) | Hard |
| DBSCAN | Density-based | Hard + noise class |
| Agglomerative (Ward) | Connectivity / linkage | Hard |
| GMM | Probabilistic (Gaussian mixture) | Soft (posterior probs) |
| Autoencoder + k-means | Learned representation | Hard |
| Dictionary Learning | Parts-based (sparse composition) | Soft (sparse codes) |

**Note**: Dictionary Learning appears both as a pseudo-label generator (here) AND as a
meta-feature extractor (Option C). These are two completely separate uses of the technique.

---

## Meta-Feature Representations (Options for Ablation)

**K is fixed at 4** for Options B and C — all datasets produce the same-length vector
regardless of n_classes. This removes dimensionality as a confound.

**Label policy**: Meta-features are computed from unlabeled data only. The user provides k
(number of clusters) at deployment, but no labels are required. All label-dependent features
(`silhouette_true`, `davies_bouldin_true`, landmarkers, `inter_intra_ratio`, `class_entropy`,
`imbalance_ratio`) have been removed from Option A.

| Option | Description | Dims | Role |
|--------|-------------|------|------|
| A | Hand-crafted (~20 label-free features via sklearn/scipy) | ~20 | Main approach |
| B | Autoencoder bottleneck (fixed K=4) | 8 | Ablation |
| C | Dictionary Learning sparse codes (fixed K=4) | 8 | Novel contribution |
| D | Distance-based features (Ferrari & de Castro 2015) | 19 | Novel contribution |
| A+B | A concatenated with B | ~28 | Additive test |
| A+C | A concatenated with C | ~28 | Main novel claim |
| A+D | A concatenated with D | ~39 | Additive test |
| C+D | C concatenated with D | 27 | Clustering-specific head-to-head |
| Random | Random 8-dim features | 8 | Sanity baseline |

If B, C, or D alone cannot beat the random baseline, they add no useful signal.
If A+C beats A alone, dictionary learning adds information beyond hand-crafted features.

---

## LSE Computation — Exact Steps

Apply identically for every method on every dataset:

1. Run clustering on features only (labels masked)
2. Build count matrix: rows = cluster IDs, columns = true class labels
3. Apply Hungarian algorithm for optimal cluster→class mapping
4. Convert cluster IDs to pseudo-class labels using the mapping
5. Train a **fixed** RF (`class_weight='balanced'`, `random_state=42`, no other tuning)
6. Evaluate on held-out test set using TRUE labels
7. `LSE = balanced_accuracy(step 6) / balanced_accuracy(groundtruth RF)`

**Balanced accuracy** is used because raw accuracy is misleading on imbalanced datasets.
**class_weight='balanced'** prevents the RF from collapsing to majority-class prediction.

**DBSCAN edge cases**:
- Noise points (−1): reassigned to nearest non-noise cluster by Euclidean distance
- More clusters than classes: Hungarian maps residuals to majority class (see hungarian.py)
- Fewer than 2 non-noise clusters: falls back to k-means

**Soft-label policy (Option H)**: GMM and Dictionary Learning use argmax (hard labels).
Option S (soft/weighted targets) is a deferred ablation.

---

## Dataset Tiers — Never Mix These

| Tier | Size | Source | Purpose |
|------|------|--------|---------|
| Meta-training pool | ~94 datasets | OpenML API | Train meta-learner. Never used for evaluation or storytelling. |
| Evaluation showcase | 8–10 curated | UCI / Kaggle / OpenML | Test meta-learner on held-out data. Never seen during training. |
| Synthetic (controlled) | 1 generated | sklearn make_classification | Separability ablation (dial 0→1, watch LSE track it). |

**OpenML filters**: 100–100k rows, 2–10 classes, 5–199 features, ≤5% missing values.
**Missing value policy**: Datasets with ≤5% missing values are accepted; missing values are imputed via median (numeric) / mode (categorical) using `SimpleImputer` fit on the training split only, applied before clustering and LSE computation in notebook 02.
**Cache downloads**: `openml.config.cache_directory = "./data/raw"`

---

## Evaluation Showcase Datasets

| Dataset | Why Included |
|---------|-------------|
| Iris | Clean baseline, high separability — validates easy cases |
| Wine | Well-separated chemical profiles |
| Breast Cancer Wisconsin | Medical domain, real annotation cost motivation |
| Heart Disease (UCI) | Mixed features, medium separability |
| Palmer Penguins | Natural cluster structure |
| Diabetes (Pima) | Overlapping classes — expect lower LSE |
| Vehicle Silhouettes | Higher dimensionality |
| Adult Income | Class imbalance, tests DBSCAN advantage |
| Credit Card Fraud | Extreme imbalance (0.17% fraud) |
| Synthetic (controlled) | Separability dial 0→1 |

---

## Meta-Learner Setup

- **Primary framework**: regression — predict the full 6-LSE vector; recommended method = argmax
- **Two architectures**: Architecture A (multi-output) vs Architecture B (per-method, de Souto style)
- **Validation**: LOO-CV — dataset size is modest
- **Primary metric**: per-method LSE MAE
- **All four metrics**: MAE, Spearman Rank Correlation (SRC), Top-1 accuracy, Top-2 + tie zone (0.02)
- **Baselines**:
  - Always k-means (lower bound)
  - **de Souto default-ranking**: always predict the method with highest average LSE across training. This is the real baseline to beat — it uses no meta-features but captures which method wins on average.
  - Oracle (upper bound)
- **SHAP**: `shap.TreeExplainer` on the saved ExtraTrees estimator (exact, not approximate)

---

## Key Implementation Rules

1. **Fix random seeds everywhere**: `random_state=42` for all sklearn objects, `torch.manual_seed(42)` for PyTorch
2. **Same RF for all LSE**: `class_weight='balanced'`, `random_state=42`, no other tuning
3. **Balanced accuracy everywhere**: both groundtruth RF and pseudo-label RF use `balanced_accuracy_score`
4. **MIN_CLASSES = 2**: binary datasets are valid for clustering-based pseudo-labeling
5. **SKIP_IDS excluded at manifest level**: not filtered silently inside LSE loop; every skip ID must have a documented reason string in the dict — bare IDs are not allowed
6. **Showcase datasets never touch training**: enforced by hard-coded ID exclusion in 01_data_pull
7. **Cache OpenML data**: set cache directory before any API calls
8. **Fixed K=4 for Options B and C**: prevents dimensionality from confounding ablation. Option D is always 19-dim.
9. **LSE confidence floor**: determined empirically from showcase calibration plots (not hardcoded 0.60)
10. **Hungarian alignment is the only time true labels are seen during clustering**
11. **MIN_FEATURES = 5**: removes trivial low-dimensional datasets not representative of real tabular ML

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Dataset collection | `openml` Python API |
| Pseudo-label generation | `scikit-learn` (KMeans, DBSCAN, AgglomerativeClustering, GaussianMixture, DictionaryLearning) |
| Deep learning (autoencoder) | `PyTorch` |
| Meta-feature extraction | `pymfe` |
| Meta-learner | `scikit-learn` (KNeighborsClassifier, MLPClassifier) |
| Hungarian alignment | `scipy.optimize.linear_sum_assignment` |
| SHAP analysis | `shap` |
| Platform UI | `streamlit` |
| Notebooks | Jupyter in VSCode |

---

## Research Contributions (for context when writing code)

1. **LSE Benchmark** — first systematic comparison of 6 pseudo-label methods on tabular data using a downstream-utility metric
2. **Predictive Meta-Learner** — predicts best method + expected LSE before any clustering runs
3. **Dictionary Learning as Pseudo-Labeler** — novel soft pseudo-label generator for tabular classification
4. **Dictionary Learning as Meta-feature Input** — novel dataset-level representation via aggregated sparse codes
5. **Meta-feature Representation Ablation** — hand-crafted vs autoencoder vs dictionary learning
6. **SHAP Meta-Explanation** — which dataset properties determine when clustering can replace annotation

---

## Key Papers to Keep in Mind

- **Jilling & Alvarez** — closest prior work on meta-learning for clustering (25 meta-features, 135 datasets). Our lower bound to beat.
- **da Silva et al. (AutoClustering)** — SHAP on meta-learners for clustering; confirms Hopkins, Silhouette landmarker, Davies-Bouldin are top drivers
- **Rivolli et al.** — canonical meta-feature taxonomy, pymfe library
- **CLUBench** — shows clustering performance matrices are low-rank, validating that meta-learning should work
- **Cheung & Thomson** — k-means pseudo-labels → RF classifier pipeline, 99.9% fidelity benchmark
- **PL-CFE** — inter-to-intra similarity ratio as a meta-feature; clustering-friendly embedding spaces improve pseudo-label quality

---

## Common Pitfalls to Avoid

- Do not tune the Random Forest used for LSE computation
- Do not let showcase datasets leak into meta-training (check by dataset ID, not name)
- Do not confuse Dictionary Learning as pseudo-labeler (Section 5) vs meta-feature extractor (Section 6) — they are separate pipelines
- Do not run SHAP before the meta-learner is trained and validated
- Do not interpret LSE as a general clustering quality metric — it only measures downstream classification utility
- Do not use internal metrics (Silhouette, Davies-Bouldin) as substitutes for LSE — they measure different things
