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
LSE = Accuracy(trained on pseudo-labels) / Accuracy(trained on ground-truth labels)
```

LSE = 1.0 means pseudo-labels are as good as human annotation.
LSE = 0.7 means we recover 70% of supervised accuracy at zero annotation cost.

The meta-learner predicts which of 6 clustering methods maximises LSE for a new dataset,
using only dataset meta-features as input — no clustering is run at prediction time.

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

```
project/
├── CLAUDE.md                    ← this file
├── data/
│   ├── raw/                     ← OpenML downloads cached here (never commit)
│   ├── meta_table/              ← meta-training CSV: meta-features + LSE targets
│   └── showcase/                ← 8–10 held-out evaluation datasets (never in training)
├── notebooks/
│   ├── 01_data_pull.ipynb       ← Phase 1: OpenML pull + LSE computation loop
│   ├── 02_lse_computation.ipynb ← Phase 1: Run 6 methods, compute LSE, build table
│   ├── 03_metafeatures.ipynb    ← Phase 2: Extract all 3 meta-feature representations
│   ├── 04_meta_learner.ipynb    ← Phase 3: Train kNN + MLP meta-learners, LOO-CV
│   └── 05_shap_analysis.ipynb   ← Phase 4: SHAP beeswarm, feature importance
├── src/
│   ├── lse.py                   ← LSE computation, Hungarian alignment
│   ├── clustering.py            ← all 6 pseudo-label generation methods
│   ├── metafeatures.py          ← Option A/B/C feature extraction
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

**Shape**: ~70 rows (one per OpenML dataset) × (~20 meta-feature columns + 6 LSE target columns + 1 best_method column)

**Structure**:
```
[dataset_id | meta-features ... | LSE_kmeans | LSE_dbscan | LSE_gmm | LSE_agg | LSE_autoenc | LSE_dictlearn | best_method]
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

## Meta-Feature Representations (3 Options for Ablation)

### Option A — Hand-Crafted (Main Approach, deployable features)
Extract label-safe dataset descriptors only. The meta-learner input must be
computable before true labels are available:
- Hopkins statistic (clusterability)
- n_instances, n_features, expected n_classes
- Intrinsic dimensionality (PCA-based)
- Mean pairwise Pearson correlation
- Skewness mean, kurtosis mean
- Feature sparsity and coefficient of variation
- Unsupervised k-means landmarks: estimated imbalance, largest-cluster fraction,
  silhouette, Davies-Bouldin, and inertia per sample

Label-aware quantities such as true class entropy, true imbalance, inter/intra
class ratio, true-label silhouette, and supervised landmarkers are diagnostics
only. Prefix them with `diag_` and exclude them from meta-learner inputs.

### Option B — Autoencoder Bottleneck (Ablation 1)
- MLP autoencoder trained per dataset (PyTorch)
- Aggregate bottleneck activations: mean + variance across all rows
- Output: 2K-dimensional vector per dataset

### Option C — Dictionary Learning Sparse Codes (Ablation 2, Key Novel Input)
- sklearn DictionaryLearning fitted per dataset
- Aggregate sparse codes: mean + variance across all rows
- Output: 2K-dimensional vector per dataset
- This is the primary novel meta-feature contribution

---

## LSE Computation — Exact Steps

Apply identically for every method on every dataset. Never deviate from this procedure:

1. Run clustering on features only (labels masked)
2. Build count matrix: rows = cluster IDs, columns = true class labels
3. Apply Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) for optimal ID→class mapping
4. Convert all cluster IDs to pseudo class labels using the mapping
5. Train a **fixed** Random Forest (default sklearn params — do NOT tune) on pseudo-labeled training set
6. Evaluate on held-out test set using TRUE labels (never pseudo-labels)
7. `LSE = step_6_accuracy / groundtruth_rf_accuracy`

**Critical**: Use the same RF hyperparameters every time. LSE should reflect
pseudo-label quality, not RF quality. Default sklearn settings, fixed random_state=42.

**Groundtruth RF**: same RF trained on the true labels — compute once per dataset and reuse
as the denominator for all 6 LSE calculations.

---

## Dataset Tiers — Never Mix These

| Tier | Size | Source | Purpose |
|------|------|--------|---------|
| Meta-training pool | 50–70 datasets | OpenML API | Train meta-learner. Never used for evaluation or storytelling. |
| Evaluation showcase | 8–10 curated | UCI / Kaggle / OpenML | Test meta-learner on held-out data. Never seen during training. |
| Synthetic (controlled) | 1 generated | sklearn make_classification | Separability ablation (dial 0→1, watch LSE track it). |

**OpenML filters**: 100–100k rows, 2–10 classes, <200 features, no missing values.
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

- **Algorithms**: kNN and MLP (both from sklearn)
- **Validation**: Leave-one-out cross-validation (LOO-CV) — dataset size is modest
- **Primary metric**: Top-1 accuracy (did predicted best method actually rank first?)
- **Secondary metric**: LSE prediction MAE (regression variant)
- **Baselines**:
  - Lower bound: always predict k-means
  - Upper bound: exhaustive search (run all 6, pick best — oracle)
- **SHAP**: Run on trained meta-learner after training, before building ablations

---

## Key Implementation Rules

1. **Fix random seeds everywhere**: `random_state=42` for all sklearn objects, `torch.manual_seed(42)` for PyTorch
2. **Same RF for all LSE**: default sklearn RandomForestClassifier, never tuned
3. **Showcase datasets never touch training**: enforce this with a hard-coded exclusion list
4. **Cache OpenML data**: set cache directory before any API calls
5. **Extract functions from notebooks**: once a function works in a notebook, move it to `src/`
6. **One notebook per phase**: do not put all phases in one notebook
7. **LSE confidence floor**: if predicted LSE < 0.60, flag as high-risk in Streamlit UI and suggest fallback
8. **Hungarian alignment is the only time true labels are seen**: never expose labels to the clustering step

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
