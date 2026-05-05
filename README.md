# Meta-Learning for Pseudo-Label Utility Prediction

Predicts which of 6 clustering algorithms will generate the best pseudo-labels
for downstream classification on a new tabular dataset — before any clustering is run.

**Core metric**: Label Substitution Efficiency (LSE)
```
LSE = balanced_accuracy(RF on pseudo-labels) / balanced_accuracy(RF on true labels)
```

---

## Project Structure

Each folder answers one research question. Run notebooks in order within each folder.

```
rq1_benchmark/
  01_data_pull.ipynb          Pull 100+ datasets from OpenML (2–10 classes)
  02_lse_computation.ipynb    Compute LSE for all 6 methods on all datasets
  03_method_redundancy.ipynb  Correlation matrix, wins, marginal contribution

rq2_meta_features/
  04_metafeatures.ipynb       Extract Options A/B/C/A+B/A+C + random sanity baseline

rq3_meta_learner/
  05_meta_learner.ipynb       Regression-primary, Architecture A+B, 4 metrics, 3 baselines
  06_shap_analysis.ipynb      TreeExplainer SHAP on ExtraTrees (exact, not approximate)

rq4_generalization/
  07_showcase_eval.ipynb      Held-out showcase evaluation + LSE calibration plots

src/
  lse.py                      LSE computation (balanced accuracy, class_weight='balanced')
  clustering.py               6 pseudo-label methods
  metafeatures.py             Options A/B/C/A+B/A+C (fixed K=4 for B and C)
  hungarian.py                Hungarian alignment
  meta_learner.py             LOO-CV, SRC metric, top-2 accuracy, de Souto baseline

data/
  raw/                        OpenML cache (auto-populated, not committed)
  meta_table/                 Generated CSVs (meta_training.csv, checkpoints, etc.)

outputs/
  figures/                    All plots
  models/                     Saved meta-learner .pkl files

app/
  streamlit_app.py            Upload → Predict → Run → Explain UI
```

---

## Research Questions

| RQ | Question | Key finding |
|----|----------|-------------|
| RQ1 | Which clustering method generates best pseudo-labels? | GMM wins most (~33%), but DBSCAN wins on imbalanced datasets |
| RQ2 | What dataset properties predict method utility? | PCA variance, class entropy, 1-NN accuracy |
| RQ3 | Can a meta-learner predict the best method a priori? | ExtraTrees beats de Souto default-ranking baseline |
| RQ4 | Does it generalize to unseen datasets? | See showcase eval results |

---

## Key Design Decisions

- **Balanced accuracy** throughout — handles imbalanced datasets correctly
- **class_weight='balanced'** in all fixed RFs — prevents majority-class collapse
- **MIN_CLASSES = 2** — binary datasets are valid for clustering-based pseudo-labeling
- **SKIP_IDS excluded at manifest level** — never downloaded, not a silent filter in LSE loop
- **Fixed K=4** for Options B and C — removes dimensionality as a confound
- **TreeExplainer** for SHAP — exact for ExtraTrees, not the KernelExplainer workaround
- **Regression-primary** — predict full LSE vector; recommended method = argmax
- **3 baselines**: always k-means, de Souto default-ranking (the real bar), oracle

---

## Execution Order

```
1. rq1_benchmark/01_data_pull.ipynb
2. rq1_benchmark/02_lse_computation.ipynb    ← slow; checkpointed
3. rq1_benchmark/03_method_redundancy.ipynb
4. rq2_meta_features/04_metafeatures.ipynb  ← slow; checkpointed
5. rq3_meta_learner/05_meta_learner.ipynb   ← slow; LOO-CV
6. rq3_meta_learner/06_shap_analysis.ipynb
7. rq4_generalization/07_showcase_eval.ipynb
```

> Delete checkpoint files in `data/meta_table/` if you change the LSE formula or
> meta-feature code and want a full recomputation from scratch.

---

## Streamlit App

After running all notebooks (step 7 above), launch the interactive platform:

```bash
cd app
streamlit run streamlit_app.py
```

**Workflow inside the app:**
1. Upload any CSV dataset
2. Select the target (class label) column
3. View clusterability check (1-NN separability)
4. See the recommended clustering method + predicted LSE bar chart
5. Expand the SHAP waterfall to understand *why* that method was chosen
6. Click **Run all 6 methods** to verify empirically (takes ~1–3 min)
