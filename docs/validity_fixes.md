# Scientific Validity Fixes

This repo now has reusable source code for the main validity fixes instead of
keeping the benchmark logic only inside notebooks.

## What Changed

- `src/hungarian.py` implements true Hungarian cluster-to-class alignment with
  `scipy.optimize.linear_sum_assignment`.
- `src/lse.py` computes LSE by training the downstream Random Forest on aligned
  pseudo-class labels, not raw cluster IDs.
- `src/clustering.py` contains the six pseudo-label methods used by the LSE
  benchmark.
- `src/data_utils.py` keeps categorical predictors through imputation and
  one-hot encoding, and provides a ground-truth-vs-majority quality gate.
- `src/metafeatures.py` makes Option A deployable by removing label-dependent
  features from the predictor input. Label-aware features are available only as
  `diag_*` diagnostics.
- `src/meta_learner.py` excludes `diag_*` columns from model inputs and can
  report expected LSE and regret in addition to top-1 accuracy.

## Recommended Rerun Order

Run these from the repo root inside the project environment that has the ML
dependencies installed. The paths are repo-relative, so they work on any
teammate's machine after cloning the repo:

```bash
python -m pip install -r requirements.txt
python -c "import numpy, pandas, sklearn, scipy, openml, torch; print('ML env OK')"
python -m unittest tests/test_alignment_lse.py
python scripts/recompute_lse.py
python scripts/recompute_metafeatures.py
```

On Windows PowerShell, the same commands are:

```powershell
python -m pip install -r requirements.txt
python -c "import numpy, pandas, sklearn, scipy, openml, torch; print('ML env OK')"
python -m unittest tests\test_alignment_lse.py
python scripts\recompute_lse.py
python scripts\recompute_metafeatures.py
```

Then rerun the meta-learner and SHAP notebooks against the `*_fixed.csv` tables,
or update the notebook CSV paths to point at:

- `data/meta_table/meta_training_fixed.csv`
- `data/meta_table/meta_training_optA_fixed.csv`
- `data/meta_table/meta_training_optB_fixed.csv`
- `data/meta_table/meta_training_optC_fixed.csv`

Do not expand the OpenML pull to 100 datasets until the fixed tables produce a
stable result and the simple baselines are beaten.
