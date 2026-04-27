# Teammate Change Summary

This branch fixes the scientific validity issues we found before expanding the
OpenML dataset pool. The old artifacts are still useful for comparison, but the
new `*_fixed.csv` outputs should be used going forward.

## Why We Changed Things

The previous LSE notebook said it used Hungarian alignment, but the actual code
used per-cluster majority mapping. That made the benchmark different from the
research plan. Some Option A meta-features also used true labels, which leaked
information into a predictor that is supposed to work before labels are known.

## Main Code Changes

- Added true Hungarian alignment in `src/hungarian.py`.
- Added corrected LSE computation in `src/lse.py`.
  - Cluster IDs are aligned to class labels.
  - The downstream Random Forest trains on aligned pseudo-class labels.
  - Diagnostics now include majority baseline, LSE lift, mapping collapse, and NMI.
- Moved pseudo-label methods into `src/clustering.py`.
- Added categorical-safe OpenML preprocessing in `src/data_utils.py`.
  - Numeric columns are imputed.
  - Categorical columns are imputed and one-hot encoded.
  - Datasets where ground-truth RF barely beats majority baseline are filtered.
- Refactored Option A in `src/metafeatures.py`.
  - The predictor features are now label-safe.
  - Label-aware values are only optional `diag_*` diagnostics.
- Updated `src/meta_learner.py`.
  - Excludes diagnostic/leakage columns from model inputs.
  - Adds expected LSE and regret utilities.
- Added scripts:
  - `scripts/recompute_lse.py`
  - `scripts/recompute_metafeatures.py`
- Added tests in `tests/test_alignment_lse.py`.
- Added `requirements.txt`.
- Updated notebooks to use general repo-relative paths.
- Updated meta-learner and SHAP notebooks to prefer fixed CSVs when present.

## Timeout Behavior

`scripts/recompute_lse.py` now skips any dataset that takes more than 300 seconds
by default. Timed-out datasets are marked as `TIMEOUT` in the checkpoint and are
excluded from the final usable fixed LSE table.

The timeout can be changed:

```powershell
python scripts\recompute_lse.py --dataset-timeout 180
```

## What To Run

Run these from the repo root:

```powershell
python -m pip install -r requirements.txt
python -c "import numpy, pandas, sklearn, scipy, openml, torch; print('ML env OK')"
python -m unittest tests\test_alignment_lse.py
python scripts\recompute_lse.py
python scripts\recompute_metafeatures.py
```

Then run these notebooks in order:

```text
notebooks/04_meta_learner.ipynb
notebooks/05_shap_analysis.ipynb
```

If `05_shap_analysis.ipynb` says `ModuleNotFoundError: No module named 'shap'`,
run this inside the notebook and restart the kernel:

```python
%pip install shap
```

## New Output Files

The fixed pipeline writes:

- `data/meta_table/meta_training_lse_fixed.csv`
- `data/meta_table/lse_fixed_checkpoint.csv`
- `data/meta_table/diagnostics_lse_fixed.csv`
- `data/meta_table/meta_training_fixed.csv`
- `data/meta_table/meta_training_optA_fixed.csv`
- `data/meta_table/meta_training_optB_fixed.csv`
- `data/meta_table/meta_training_optC_fixed.csv`

Use the fixed files for new analysis. Keep the older CSVs only for comparison.

## Important Note

Do not increase the OpenML pull to 100 datasets yet. First confirm that the
fixed benchmark beats the simple baselines and gives stable results.
