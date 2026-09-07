"""
Meta-learner evaluation utilities.

Provides LOO-CV evaluation for:
  - Classification  : predict best_method label
  - Regression (A)  : multi-output, predict all 6 LSE values jointly
  - Regression (B)  : per-method, 6 separate single-output regressors (de Souto style)

Metrics
-------
  Per-method MAE          : primary regression quality metric
  Spearman Rank Corr      : whole-ranking quality, de Souto 2008 formula
  Top-1 accuracy          : did argmax match true best?
  Top-2 + tie zone        : true best in top-2 OR within 0.02 LSE of predicted top
"""

import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import binomtest, wilcoxon
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVC, SVR

LSE_COLS    = ['LSE_kmeans', 'LSE_dbscan', 'LSE_agg', 'LSE_gmm', 'LSE_autoenc', 'LSE_dictlearn']
META_COLS   = ['dataset_id', 'best_method', 'gt_accuracy']
METHOD_NAMES = ['kmeans', 'dbscan', 'agg', 'gmm', 'autoenc', 'dictlearn']


# ── Data extraction helpers ────────────────────────────────────────────────────

def extract_Xy_clf(df):
    """Return (X, y_str, feature_cols, dataset_ids) for classification."""
    feat_cols = [c for c in df.columns if c not in META_COLS + LSE_COLS]
    X = df[feat_cols].values.astype(float)
    y = df['best_method'].values
    ids = df['dataset_id'].values
    return X, y, feat_cols, ids


def extract_Xy_reg(df):
    """Return (X, Y_lse, feature_cols, dataset_ids) for regression."""
    feat_cols = [c for c in df.columns if c not in META_COLS + LSE_COLS]
    X = df[feat_cols].values.astype(float)
    Y = df[LSE_COLS].values.astype(float)
    ids = df['dataset_id'].values
    return X, Y, feat_cols, ids


# ── Metric functions ───────────────────────────────────────────────────────────

def spearman_rank_correlation(Y_pred, Y_true):
    """
    Spearman Rank Correlation (SRC) following de Souto et al. 2008.

    For each dataset, rank methods by predicted LSE and by true LSE.
    SRC = 1 - 6 * sum(D_i^2) / (P^3 - P)
    where D_i = rank(true_i) - rank(pred_i), P = number of methods.

    Returns mean SRC across all datasets (NaN rows skipped).
    """
    P = Y_pred.shape[1]
    srcs = []
    for i in range(len(Y_pred)):
        true_row = Y_true[i]
        pred_row = Y_pred[i]
        if np.isnan(true_row).any() or np.isnan(pred_row).any():
            continue
        # Rank 1 = highest LSE (best method)
        true_ranks = P + 1 - np.argsort(np.argsort(-true_row)) - 1  # rank from 1
        pred_ranks = P + 1 - np.argsort(np.argsort(-pred_row)) - 1
        D = true_ranks - pred_ranks
        src = 1.0 - 6.0 * np.sum(D ** 2) / (P ** 3 - P)
        srcs.append(src)
    return float(np.mean(srcs)) if srcs else float('nan')


def top2_accuracy_with_tie_zone(pred_best_indices, true_best_indices,
                                 Y_pred, Y_true, tie=0.02):
    """
    Top-2 accuracy with tie zone.

    A prediction is 'acceptable' if EITHER:
      (a) the true best method is among the top-2 predicted methods, OR
      (b) the true best method's LSE is within `tie` of the predicted best's LSE.

    The tie zone prevents penalising predictions where two methods are
    essentially equivalent (e.g. differ by 0.005 LSE).
    """
    n = len(pred_best_indices)
    correct = 0
    for i in range(n):
        true_idx = true_best_indices[i]
        pred_idx = pred_best_indices[i]

        # Top-2 indices (sorted by predicted LSE, descending)
        top2_idx = np.argsort(-Y_pred[i])[:2]

        # Tie zone check
        if np.isnan(Y_true[i, true_idx]) or np.isnan(Y_pred[i, pred_idx]):
            continue
        lse_gap = abs(Y_true[i, true_idx] - Y_pred[i, pred_idx])

        if true_idx in top2_idx or lse_gap <= tie:
            correct += 1

    return float(correct / n) if n > 0 else float('nan')


# ── Baseline functions ─────────────────────────────────────────────────────────

def baseline_always(method_name, y_true):
    """Accuracy of always predicting a fixed method."""
    return float((y_true == method_name).mean())


def default_ranking_baseline(df, lse_cols=None):
    """
    de Souto default-ranking baseline.

    Computes average LSE per method across all datasets. At prediction time,
    always recommends the method with the highest average LSE regardless of the
    new dataset's properties. This is the real baseline to beat — it captures
    which method is best on average without using meta-features at all.

    Returns
    -------
    default_method : str   method with highest average LSE
    avg_lse_vector : dict  {method: avg_lse}
    baseline_expected_lse : float  average LSE of the default method
    """
    if lse_cols is None:
        lse_cols = LSE_COLS
    avg = df[lse_cols].mean()
    default_col = avg.idxmax()
    default_method = default_col.replace('LSE_', '')
    return {
        'default_method': default_method,
        'avg_lse_vector': {c.replace('LSE_', ''): float(v) for c, v in avg.items()},
        'baseline_expected_lse': float(avg.max()),
        'baseline_accuracy': float((df['best_method'] == default_method).mean()),
    }


def oracle_expected_lse(df, lse_cols=None):
    """Mean of the best-method LSE (upper bound: always pick the right method)."""
    if lse_cols is None:
        lse_cols = LSE_COLS
    return float(df[lse_cols].max(axis=1).mean())


def baseline_random(df, lse_cols=None, n_trials=2000, seed=42):
    """
    Uniform-random baseline: for each dataset, pick one of the six methods
    with equal probability and score it. Repeated over `n_trials` draws to
    report a stable expected value plus a 95% interval over the draws,
    rather than a single lucky/unlucky random seed.

    Analytically, expected accuracy is exactly 1/len(lse_cols) since exactly
    one label is correct per dataset; the simulation is kept anyway so the
    reported number carries an interval, not just a point guess.
    """
    if lse_cols is None:
        lse_cols = LSE_COLS
    rng = np.random.default_rng(seed)
    n_methods = len(lse_cols)
    method_names = np.array([c.replace('LSE_', '') for c in lse_cols])
    lse_matrix = df[lse_cols].values
    true_best = df['best_method'].values
    n = len(df)

    accs = np.empty(n_trials)
    exp_lses = np.empty(n_trials)
    for t in range(n_trials):
        choice_idx = rng.integers(0, n_methods, size=n)
        accs[t] = (method_names[choice_idx] == true_best).mean()
        # nanmean: a handful of datasets have NaN for a method that failed to
        # run on them (e.g. DBSCAN degenerate clustering); skip those draws
        # the same way default_ranking_baseline's df.mean() already does,
        # rather than letting one NaN pick poison the whole trial's average.
        exp_lses[t] = np.nanmean(lse_matrix[np.arange(n), choice_idx])

    return {
        'accuracy_mean':      float(accs.mean()),
        'accuracy_ci':        (float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))),
        'expected_lse_mean':  float(exp_lses.mean()),
        'expected_lse_ci':    (float(np.percentile(exp_lses, 2.5)), float(np.percentile(exp_lses, 97.5))),
    }


# ── Statistical significance ───────────────────────────────────────────────────

def bootstrap_ci_diff(mask_a, mask_b, n_boot=10000, seed=42, ci=0.95):
    """
    Paired bootstrap CI for the accuracy difference mean(mask_a) - mean(mask_b),
    resampling datasets (not predictions independently) since both masks come
    from the same LOO-CV datasets in the same order.
    """
    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)
    n = len(mask_a)
    rng = np.random.default_rng(seed)
    idx_all = np.arange(n)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.choice(idx_all, size=n, replace=True)
        diffs[b] = mask_a[idx].mean() - mask_b[idx].mean()
    lo_pct = (1 - ci) / 2 * 100
    hi_pct = (1 + ci) / 2 * 100
    return {
        'observed_diff': float(mask_a.mean() - mask_b.mean()),
        'ci_lo':         float(np.percentile(diffs, lo_pct)),
        'ci_hi':         float(np.percentile(diffs, hi_pct)),
    }


def mcnemar_exact_test(mask_a, mask_b):
    """
    Exact McNemar's test on paired correct/incorrect masks from two classifiers
    evaluated on the same LOO-CV datasets. Tests whether the two models'
    disagreements are asymmetric (one tends to be right when the other is
    wrong more often than the reverse) rather than whether either is "good".
    """
    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)
    n10 = int(np.sum(mask_a & ~mask_b))   # a correct, b incorrect
    n01 = int(np.sum(~mask_a & mask_b))   # a incorrect, b correct
    n_discordant = n10 + n01
    if n_discordant == 0:
        p_value = 1.0
    else:
        p_value = binomtest(min(n10, n01), n_discordant, 0.5, alternative='two-sided').pvalue
    return {'n10': n10, 'n01': n01, 'n_discordant': n_discordant, 'p_value': float(p_value)}


def wilcoxon_paired_errors(err_a, err_b):
    """
    Wilcoxon signed-rank test on paired per-dataset absolute errors between two
    regression configurations evaluated on the same datasets in the same order.
    Pairs with a NaN in either array (failed LSE for that dataset) are dropped.
    """
    err_a = np.asarray(err_a, dtype=float)
    err_b = np.asarray(err_b, dtype=float)
    valid = ~(np.isnan(err_a) | np.isnan(err_b))
    a, b = err_a[valid], err_b[valid]
    if len(a) == 0 or np.allclose(a, b):
        return {'statistic': float('nan'), 'p_value': 1.0, 'n': int(valid.sum()), 'mean_diff': 0.0}
    stat, p = wilcoxon(a, b)
    return {'statistic': float(stat), 'p_value': float(p), 'n': int(valid.sum()),
            'mean_diff': float((a - b).mean())}


# ── LOO-CV: Classification ─────────────────────────────────────────────────────

def loo_classify(pipeline, X, y):
    """
    LOO-CV classification.

    Returns
    -------
    dict with:
        accuracy      float   Top-1 accuracy
        preds         array   Predicted labels (original string values)
        trues         array   True labels
        correct_mask  array   bool per sample
    """
    le = LabelEncoder().fit(y)
    y_enc = le.transform(y)

    loo = LeaveOneOut()
    preds_enc, trues_enc = [], []
    for train_idx, test_idx in loo.split(X):
        clf = clone(pipeline)
        clf.fit(X[train_idx], y_enc[train_idx])
        preds_enc.append(clf.predict(X[test_idx])[0])
        trues_enc.append(y_enc[test_idx[0]])

    preds_enc = np.array(preds_enc)
    trues_enc = np.array(trues_enc)
    preds = le.inverse_transform(preds_enc)
    trues = le.inverse_transform(trues_enc)
    return {
        'accuracy': float((preds == trues).mean()),
        'preds': preds,
        'trues': trues,
        'correct_mask': preds == trues,
    }


# ── LOO-CV: Regression (Architecture A — multi-output) ────────────────────────

def loo_regress(pipeline, X, Y_lse, df_lse):
    """
    LOO-CV multi-output regression; derive predicted best method by argmax.

    Parameters
    ----------
    Y_lse   : (n, 6) array of true LSE values
    df_lse  : DataFrame with LSE_COLS and 'best_method'

    Returns
    -------
    dict with MAE, SRC, Top-1, Top-2+tie, argmax accuracy, expected LSE
    """
    loo = LeaveOneOut()
    Y_pred = np.zeros_like(Y_lse, dtype=float)
    nan_mask = np.isnan(Y_lse).any(axis=1)

    for train_idx, test_idx in loo.split(X):
        train_valid = train_idx[~nan_mask[train_idx]]
        reg = clone(pipeline)
        reg.fit(X[train_valid], Y_lse[train_valid])
        Y_pred[test_idx] = reg.predict(X[test_idx])

    valid = ~nan_mask
    mae_per_col = np.abs(Y_pred[valid] - Y_lse[valid]).mean(axis=0)

    pred_best_idx = np.argmax(Y_pred, axis=1)
    true_best_idx = np.argmax(Y_lse, axis=1)
    pred_best = np.array([METHOD_NAMES[i] for i in pred_best_idx])
    true_best = df_lse['best_method'].values

    src = spearman_rank_correlation(Y_pred[valid], Y_lse[valid])
    top1_acc = float((pred_best == true_best).mean())
    top2_acc = top2_accuracy_with_tie_zone(
        pred_best_idx, true_best_idx, Y_pred, Y_lse
    )

    expected_lse = np.array([
        Y_lse[i, pred_best_idx[i]]
        for i in range(len(Y_lse)) if not nan_mask[i]
    ])

    abs_err_per_dataset = np.full(len(Y_lse), np.nan)
    abs_err_per_dataset[valid] = np.abs(Y_pred[valid] - Y_lse[valid]).mean(axis=1)

    return {
        'mae_per_col':     mae_per_col,
        'mae_mean':        float(mae_per_col.mean()),
        'src':             src,
        'top1_accuracy':   top1_acc,
        'top2_accuracy':   top2_acc,
        'argmax_accuracy': top1_acc,          # alias for backward compat
        'predicted_lse':   Y_pred,
        'expected_lse':    float(np.nanmean(expected_lse)),
        'abs_err_per_dataset': abs_err_per_dataset,
    }


# ── LOO-CV: Regression (Architecture B — per-method, de Souto style) ──────────

def _unwrap_multioutput(pipeline):
    """Replace any MultiOutputRegressor(est) step with est directly.

    Architecture B fits single-output regressors per method column, so
    MultiOutputRegressor wrappers (used to make SVR work with Architecture A)
    must be peeled off before per-column fitting.
    """
    new_steps = []
    for name, step in pipeline.steps:
        if isinstance(step, MultiOutputRegressor):
            new_steps.append((name, clone(step.estimator)))
        else:
            new_steps.append((name, step))
    return Pipeline(new_steps)


def loo_regress_per_method(pipeline_template, X, Y_lse, df_lse):
    """
    LOO-CV regression with 6 separate single-output regressors (one per method).

    Following de Souto et al. 2008: each method gets its own specialized regressor.
    At inference, combine the 6 predictions into a vector and take argmax.

    Parameters
    ----------
    pipeline_template : sklearn Pipeline (single-output)
    X                 : (n, p) feature matrix
    Y_lse             : (n, 6) true LSE matrix
    df_lse            : DataFrame with 'best_method' column

    Returns
    -------
    Same dict structure as loo_regress()
    """
    pipeline_template = _unwrap_multioutput(pipeline_template)

    loo = LeaveOneOut()
    Y_pred = np.zeros_like(Y_lse, dtype=float)
    nan_mask = np.isnan(Y_lse).any(axis=1)

    for train_idx, test_idx in loo.split(X):
        train_valid = train_idx[~nan_mask[train_idx]]
        for col_idx in range(Y_lse.shape[1]):
            y_col = Y_lse[train_valid, col_idx]
            valid_col = ~np.isnan(y_col)
            if valid_col.sum() < 2:
                Y_pred[test_idx, col_idx] = np.nanmean(Y_lse[:, col_idx])
                continue
            reg = clone(pipeline_template)
            reg.fit(X[train_valid][valid_col], y_col[valid_col])
            Y_pred[test_idx, col_idx] = reg.predict(X[test_idx])[0]

    valid = ~nan_mask
    mae_per_col = np.abs(Y_pred[valid] - Y_lse[valid]).mean(axis=0)

    pred_best_idx = np.argmax(Y_pred, axis=1)
    true_best_idx = np.argmax(Y_lse, axis=1)
    pred_best = np.array([METHOD_NAMES[i] for i in pred_best_idx])
    true_best = df_lse['best_method'].values

    src = spearman_rank_correlation(Y_pred[valid], Y_lse[valid])
    top1_acc = float((pred_best == true_best).mean())
    top2_acc = top2_accuracy_with_tie_zone(
        pred_best_idx, true_best_idx, Y_pred, Y_lse
    )

    expected_lse = np.array([
        Y_lse[i, pred_best_idx[i]]
        for i in range(len(Y_lse)) if not nan_mask[i]
    ])

    abs_err_per_dataset = np.full(len(Y_lse), np.nan)
    abs_err_per_dataset[valid] = np.abs(Y_pred[valid] - Y_lse[valid]).mean(axis=1)

    return {
        'mae_per_col':     mae_per_col,
        'mae_mean':        float(mae_per_col.mean()),
        'src':             src,
        'top1_accuracy':   top1_acc,
        'top2_accuracy':   top2_acc,
        'argmax_accuracy': top1_acc,
        'predicted_lse':   Y_pred,
        'expected_lse':    float(np.nanmean(expected_lse)),
        'abs_err_per_dataset': abs_err_per_dataset,
    }


# ── Reporting ──────────────────────────────────────────────────────────────────

def report_clf_results(label, results):
    acc = results['accuracy']
    pred_dist = Counter(results['preds'])
    print(f"  {label:45s}  acc={acc:.3f}  dist={dict(pred_dist)}")


def report_reg_results(label, results):
    print(f"  {label:45s}  MAE={results['mae_mean']:.4f}  "
          f"SRC={results['src']:.3f}  "
          f"Top1={results['top1_accuracy']:.3f}  "
          f"Top2={results['top2_accuracy']:.3f}  "
          f"E[LSE]={results['expected_lse']:.3f}")


# ── Model candidates ───────────────────────────────────────────────────────────

def build_classifier_candidates(best_k=1, random_state=42):
    return {
        'kNN': Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
            ('clf', KNeighborsClassifier(n_neighbors=best_k)),
        ]),
        'LogReg': Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
            ('clf', LogisticRegression(C=0.5, max_iter=5000,
                                       class_weight='balanced',
                                       random_state=random_state)),
        ]),
        'SVC-RBF': Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
            ('clf', SVC(kernel='rbf', C=1.0, gamma='scale',
                        class_weight='balanced',
                        random_state=random_state)),
        ]),
        'ExtraTrees': Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('clf', ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2,
                                         class_weight='balanced_subsample',
                                         random_state=random_state)),
        ]),
        'RF': Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('clf', RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                            class_weight='balanced_subsample',
                                            random_state=random_state)),
        ]),
        'MLP': Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
            ('clf', MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-2,
                                   max_iter=1500, random_state=random_state)),
        ]),
    }


def build_regressor_candidates(best_k=1, random_state=42):
    return {
        'kNN': Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
            ('reg', KNeighborsRegressor(n_neighbors=best_k)),
        ]),
        'RF': Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('reg', RandomForestRegressor(n_estimators=400, min_samples_leaf=2,
                                           random_state=random_state)),
        ]),
        'ExtraTrees': Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('reg', ExtraTreesRegressor(n_estimators=400, min_samples_leaf=2,
                                         random_state=random_state)),
        ]),
        'MLP': Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
            ('reg', MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-2,
                                  max_iter=1500, random_state=random_state)),
        ]),
        'SVR-RBF': Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
            ('reg', MultiOutputRegressor(
                SVR(kernel='rbf', C=1.0, epsilon=0.05, gamma='scale')
            )),
        ]),
    }
