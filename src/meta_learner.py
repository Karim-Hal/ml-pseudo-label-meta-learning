"""
Meta-learner evaluation utilities for Phase 4.

Provides LOO-CV evaluation for both classification (predict best_method)
and regression (predict all 6 LSE values).
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import LabelEncoder

LSE_COLS = ['LSE_kmeans', 'LSE_dbscan', 'LSE_agg', 'LSE_gmm', 'LSE_autoenc', 'LSE_dictlearn']
META_COLS = [
    'dataset_id',
    'best_method',
    'gt_accuracy',
    'majority_accuracy',
    'gt_lift_over_majority',
    'failure_reason',
]
METHOD_NAMES = ['kmeans', 'dbscan', 'agg', 'gmm', 'autoenc', 'dictlearn']
DIAGNOSTIC_PREFIXES = ('diag_',)


def extract_Xy_clf(df):
    """
    Return (X, y_str, feature_cols, dataset_ids) for classification.

    X      : float array (n_datasets, n_features)
    y_str  : string array of best_method labels
    """
    feat_cols = [
        c for c in df.columns
        if c not in META_COLS + LSE_COLS
        and not any(c.startswith(prefix) for prefix in DIAGNOSTIC_PREFIXES)
    ]
    X = df[feat_cols].values.astype(float)
    y = df['best_method'].values
    ids = df['dataset_id'].values
    return X, y, feat_cols, ids


def extract_Xy_reg(df):
    """
    Return (X, Y_lse, feature_cols, dataset_ids) for multi-output regression.

    Y_lse  : float array (n_datasets, 6) of LSE values
    """
    feat_cols = [
        c for c in df.columns
        if c not in META_COLS + LSE_COLS
        and not any(c.startswith(prefix) for prefix in DIAGNOSTIC_PREFIXES)
    ]
    X = df[feat_cols].values.astype(float)
    Y = df[LSE_COLS].values.astype(float)
    ids = df['dataset_id'].values
    return X, Y, feat_cols, ids


def loo_classify(pipeline, X, y):
    """
    LOO-CV classification.

    y can be a string array — labels are integer-encoded internally so that
    MLP early_stopping (which calls np.isnan on predictions) doesn't crash.

    Returns
    -------
    results : dict with keys:
        accuracy      float   Top-1 accuracy
        preds         list    Predicted labels (original string values)
        trues         list    True labels (original string values)
        correct_mask  np.ndarray  bool per sample
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


def loo_regress(pipeline, X, Y_lse, df_lse):
    """
    LOO-CV multi-output regression, then derive predicted best method by argmax.

    Parameters
    ----------
    Y_lse   : (n, 6) array of true LSE values
    df_lse  : DataFrame with LSE_COLS (same order as Y_lse rows)

    Returns
    -------
    results : dict with keys:
        mae_per_col     np.ndarray (6,) MAE per LSE column
        mae_mean        float
        argmax_accuracy float  accuracy of argmax-predicted best method
        predicted_lse   np.ndarray (n, 6)
        expected_lse    float  mean LSE of the predicted method per dataset
    """
    loo = LeaveOneOut()
    Y_pred = np.zeros_like(Y_lse, dtype=float)

    nan_mask = np.isnan(Y_lse).any(axis=1)  # skip rows with any NaN target

    for train_idx, test_idx in loo.split(X):
        train_valid = train_idx[~nan_mask[train_idx]]
        reg = clone(pipeline)
        reg.fit(X[train_valid], Y_lse[train_valid])
        Y_pred[test_idx] = reg.predict(X[test_idx])

    # MAE only on rows where all targets are valid
    valid = ~nan_mask
    mae_per_col = np.abs(Y_pred[valid] - Y_lse[valid]).mean(axis=0)

    # Argmax accuracy
    pred_best_idx = np.argmax(Y_pred, axis=1)
    pred_best = np.array([METHOD_NAMES[i] for i in pred_best_idx])
    true_best = df_lse['best_method'].values
    argmax_acc = float((pred_best == true_best).mean())

    # Expected LSE/regret: actual utility of predicted method versus oracle.
    expected_lse = np.array([
        Y_lse[i, pred_best_idx[i]]
        for i in range(len(Y_lse))
        if not nan_mask[i]
    ])
    oracle_lse = np.nanmax(Y_lse[valid], axis=1)
    regret = oracle_lse - expected_lse

    return {
        'mae_per_col': mae_per_col,
        'mae_mean': float(mae_per_col.mean()),
        'argmax_accuracy': argmax_acc,
        'predicted_lse': Y_pred,
        'expected_lse': float(np.nanmean(expected_lse)),
        'mean_regret': float(np.nanmean(regret)),
    }


def classification_utility(results, df_lse):
    """
    Compute expected LSE and regret for a classification meta-learner result.

    Top-1 accuracy can be harsh when several methods are nearly tied. This
    reports the downstream utility actually obtained by the predicted method.
    """
    pred_methods = np.asarray(results['preds'])
    true_methods = np.asarray(results['trues'])
    y_cols = df_lse[LSE_COLS].values.astype(float)
    method_to_idx = {name: i for i, name in enumerate(METHOD_NAMES)}

    chosen_lse = []
    oracle_lse = []
    valid_mask = []
    for i, pred in enumerate(pred_methods):
        idx = method_to_idx.get(pred)
        row = y_cols[i]
        if idx is None or np.isnan(row[idx]) or np.isnan(row).all():
            valid_mask.append(False)
            continue
        valid_mask.append(True)
        chosen_lse.append(row[idx])
        oracle_lse.append(np.nanmax(row))

    chosen_lse = np.asarray(chosen_lse, dtype=float)
    oracle_lse = np.asarray(oracle_lse, dtype=float)
    regret = oracle_lse - chosen_lse

    return {
        'top1_accuracy': float((pred_methods == true_methods).mean()),
        'expected_lse': float(np.nanmean(chosen_lse)),
        'oracle_lse': float(np.nanmean(oracle_lse)),
        'mean_regret': float(np.nanmean(regret)),
        'valid_utility_rows': int(np.sum(valid_mask)),
    }


def baseline_always(method_name, y_true):
    """Accuracy of always predicting a fixed method."""
    return float((y_true == method_name).mean())


def oracle_expected_lse(df):
    """Mean of the best-method LSE (upper bound: always pick the right method)."""
    return float(df[LSE_COLS].max(axis=1).mean())


def report_clf_results(label, results):
    """Print a compact classification result line."""
    acc = results['accuracy']
    from collections import Counter
    pred_dist = Counter(results['preds'])
    print(f"  {label:40s}  Top-1 acc = {acc:.3f}  pred_dist={dict(pred_dist)}")
