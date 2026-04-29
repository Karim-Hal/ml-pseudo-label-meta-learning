"""
Meta-learner evaluation utilities for Phase 4.

Provides LOO-CV evaluation for both classification (predict best_method)
and regression (predict all 6 LSE values).
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

LSE_COLS = ['LSE_kmeans', 'LSE_dbscan', 'LSE_agg', 'LSE_gmm', 'LSE_autoenc', 'LSE_dictlearn']
META_COLS = ['dataset_id', 'best_method', 'gt_accuracy']
METHOD_NAMES = ['kmeans', 'dbscan', 'agg', 'gmm', 'autoenc', 'dictlearn']


def extract_Xy_clf(df):
    """
    Return (X, y_str, feature_cols, dataset_ids) for classification.

    X      : float array (n_datasets, n_features)
    y_str  : string array of best_method labels
    """
    feat_cols = [c for c in df.columns if c not in META_COLS + LSE_COLS]
    X = df[feat_cols].values.astype(float)
    y = df['best_method'].values
    ids = df['dataset_id'].values
    return X, y, feat_cols, ids


def extract_Xy_reg(df):
    """
    Return (X, Y_lse, feature_cols, dataset_ids) for multi-output regression.

    Y_lse  : float array (n_datasets, 6) of LSE values
    """
    feat_cols = [c for c in df.columns if c not in META_COLS + LSE_COLS]
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

    # Expected LSE: for each dataset, the actual LSE of the predicted method
    expected_lse = np.array([
        Y_lse[i, pred_best_idx[i]]
        for i in range(len(Y_lse))
        if not nan_mask[i]
    ])

    return {
        'mae_per_col': mae_per_col,
        'mae_mean': float(mae_per_col.mean()),
        'argmax_accuracy': argmax_acc,
        'predicted_lse': Y_pred,
        'expected_lse': float(np.nanmean(expected_lse)),
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


def build_classifier_candidates(best_k=1, random_state=42):
    """
    Candidate classifiers for the dataset-level meta-learning task.

    The current notebook originally compares only kNN and MLP; these extra
    candidates provide stronger baselines for small, imbalanced meta-tables.
    """
    return {
        'kNN': Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
            ('clf', KNeighborsClassifier(n_neighbors=best_k)),
        ]),
        'LogReg': Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
            ('clf', LogisticRegression(
                C=0.5,
                max_iter=5000,
                class_weight='balanced',
                random_state=random_state,
            )),
        ]),
        'SVC-RBF': Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
            ('clf', SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                class_weight='balanced',
                random_state=random_state,
            )),
        ]),
        'ExtraTrees': Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('clf', ExtraTreesClassifier(
                n_estimators=400,
                min_samples_leaf=2,
                class_weight='balanced_subsample',
                random_state=random_state,
            )),
        ]),
        'RF': Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('clf', RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=2,
                class_weight='balanced_subsample',
                random_state=random_state,
            )),
        ]),
        'MLP': Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
            ('clf', MLPClassifier(
                hidden_layer_sizes=(64, 32),
                alpha=1e-2,
                max_iter=1500,
                random_state=random_state,
            )),
        ]),
    }


def build_regressor_candidates(best_k=1, random_state=42):
    """Candidate regressors for predicting the six raw LSE targets."""
    return {
        'kNN': Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
            ('reg', KNeighborsRegressor(n_neighbors=best_k)),
        ]),
        'RF': Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('reg', RandomForestRegressor(
                n_estimators=400,
                min_samples_leaf=2,
                random_state=random_state,
            )),
        ]),
        'ExtraTrees': Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('reg', ExtraTreesRegressor(
                n_estimators=400,
                min_samples_leaf=2,
                random_state=random_state,
            )),
        ]),
        'MLP': Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
            ('reg', MLPRegressor(
                hidden_layer_sizes=(64, 32),
                alpha=1e-2,
                max_iter=1500,
                random_state=random_state,
            )),
        ]),
    }
