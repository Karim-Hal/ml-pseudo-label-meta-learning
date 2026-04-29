"""LSE computation using Hungarian alignment, and groundtruth accuracy."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from hungarian import build_mapping

SEED = 42


def groundtruth_accuracy(X_tr_raw, y_tr, X_te_raw, y_te):
    """RF trained on true labels — denominator for all LSE values."""
    rf = RandomForestClassifier(random_state=SEED)
    rf.fit(X_tr_raw, y_tr)
    return rf.score(X_te_raw, y_te)


def compute_lse(X_tr_raw, y_tr, X_te_raw, y_te, pseudo_labels_tr, gt_acc,
                method_name='', dataset_name='', verbose=True):
    """
    Compute LSE for one (dataset, method) pair.

    Pipeline
    --------
    1. Compute cluster structure diagnostics.
    2. Build cluster→class mapping via Hungarian alignment (build_mapping).
    3. Convert pseudo_labels_tr to pseudo-class labels using the mapping.
    4. Train a fixed RF on (X_tr_raw, pseudo_y_train) — class labels, not IDs.
    5. Predict class labels directly on the test set.
    6. LSE = test_accuracy / gt_acc.

    Parameters
    ----------
    X_tr_raw, X_te_raw : ndarray  (unscaled; RF is scale-invariant)
    y_tr, y_te         : ndarray  integer class labels (0..n_classes-1)
    pseudo_labels_tr   : ndarray  cluster IDs from a clustering method
    gt_acc             : float    RF accuracy on true labels (denominator)

    Returns
    -------
    lse  : float   acc / gt_acc (unclipped)
    diag : dict    flat diagnostic record
    """
    diag = {'dataset': dataset_name, 'method': method_name}

    # 1. Cluster structure diagnostics
    unique_clusters, cluster_sizes = np.unique(pseudo_labels_tr, return_counts=True)
    n_clusters = len(unique_clusters)
    n_classes  = len(np.unique(y_tr))
    largest_cluster_frac = cluster_sizes.max() / cluster_sizes.sum()

    diag['n_clusters']           = int(n_clusters)
    diag['n_classes']            = int(n_classes)
    diag['largest_cluster_frac'] = round(float(largest_cluster_frac), 3)
    diag['cluster_collapse']     = bool(n_clusters < n_classes)
    diag['cluster_degenerate']   = bool(largest_cluster_frac > 0.90)

    # 2. Hungarian alignment
    mapping = build_mapping(pseudo_labels_tr, y_tr)
    unique_mapped_classes = len(set(mapping.values()))
    diag['unique_mapped_classes'] = int(unique_mapped_classes)
    diag['mapping_collapse']      = bool(unique_mapped_classes < n_classes)

    # 3. Translate cluster IDs → pseudo-class labels
    pseudo_y_train = np.array([mapping[int(c)] for c in pseudo_labels_tr])

    # 4. Train RF on pseudo-class labels (not cluster IDs)
    rf = RandomForestClassifier(random_state=SEED)
    rf.fit(X_tr_raw, pseudo_y_train)
    train_acc_on_pseudo = rf.score(X_tr_raw, pseudo_y_train)
    diag['rf_train_acc_on_pseudo'] = round(float(train_acc_on_pseudo), 3)
    diag['rf_underfit_pseudo']     = bool(train_acc_on_pseudo < 0.80)

    # 5. Predict directly — RF already outputs class labels
    test_preds_classes = rf.predict(X_te_raw)
    diag['unique_predicted_classes'] = int(len(np.unique(test_preds_classes)))

    # 6. Accuracy and LSE
    acc = float((test_preds_classes == y_te).mean())
    fallback = int(np.bincount(y_tr.astype(int)).argmax())
    majority_baseline = float((y_te == fallback).mean())
    random_baseline   = 1.0 / n_classes

    diag['acc']               = round(acc, 3)
    diag['majority_baseline'] = round(majority_baseline, 3)
    diag['random_baseline']   = round(random_baseline, 3)
    diag['gt_acc']            = round(float(gt_acc), 3)

    lse_ratio  = acc / gt_acc if gt_acc > 0 else 0.0
    lift_denom = gt_acc - majority_baseline
    lse_lift   = (acc - majority_baseline) / lift_denom if lift_denom > 1e-6 else 0.0
    diag['lse_ratio'] = round(float(lse_ratio), 3)
    diag['lse_lift']  = round(float(lse_lift),  3)

    # 7. Failure-mode flags
    diag['beats_majority']    = bool(acc > majority_baseline + 0.01)
    diag['matches_majority']  = bool(abs(acc - majority_baseline) <= 0.01)
    diag['gt_beats_majority'] = bool(gt_acc > majority_baseline + 0.05)

    if verbose:
        flags = []
        if diag['cluster_collapse']:      flags.append('CLUSTER_COLLAPSE')
        if diag['cluster_degenerate']:    flags.append('DEGENERATE_CLUSTER')
        if diag['mapping_collapse']:      flags.append('MAPPING_COLLAPSE')
        if diag['matches_majority']:      flags.append('MATCHES_MAJORITY')
        if diag['rf_underfit_pseudo']:    flags.append('RF_UNDERFIT')
        if not diag['gt_beats_majority']: flags.append('GT_NEAR_MAJORITY')
        flag_str = '  [' + ', '.join(flags) + ']' if flags else ''
        print(f"    {method_name:10s}  n_cl={n_clusters}/{n_classes}  "
              f"maj_base={majority_baseline:.3f}  acc={acc:.3f}  "
              f"lse_r={lse_ratio:.3f}  lse_lift={lse_lift:.3f}{flag_str}")

    return float(lse_ratio), diag
