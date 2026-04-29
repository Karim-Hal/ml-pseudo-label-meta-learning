"""LSE computation using balanced accuracy and Hungarian alignment.

LSE = balanced_accuracy(RF on pseudo-labels) / balanced_accuracy(RF on true labels)

Balanced accuracy handles class imbalance correctly: a method that degenerates to
majority-class prediction scores 1/n_classes instead of the raw majority rate.
class_weight='balanced' prevents the fixed RF from doing the same.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score

from hungarian import build_mapping

SEED = 42


def groundtruth_accuracy(X_tr_raw, y_tr, X_te_raw, y_te):
    """RF trained on true labels — denominator for all LSE values.

    Uses balanced_accuracy_score so imbalanced datasets (e.g. Credit Card Fraud)
    are evaluated fairly. class_weight='balanced' prevents the RF from collapsing
    to majority-class prediction on skewed targets.
    """
    rf = RandomForestClassifier(random_state=SEED, class_weight='balanced')
    rf.fit(X_tr_raw, y_tr)
    return float(balanced_accuracy_score(y_te, rf.predict(X_te_raw)))


def compute_lse(X_tr_raw, y_tr, X_te_raw, y_te, pseudo_labels_tr, gt_acc,
                method_name='', dataset_name='', verbose=True):
    """
    Compute LSE for one (dataset, method) pair.

    Pipeline
    --------
    1. Cluster structure diagnostics.
    2. Build cluster→class mapping via Hungarian alignment (build_mapping).
    3. Convert pseudo_labels_tr → pseudo-class labels using the mapping.
    4. Train RF(class_weight='balanced') on (X_tr_raw, pseudo_y_train).
    5. Predict class labels on test set using true labels.
    6. LSE = balanced_accuracy(step 5) / gt_acc.

    Parameters
    ----------
    X_tr_raw, X_te_raw : ndarray  (unscaled; RF is scale-invariant)
    y_tr, y_te         : ndarray  integer class labels (0..n_classes-1)
    pseudo_labels_tr   : ndarray  cluster IDs from a clustering method
    gt_acc             : float    balanced RF accuracy on true labels (denominator)

    Returns
    -------
    lse  : float   balanced_acc / gt_acc (unclipped)
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

    # 4. Train RF on pseudo-class labels — class_weight='balanced' prevents
    #    degeneration on imbalanced datasets (e.g. majority-only prediction)
    rf = RandomForestClassifier(random_state=SEED, class_weight='balanced')
    rf.fit(X_tr_raw, pseudo_y_train)
    train_bal_acc_on_pseudo = balanced_accuracy_score(pseudo_y_train, rf.predict(X_tr_raw))
    diag['rf_train_bal_acc_on_pseudo'] = round(float(train_bal_acc_on_pseudo), 3)
    diag['rf_underfit_pseudo']         = bool(train_bal_acc_on_pseudo < 0.80)

    # 5. Predict on test set
    test_preds_classes = rf.predict(X_te_raw)
    diag['unique_predicted_classes'] = int(len(np.unique(test_preds_classes)))

    # 6. Balanced accuracy and LSE
    bal_acc = float(balanced_accuracy_score(y_te, test_preds_classes))

    # Majority baseline for balanced accuracy = 1/n_classes
    # (predicting the majority class gives recall=1 for that class, 0 for all others)
    majority_baseline_bal = 1.0 / n_classes
    random_baseline       = 1.0 / n_classes  # same for balanced accuracy

    diag['bal_acc']               = round(bal_acc, 3)
    diag['majority_baseline_bal'] = round(majority_baseline_bal, 3)
    diag['gt_acc']                = round(float(gt_acc), 3)

    lse_ratio  = bal_acc / gt_acc if gt_acc > 0 else 0.0
    lift_denom = gt_acc - majority_baseline_bal
    lse_lift   = (bal_acc - majority_baseline_bal) / lift_denom if lift_denom > 1e-6 else 0.0
    diag['lse_ratio'] = round(float(lse_ratio), 3)
    diag['lse_lift']  = round(float(lse_lift),  3)

    # 7. Failure-mode flags
    diag['beats_majority']    = bool(bal_acc > majority_baseline_bal + 0.01)
    diag['matches_majority']  = bool(abs(bal_acc - majority_baseline_bal) <= 0.01)
    diag['gt_beats_majority'] = bool(gt_acc > majority_baseline_bal + 0.05)

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
              f"bal_acc={bal_acc:.3f}  gt_bal={gt_acc:.3f}  "
              f"lse={lse_ratio:.3f}  lift={lse_lift:.3f}{flag_str}")

    return float(lse_ratio), diag
