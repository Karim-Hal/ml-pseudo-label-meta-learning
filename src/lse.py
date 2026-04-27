"""Label Substitution Efficiency computation."""

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import normalized_mutual_info_score

try:
    from .hungarian import align_cluster_labels
except ImportError:  # pragma: no cover - supports notebook sys.path imports
    from hungarian import align_cluster_labels

SEED = 42


@dataclass(frozen=True)
class LSEResult:
    """LSE result for one dataset/method pair."""

    lse_ratio: float
    pseudo_accuracy: float
    gt_accuracy: float
    lse_lift: float
    diagnostics: dict
    pseudo_labels_train_aligned: np.ndarray


def majority_accuracy(y):
    """Accuracy of always predicting the most frequent class."""
    y = np.asarray(y)
    if len(y) == 0:
        raise ValueError("cannot compute majority accuracy for empty labels")
    _, counts = np.unique(y, return_counts=True)
    return float(counts.max() / counts.sum())


def groundtruth_accuracy(X_train, y_train, X_test, y_test, random_state=SEED):
    """RF denominator for LSE: same model family, trained on true labels."""
    rf = RandomForestClassifier(random_state=random_state)
    rf.fit(X_train, y_train)
    return float(rf.score(X_test, y_test))


def compute_lse(
    X_train,
    y_train,
    X_test,
    y_test,
    pseudo_labels_train,
    gt_accuracy=None,
    method_name="",
    dataset_name="",
    random_state=SEED,
):
    """
    Compute LSE after true Hungarian cluster-to-class alignment.

    The downstream RF is trained on aligned pseudo-class labels, not raw cluster
    IDs. True labels are used only for alignment diagnostics and final scoring.
    """
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    pseudo_labels_train = np.asarray(pseudo_labels_train)

    if len(pseudo_labels_train) != len(y_train):
        raise ValueError("pseudo_labels_train must match y_train length")

    if gt_accuracy is None:
        gt_accuracy = groundtruth_accuracy(
            X_train, y_train, X_test, y_test, random_state=random_state
        )
    if gt_accuracy <= 0:
        raise ValueError("gt_accuracy must be positive to compute LSE")

    alignment = align_cluster_labels(pseudo_labels_train, y_train)
    pseudo_y_train = alignment.aligned_labels

    rf = RandomForestClassifier(random_state=random_state)
    rf.fit(X_train, pseudo_y_train)
    y_pred = rf.predict(X_test)
    pseudo_acc = float((y_pred == y_test).mean())

    train_majority = majority_accuracy(y_train)
    test_majority = majority_accuracy(y_test)
    lse_ratio = pseudo_acc / gt_accuracy
    lift_denom = gt_accuracy - test_majority
    lse_lift = (
        (pseudo_acc - test_majority) / lift_denom
        if lift_denom > 1e-6
        else 0.0
    )

    cluster_values, cluster_counts = np.unique(pseudo_labels_train, return_counts=True)
    n_clusters = len(cluster_values)
    n_classes = len(np.unique(y_train))
    largest_cluster_frac = float(cluster_counts.max() / cluster_counts.sum())

    diagnostics = {
        "dataset": dataset_name,
        "method": method_name,
        "n_clusters": int(n_clusters),
        "n_classes": int(n_classes),
        "largest_cluster_frac": round(largest_cluster_frac, 6),
        "cluster_collapse": bool(n_clusters < n_classes),
        "cluster_degenerate": bool(largest_cluster_frac > 0.90),
        "unique_mapped_classes": int(alignment.unique_mapped_classes),
        "mapping_collapse": bool(alignment.unique_mapped_classes < n_classes),
        "unassigned_clusters": int(alignment.n_unassigned_clusters),
        "train_majority_baseline": round(train_majority, 6),
        "majority_baseline": round(test_majority, 6),
        "gt_accuracy": round(float(gt_accuracy), 6),
        "pseudo_accuracy": round(pseudo_acc, 6),
        "lse_ratio": round(float(lse_ratio), 6),
        "lse_lift": round(float(lse_lift), 6),
        "beats_majority": bool(pseudo_acc > test_majority + 0.01),
        "matches_majority": bool(abs(pseudo_acc - test_majority) <= 0.01),
        "gt_beats_majority": bool(gt_accuracy > test_majority + 0.05),
        "nmi_train": round(
            float(normalized_mutual_info_score(y_train, pseudo_labels_train)),
            6,
        ),
    }

    return LSEResult(
        lse_ratio=float(lse_ratio),
        pseudo_accuracy=pseudo_acc,
        gt_accuracy=float(gt_accuracy),
        lse_lift=float(lse_lift),
        diagnostics=diagnostics,
        pseudo_labels_train_aligned=pseudo_y_train,
    )
