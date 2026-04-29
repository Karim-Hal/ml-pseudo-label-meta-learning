"""Hungarian alignment of cluster IDs to class labels."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def build_mapping(pseudo_labels, true_labels):
    """
    Map cluster IDs to class labels by maximising total overlap.

    Uses scipy.optimize.linear_sum_assignment on the negated contingency
    matrix so the solver minimises cost == maximises overlap.

    Rectangular case
    ----------------
    n_clusters > n_classes (common with DBSCAN): the contingency matrix
    is padded with zero columns to make it square so every cluster row gets
    an assignment.  Clusters matched to padding columns fall back to the
    dataset majority class — documented here because it is a deliberate
    design choice, not a bug.

    n_clusters < n_classes: some classes will never appear in predictions.
    That is expected and requires no special handling.

    Parameters
    ----------
    pseudo_labels : array-like, shape (n_samples,)
        Cluster assignments (integers, e.g. from KMeans.fit_predict).
    true_labels : array-like, shape (n_samples,)
        Ground-truth class labels (integers 0..n_classes-1 from LabelEncoder).

    Returns
    -------
    mapping : dict  {cluster_id -> class_id}
    """
    pseudo_labels = np.asarray(pseudo_labels)
    true_labels   = np.asarray(true_labels, dtype=int)

    cluster_ids = np.unique(pseudo_labels)
    class_ids   = np.unique(true_labels)
    n_clusters  = len(cluster_ids)
    n_classes   = len(class_ids)

    # C[i, j] = #{samples in cluster i with class j}
    C = np.zeros((n_clusters, n_classes), dtype=int)
    for i, cid in enumerate(cluster_ids):
        mask = pseudo_labels == cid
        for j, cls in enumerate(class_ids):
            C[i, j] = int(np.sum(true_labels[mask] == cls))

    # Pad with zero columns when n_clusters > n_classes so every cluster row
    # is assigned.  The zero columns are dominated by real ones in the solver,
    # so real classes are consumed first and extra clusters land on padding.
    if n_clusters > n_classes:
        pad  = n_clusters - n_classes
        C_sq = np.hstack([C, np.zeros((n_clusters, pad), dtype=int)])
    else:
        C_sq = C

    row_ind, col_ind = linear_sum_assignment(-C_sq)

    majority_class = int(class_ids[np.bincount(true_labels).argmax()])

    mapping = {}
    for ri, ci in zip(row_ind, col_ind):
        cid = int(cluster_ids[ri])
        mapping[cid] = int(class_ids[ci]) if ci < n_classes else majority_class

    return mapping
