"""
Hungarian alignment utilities for pseudo-label evaluation.

Cluster IDs are arbitrary. These helpers map cluster IDs to class labels by
maximising overlap with the training labels, using scipy's linear assignment
solver on the cluster/class contingency matrix.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class AlignmentResult:
    """Result of aligning cluster IDs to class labels."""

    aligned_labels: np.ndarray
    mapping: dict
    contingency: np.ndarray
    cluster_values: np.ndarray
    class_values: np.ndarray
    n_unassigned_clusters: int
    unique_mapped_classes: int


def contingency_matrix(cluster_labels, true_labels):
    """
    Build a count matrix with rows as cluster IDs and columns as true classes.

    Returns
    -------
    matrix, cluster_values, class_values
    """
    clusters = np.asarray(cluster_labels)
    classes = np.asarray(true_labels)
    if clusters.shape[0] != classes.shape[0]:
        raise ValueError("cluster_labels and true_labels must have the same length")
    if clusters.shape[0] == 0:
        raise ValueError("cannot align an empty label array")

    cluster_values = np.unique(clusters)
    class_values = np.unique(classes)
    matrix = np.zeros((len(cluster_values), len(class_values)), dtype=int)

    cluster_index = {value: i for i, value in enumerate(cluster_values)}
    class_index = {value: j for j, value in enumerate(class_values)}
    for cluster, cls in zip(clusters, classes):
        matrix[cluster_index[cluster], class_index[cls]] += 1

    return matrix, cluster_values, class_values


def hungarian_mapping(cluster_labels, true_labels):
    """
    Return the optimal cluster -> class mapping for maximum overlap.

    If there are more clusters than classes, the Hungarian solver can only assign
    one cluster per class. Remaining clusters are mapped to their own majority
    class so every training sample receives a usable pseudo-class label.
    """
    matrix, cluster_values, class_values = contingency_matrix(cluster_labels, true_labels)
    row_ind, col_ind = linear_sum_assignment(-matrix)

    mapping = {
        cluster_values[row]: class_values[col]
        for row, col in zip(row_ind, col_ind)
    }

    for row, cluster in enumerate(cluster_values):
        if cluster in mapping:
            continue
        majority_col = int(np.argmax(matrix[row]))
        mapping[cluster] = class_values[majority_col]

    return mapping, matrix, cluster_values, class_values


def align_cluster_labels(cluster_labels, true_labels):
    """Align arbitrary cluster IDs to true class IDs and return diagnostics."""
    clusters = np.asarray(cluster_labels)
    mapping, matrix, cluster_values, class_values = hungarian_mapping(clusters, true_labels)
    aligned = np.asarray([mapping[c] for c in clusters])
    assigned_by_hungarian = min(len(cluster_values), len(class_values))

    return AlignmentResult(
        aligned_labels=aligned,
        mapping=mapping,
        contingency=matrix,
        cluster_values=cluster_values,
        class_values=class_values,
        n_unassigned_clusters=max(0, len(cluster_values) - assigned_by_hungarian),
        unique_mapped_classes=len(set(mapping.values())),
    )
