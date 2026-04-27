import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hungarian import align_cluster_labels, contingency_matrix
from lse import compute_lse, groundtruth_accuracy
from metafeatures import extract_optA


class HungarianAlignmentTests(unittest.TestCase):
    def test_hungarian_finds_non_identity_mapping(self):
        clusters = np.array([10, 10, 20, 20, 20, 30])
        true = np.array([1, 1, 0, 0, 0, 2])

        matrix, cluster_values, class_values = contingency_matrix(clusters, true)
        result = align_cluster_labels(clusters, true)

        self.assertEqual(matrix.shape, (3, 3))
        self.assertEqual(cluster_values.tolist(), [10, 20, 30])
        self.assertEqual(class_values.tolist(), [0, 1, 2])
        self.assertEqual(result.mapping[10], 1)
        self.assertEqual(result.mapping[20], 0)
        self.assertEqual(result.mapping[30], 2)
        np.testing.assert_array_equal(result.aligned_labels, true)


class LSETests(unittest.TestCase):
    def test_lse_trains_on_aligned_pseudo_classes(self):
        X_train = np.array(
            [[0.0], [0.1], [0.2], [3.0], [3.1], [3.2]],
            dtype=float,
        )
        y_train = np.array([0, 0, 0, 1, 1, 1])
        X_test = np.array([[0.05], [3.05]], dtype=float)
        y_test = np.array([0, 1])

        # Deliberately swapped arbitrary cluster IDs.
        pseudo = np.array([7, 7, 7, 4, 4, 4])
        gt_acc = groundtruth_accuracy(X_train, y_train, X_test, y_test)
        result = compute_lse(X_train, y_train, X_test, y_test, pseudo, gt_accuracy=gt_acc)

        self.assertEqual(result.diagnostics["unique_mapped_classes"], 2)
        self.assertGreaterEqual(result.lse_ratio, 0.99)


class MetaFeatureTests(unittest.TestCase):
    def test_option_a_can_run_without_labels(self):
        X = np.array(
            [
                [0.0, 1.0],
                [0.1, 1.1],
                [4.0, 3.9],
                [4.1, 4.2],
                [8.0, 8.1],
                [8.2, 8.0],
            ],
            dtype=float,
        )
        feats = extract_optA(X, n_classes=3)

        self.assertIn("kmeans_est_imbalance_ratio", feats)
        self.assertIn("kmeans_silhouette", feats)
        self.assertNotIn("class_entropy", feats)
        self.assertNotIn("knn1_accuracy", feats)


if __name__ == "__main__":
    unittest.main()
