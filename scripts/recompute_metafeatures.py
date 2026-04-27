"""Recompute meta-feature tables from a fixed LSE table."""

import argparse
import os
import sys

import openml
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from data_utils import load_openml_classification
from metafeatures import extract_optA, extract_optA_label_diagnostics, extract_optB, extract_optC

LSE_COLS = ["LSE_kmeans", "LSE_dbscan", "LSE_agg", "LSE_gmm", "LSE_autoenc", "LSE_dictlearn"]
TARGET_COLS = ["dataset_id"] + LSE_COLS + [
    "best_method",
    "gt_accuracy",
    "majority_accuracy",
    "gt_lift_over_majority",
]


def _vector_row(prefix, dataset_id, values):
    half = len(values) // 2
    row = {"dataset_id": dataset_id}
    for i, value in enumerate(values[:half]):
        row[f"{prefix}_mean_{i}"] = float(value)
    for i, value in enumerate(values[half:]):
        row[f"{prefix}_var_{i}"] = float(value)
    return row


def recompute(args):
    openml.config.cache_directory = args.cache_dir
    lse_df = pd.read_csv(args.lse_table)
    ids = lse_df["dataset_id"].astype(int).tolist()

    rows_a = []
    rows_b = []
    rows_c = []

    for i, did in enumerate(ids):
        X_train, _X_test, y_train, _y_test, metadata = load_openml_classification(did)
        n_classes = int(metadata["n_classes"])

        feats_a = extract_optA(X_train, n_classes=n_classes)
        feats_a["dataset_id"] = did
        if args.include_label_diagnostics:
            feats_a.update(extract_optA_label_diagnostics(X_train, y_train))
        rows_a.append(feats_a)

        rows_b.append(_vector_row("ae", did, extract_optB(X_train, n_classes)))
        rows_c.append(_vector_row("dl", did, extract_optC(X_train, n_classes)))

        print(f"[{i+1:3d}/{len(ids)}] did={did} optA/optB/optC done")

    targets = lse_df[[c for c in TARGET_COLS if c in lse_df.columns]]
    out_a = targets.merge(pd.DataFrame(rows_a), on="dataset_id", how="inner")
    out_b = targets.merge(pd.DataFrame(rows_b), on="dataset_id", how="inner")
    out_c = targets.merge(pd.DataFrame(rows_c), on="dataset_id", how="inner")

    os.makedirs(args.output_dir, exist_ok=True)
    out_a.to_csv(os.path.join(args.output_dir, "meta_training_optA_fixed.csv"), index=False)
    out_b.to_csv(os.path.join(args.output_dir, "meta_training_optB_fixed.csv"), index=False)
    out_c.to_csv(os.path.join(args.output_dir, "meta_training_optC_fixed.csv"), index=False)
    out_a.to_csv(os.path.join(args.output_dir, "meta_training_fixed.csv"), index=False)

    print(f"Wrote fixed feature tables to {args.output_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lse-table", default=os.path.join(ROOT, "data", "meta_table", "meta_training_lse_fixed.csv"))
    parser.add_argument("--output-dir", default=os.path.join(ROOT, "data", "meta_table"))
    parser.add_argument("--cache-dir", default=os.path.join(ROOT, "data", "raw"))
    parser.add_argument("--include-label-diagnostics", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    recompute(parse_args())
