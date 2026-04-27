"""Recompute the LSE benchmark table from an OpenML manifest.

Example
-------
python scripts/recompute_lse.py \
  --manifest data/meta_table/dataset_manifest.csv \
  --output data/meta_table/meta_training_lse_fixed.csv \
  --diagnostics data/meta_table/diagnostics_lse_fixed.csv
"""

import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import openml
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from clustering import METHODS
from data_utils import groundtruth_quality_gate, load_openml_classification
from lse import compute_lse

SHOWCASE_IDS = {61, 187, 15, 53, 40966, 37, 54, 1590, 1597}


def _best_method(row, lse_cols):
    vals = {col: row[col] for col in lse_cols if pd.notna(row.get(col))}
    if not vals:
        return "FAILED"
    return max(vals, key=vals.get).replace("LSE_", "")


def _process_dataset_worker(dataset_row, min_gt_lift, cache_dir, queue):
    """Process one dataset in a child process so the parent can enforce timeout."""
    openml.config.cache_directory = cache_dir

    did = int(dataset_row["dataset_id"])
    name = str(dataset_row.get("name", did))
    lse_cols = list(METHODS.keys())
    diagnostics = []
    logs = []
    t0 = time.time()

    rec = {"dataset_id": did, "failure_reason": ""}
    for col in lse_cols:
        rec[col] = np.nan

    try:
        X_train, X_test, y_train, y_test, metadata = load_openml_classification(did)
        n_classes = int(metadata["n_classes"])
        gate = groundtruth_quality_gate(
            X_train,
            y_train,
            X_test,
            y_test,
            min_lift=min_gt_lift,
        )
        rec["gt_accuracy"] = round(gate["gt_accuracy"], 6)
        rec["majority_accuracy"] = round(gate["majority_accuracy"], 6)
        rec["gt_lift_over_majority"] = round(gate["gt_lift_over_majority"], 6)
        rec["n_transformed_features"] = metadata["n_transformed_features"]

        if not gate["keep"]:
            rec["best_method"] = "FILTERED"
            rec["failure_reason"] = gate["failure_reason"]
            logs.append(f"FILTER {did} {name} ({gate['failure_reason']})")
            queue.put((rec, diagnostics, logs))
            return

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        logs.append(
            f"{did} {name[:35]:35s} "
            f"gt={gate['gt_accuracy']:.3f} maj={gate['majority_accuracy']:.3f}"
        )

        for col, fn in METHODS.items():
            method = col.replace("LSE_", "")
            try:
                pseudo = fn(X_train_scaled, n_classes)
                lse = compute_lse(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    pseudo,
                    gt_accuracy=gate["gt_accuracy"],
                    method_name=method,
                    dataset_name=name,
                )
                rec[col] = round(lse.lse_ratio, 6)
                diagnostics.append(lse.diagnostics)
                logs.append(
                    f"    {method:10s} lse={lse.lse_ratio:.3f} "
                    f"lift={lse.lse_lift:.3f} nmi={lse.diagnostics['nmi_train']:.3f}"
                )
            except Exception as exc:
                diagnostics.append(
                    {
                        "dataset": name,
                        "method": method,
                        "failed": True,
                        "error": str(exc)[:300],
                    }
                )
                logs.append(f"    {method:10s} FAILED: {exc}")

        rec["best_method"] = _best_method(rec, lse_cols)
        logs.append(f"           best={rec['best_method']} ({time.time() - t0:.1f}s)")

    except Exception as exc:
        rec["best_method"] = "FAILED"
        rec["failure_reason"] = str(exc)[:300]
        logs.append(f"FAIL {did} {name}: {exc}")

    queue.put((rec, diagnostics, logs))


def _run_dataset_with_timeout(dataset_row, args):
    """Run one dataset and return (record, diagnostics, logs), timing out if needed."""
    did = int(dataset_row["dataset_id"])
    name = str(dataset_row.get("name", did))
    queue = mp.Queue()
    process = mp.Process(
        target=_process_dataset_worker,
        args=(dataset_row, args.min_gt_lift, args.cache_dir, queue),
    )
    process.start()
    process.join(args.dataset_timeout)

    if process.is_alive():
        process.terminate()
        process.join(10)
        if process.is_alive():
            process.kill()
            process.join()

        rec = {
            "dataset_id": did,
            "best_method": "TIMEOUT",
            "failure_reason": f"dataset_timeout_over_{args.dataset_timeout}s",
        }
        for col in METHODS:
            rec[col] = np.nan
        diagnostics = [
            {
                "dataset": name,
                "failed": True,
                "error": rec["failure_reason"],
            }
        ]
        logs = [f"TIMEOUT {did} {name} after {args.dataset_timeout}s; skipping dataset"]
        return rec, diagnostics, logs

    if queue.empty():
        rec = {
            "dataset_id": did,
            "best_method": "FAILED",
            "failure_reason": "worker_exited_without_result",
        }
        for col in METHODS:
            rec[col] = np.nan
        return rec, [], [f"FAIL {did} {name}: worker exited without result"]

    return queue.get()


def recompute(args):
    openml.config.cache_directory = args.cache_dir
    manifest = pd.read_csv(args.manifest)

    leaked = set(manifest["dataset_id"].astype(int)) & SHOWCASE_IDS
    if leaked:
        raise ValueError(f"showcase datasets leaked into manifest: {sorted(leaked)}")

    lse_cols = list(METHODS.keys())
    results = []
    diagnostics = []

    for i, row in manifest.iterrows():
        rec, diag_rows, logs = _run_dataset_with_timeout(row.to_dict(), args)
        for j, message in enumerate(logs):
            prefix = f"[{i+1:3d}/{len(manifest)}] " if j == 0 else ""
            print(prefix + message)

        results.append(rec)
        diagnostics.extend(diag_rows)
        pd.DataFrame(results).to_csv(args.checkpoint, index=False)

    out_df = pd.DataFrame(results)
    usable = out_df[~out_df["best_method"].isin(["FAILED", "FILTERED", "TIMEOUT"])].copy()
    usable.to_csv(args.output, index=False)
    pd.DataFrame(diagnostics).to_csv(args.diagnostics, index=False)

    print(f"\nWrote usable LSE table: {args.output} shape={usable.shape}")
    print(f"Wrote diagnostics:      {args.diagnostics}")
    print(f"Wrote checkpoint:       {args.checkpoint}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=os.path.join(ROOT, "data", "meta_table", "dataset_manifest.csv"))
    parser.add_argument("--output", default=os.path.join(ROOT, "data", "meta_table", "meta_training_lse_fixed.csv"))
    parser.add_argument("--diagnostics", default=os.path.join(ROOT, "data", "meta_table", "diagnostics_lse_fixed.csv"))
    parser.add_argument("--checkpoint", default=os.path.join(ROOT, "data", "meta_table", "lse_fixed_checkpoint.csv"))
    parser.add_argument("--cache-dir", default=os.path.join(ROOT, "data", "raw"))
    parser.add_argument("--min-gt-lift", type=float, default=0.05)
    parser.add_argument("--dataset-timeout", type=int, default=300)
    return parser.parse_args()


if __name__ == "__main__":
    recompute(parse_args())
