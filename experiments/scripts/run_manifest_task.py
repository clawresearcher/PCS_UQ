#!/usr/bin/env python3
"""Run one hash-bound PCS-UQ task selected by its task-inventory row."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from experiments.scripts.reproduction_contract import (
    artifact_paths,
    artifact_provenance_path,
    load_and_bind_inventory,
    write_artifact_provenance,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--repository", type=Path, default=Path("."))
    args = parser.parse_args()

    rows = load_and_bind_inventory(args.inventory, args.repository)
    if args.task_id < 0 or args.task_id >= len(rows):
        raise SystemExit(f"task ID outside 0..{len(rows) - 1}: {args.task_id}")
    row = rows[args.task_id]
    if int(row["task_id"]) != args.task_id:
        raise SystemExit("inventory task IDs are not ordered")

    module = (
        "experiments.scripts.run_regression_exp"
        if row["family"] == "regression"
        else "experiments.scripts.run_classification_exp"
    )
    estimator_argument = row["estimator"]
    if estimator_argument == "__candidate_pool__":
        estimator_argument = (
            "XGBoost" if row["family"] == "regression" else "HistGradientBoosting"
        )
    results_root = (
        args.repository / "experiments/results/reg_max"
        if row["family"] == "regression"
        else args.repository / "experiments/results/class_max"
    )
    command = [
        sys.executable,
        "-m",
        module,
        "--dataset",
        row["dataset"],
        "--UQ_method",
        row["method"],
        "--seed",
        row["seed"],
        "--estimator",
        estimator_argument,
        "--train_size",
        row["train_size"],
    ]
    paths = artifact_paths(results_root, row)
    provenance_paths = [artifact_provenance_path(path) for path in paths.values()]
    if any(path.exists() for path in (*paths.values(), *provenance_paths)):
        existing = [
            str(path)
            for path in (*paths.values(), *provenance_paths)
            if path.exists()
        ]
        raise SystemExit(
            "refusing to relabel existing task artifacts; remove or archive first: "
            f"{existing}"
        )
    print("PCS_UQ_TASK", *command, flush=True)
    subprocess.run(command, check=True, cwd=args.repository)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise SystemExit(f"producer did not emit expected artifacts: {missing}")
    write_artifact_provenance(row, args.inventory, args.repository, paths)


if __name__ == "__main__":
    main()
