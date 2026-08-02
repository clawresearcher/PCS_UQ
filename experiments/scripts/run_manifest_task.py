#!/usr/bin/env python3
"""Run one hash-bound PCS-UQ task selected by its task-inventory row."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
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
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--results-root", type=Path)
    args = parser.parse_args()

    rows = load_and_bind_inventory(args.inventory, args.repository)
    if (args.task_index is None) == (args.task_id is None):
        raise SystemExit("provide exactly one of --task-index or --task-id")
    if args.task_index is not None:
        if args.task_index < 0 or args.task_index >= len(rows):
            raise SystemExit(
                f"task index outside 0..{len(rows) - 1}: {args.task_index}"
            )
        row = rows[args.task_index]
    else:
        matches = [row for row in rows if int(row["task_id"]) == args.task_id]
        if len(matches) != 1:
            raise SystemExit(f"task ID is not unique in inventory: {args.task_id}")
        row = matches[0]

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
    default_results_root = (
        args.repository / "experiments/results/reg_max"
        if row["family"] == "regression"
        else args.repository / "experiments/results/class_max"
    )
    results_root = (
        args.results_root
        if args.results_root is not None
        else default_results_root
    )
    if not results_root.is_absolute():
        results_root = args.repository / results_root
    results_root.parent.mkdir(parents=True, exist_ok=True)
    final_paths = artifact_paths(results_root, row)
    final_sidecars = [artifact_provenance_path(path) for path in final_paths.values()]
    if any(path.exists() for path in (*final_paths.values(), *final_sidecars)):
        existing = [
            str(path)
            for path in (*final_paths.values(), *final_sidecars)
            if path.exists()
        ]
        raise SystemExit(
            "refusing to overwrite existing task artifacts; use a clean --results-root: "
            f"{existing}"
        )

    temporary_root = Path(
        tempfile.mkdtemp(prefix=f"pcs-uq-task-{row['task_id']}-", dir=results_root.parent)
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
        "--results_dir",
        str(temporary_root),
    ]
    try:
        print("PCS_UQ_TASK", *command, flush=True)
        subprocess.run(command, check=True, cwd=args.repository)
        temporary_paths = artifact_paths(temporary_root, row)
        missing = [str(path) for path in temporary_paths.values() if not path.is_file()]
        if missing:
            raise SystemExit(f"producer did not emit expected artifacts: {missing}")
        if any(path.exists() for path in (*final_paths.values(), *final_sidecars)):
            raise SystemExit("task destination changed during execution; refusing publication")
        for kind, temporary_path in temporary_paths.items():
            final_path = final_paths[kind]
            final_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path.replace(final_path)
        write_artifact_provenance(row, args.inventory, args.repository, final_paths)
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


if __name__ == "__main__":
    main()
