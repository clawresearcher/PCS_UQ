#!/usr/bin/env python3
"""Run one hash-bound PCS-UQ task selected by its task-inventory row."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from experiments.scripts.reproduction_contract import load_and_bind_inventory


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
        row["estimator"],
        "--train_size",
        row["train_size"],
    ]
    print("PCS_UQ_TASK", *command, flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
