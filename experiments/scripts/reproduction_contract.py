#!/usr/bin/env python3
"""Freeze and validate PCS-UQ cluster task matrices and result completeness."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

SEEDS = tuple(range(777, 787))
TRAIN_SIZE = "0.8"
REGRESSION_DATASETS = (
    "data_ca_housing",
    "data_diamond",
    "data_parkinsons",
    "data_airfoil",
    "data_computer",
    "data_concrete",
    "data_powerplant",
    "data_miami_housing",
    "data_insurance",
    "data_qsar",
    "data_energy_efficiency",
    "data_kin8nm",
    "data_naval_propulsion",
    "data_superconductor",
    "data_elevator",
    "data_protein_structure",
    "data_debutanizer",
)
REGRESSION_METHODS = (
    "split_conformal",
    "split_conformal_ensemble",
    "split_conformal_alt",
    "split_conformal_ensemble_alt",
    "studentized_conformal",
    "studentized_conformal_ensemble",
    "studentized_conformal_alt",
    "studentized_conformal_ensemble_alt",
    "jackknife_bootstrap",
    "jackknife_bootstrap_ensemble",
    "majority_vote",
    "majority_vote_alt",
    "pcs_uq",
    "pcs_uq_alt",
    "pcs_oob",
    "pcs_oob_downsample",
    "pcs_oob_fixed_method",
    "pcs_oob_downsample_fixed_method",
)
REGRESSION_ALL_ESTIMATORS = (
    "XGBoost",
    "RandomForest",
    "ExtraTrees",
    "AdaBoost",
    "OLS",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "MLP",
)
REGRESSION_REDUCED_METHODS = frozenset(
    {
        "split_conformal_ensemble",
        "split_conformal_ensemble_alt",
        "studentized_conformal_ensemble",
        "studentized_conformal_ensemble_alt",
        "jackknife_bootstrap_ensemble",
        "majority_vote",
        "majority_vote_alt",
        "pcs_uq",
        "pcs_uq_alt",
        "pcs_oob",
        "pcs_oob_downsample",
    }
)
REGRESSION_REDUCED_ESTIMATORS = ("XGBoost",)
CLASSIFICATION_DATASETS = (
    "data_language",
    "data_yeast",
    "data_chess",
    "data_cover_type",
    "data_isolet",
    "data_dionis",
)
CLASSIFICATION_METHODS = (
    "split_conformal_aps",
    "split_conformal_raps",
    "majority_vote",
    "pcs_oob",
    "split_conformal_topk",
)
CLASSIFICATION_ALL_ESTIMATORS = (
    "LogisticRegression",
    "RandomForest",
    "AdaBoost",
    "MLP",
    "XGBoost",
)
CLASSIFICATION_REDUCED_METHODS = frozenset({"majority_vote", "pcs_oob"})
CLASSIFICATION_REDUCED_ESTIMATORS = ("HistGradientBoosting",)


class ReproductionError(RuntimeError):
    """Raised when a run cannot be certified complete."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def task_rows(family: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if family == "regression":
        for method in REGRESSION_METHODS:
            estimators = (
                REGRESSION_REDUCED_ESTIMATORS
                if method in REGRESSION_REDUCED_METHODS
                else REGRESSION_ALL_ESTIMATORS
            )
            for dataset in REGRESSION_DATASETS:
                for seed in SEEDS:
                    for estimator in estimators:
                        method_name = method
                        if method in {
                            "pcs_oob_fixed_method",
                            "pcs_oob_downsample_fixed_method",
                        }:
                            method_name = f"{method}_{estimator}"
                        rows.append(
                            {
                                "task_id": len(rows),
                                "family": family,
                                "dataset": dataset,
                                "method": method,
                                "method_name": method_name,
                                "estimator": estimator,
                                "seed": seed,
                                "train_size": TRAIN_SIZE,
                            }
                        )
    elif family == "classification":
        for method in CLASSIFICATION_METHODS:
            estimators = (
                CLASSIFICATION_REDUCED_ESTIMATORS
                if method in CLASSIFICATION_REDUCED_METHODS
                else CLASSIFICATION_ALL_ESTIMATORS
            )
            for dataset in CLASSIFICATION_DATASETS:
                for seed in SEEDS:
                    for estimator in estimators:
                        rows.append(
                            {
                                "task_id": len(rows),
                                "family": family,
                                "dataset": dataset,
                                "method": method,
                                "method_name": method,
                                "estimator": estimator,
                                "seed": seed,
                                "train_size": TRAIN_SIZE,
                            }
                        )
    else:
        raise ReproductionError(f"unsupported family: {family}")
    return rows


def write_inventory(family: str, output: Path, repository: Path) -> None:
    rows = task_rows(family)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "schema": "pcs-uq-task-inventory-v1",
        "family": family,
        "repository_revision": git_revision(repository),
        "contract_sha256": sha256(Path(__file__).resolve()),
        "task_count": len(rows),
        "task_inventory": str(output),
        "task_inventory_sha256": sha256(output),
        "seeds": list(SEEDS),
        "train_size": TRAIN_SIZE,
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"PCS_UQ_INVENTORY_OK family={family} tasks={len(rows)} "
        f"sha256={metadata['task_inventory_sha256']}"
    )


def load_inventory(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ReproductionError("task inventory is empty")
    expected_ids = [str(index) for index in range(len(rows))]
    if [row["task_id"] for row in rows] != expected_ids:
        raise ReproductionError("task inventory IDs are not contiguous and ordered")
    return rows


def metric_path(results_root: Path, row: dict[str, str]) -> Path:
    return (
        results_root
        / row["dataset"]
        / (
            f"{row['method_name']}_seed_{row['seed']}_"
            f"train_size_{row['train_size']}_metrics.pkl"
        )
    )


def subgroup_path(results_root: Path, row: dict[str, str]) -> Path:
    metric = metric_path(results_root, row)
    return metric.with_name(metric.name.replace("_metrics.pkl", "_subgroup_metrics.pkl"))


def finite_scalars(value: Any) -> Iterable[float]:
    if isinstance(value, dict):
        for nested in value.values():
            yield from finite_scalars(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from finite_scalars(nested)
    elif isinstance(value, (int, float)):
        yield float(value)


def validate_pickle(path: Path) -> str:
    try:
        with path.open("rb") as handle:
            value = pickle.load(handle)
    except Exception as exc:
        raise ReproductionError(f"cannot load {path}: {exc}") from exc
    if not isinstance(value, dict) or not value:
        raise ReproductionError(f"empty or non-dict artifact: {path}")
    scalars = list(finite_scalars(value))
    if scalars:
        import math

        if not all(math.isfinite(item) for item in scalars):
            raise ReproductionError(f"non-finite metric in {path}")
    return sha256(path)


def collect(
    family: str,
    inventory: Path,
    results_root: Path,
    output: Path,
    require_subgroups: bool,
    repository: Path,
) -> None:
    rows = load_inventory(inventory)
    expected = task_rows(family)
    canonical = [{key: str(value) for key, value in row.items()} for row in expected]
    if rows != canonical:
        raise ReproductionError(
            "inventory does not match the frozen matrix for this collector version"
        )
    records = []
    for row in rows:
        metric = metric_path(results_root, row)
        if not metric.is_file():
            raise ReproductionError(f"missing result: {metric}")
        record = dict(row)
        record["metrics_path"] = str(metric)
        record["metrics_sha256"] = validate_pickle(metric)
        if require_subgroups:
            subgroup = subgroup_path(results_root, row)
            if not subgroup.is_file():
                raise ReproductionError(f"missing subgroup result: {subgroup}")
            record["subgroup_path"] = str(subgroup)
            record["subgroup_sha256"] = validate_pickle(subgroup)
        records.append(record)

    key_counts = Counter(
        (row["dataset"], row["method"], row["estimator"], row["seed"])
        for row in records
    )
    if set(key_counts.values()) != {1}:
        raise ReproductionError("duplicate scientific task key in completed panel")

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    fields = list(records[0])
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    tmp.replace(output)
    report = {
        "schema": "pcs-uq-completion-report-v1",
        "status": "complete",
        "family": family,
        "repository_revision": git_revision(repository),
        "contract_sha256": sha256(Path(__file__).resolve()),
        "inventory_path": str(inventory),
        "inventory_sha256": sha256(inventory),
        "results_root": str(results_root),
        "task_count": len(records),
        "completed_rows_path": str(output),
        "completed_rows_sha256": sha256(output),
        "require_subgroups": require_subgroups,
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"PCS_UQ_COMPLETE family={family} tasks={len(records)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory_parser = subparsers.add_parser("inventory")
    inventory_parser.add_argument(
        "--family", choices=("regression", "classification"), required=True
    )
    inventory_parser.add_argument("--output", type=Path, required=True)
    inventory_parser.add_argument("--repository", type=Path, default=Path("."))

    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument(
        "--family", choices=("regression", "classification"), required=True
    )
    collect_parser.add_argument("--inventory", type=Path, required=True)
    collect_parser.add_argument("--results-root", type=Path, required=True)
    collect_parser.add_argument("--output", type=Path, required=True)
    collect_parser.add_argument("--require-subgroups", action="store_true")
    collect_parser.add_argument("--repository", type=Path, default=Path("."))

    args = parser.parse_args()
    if args.command == "inventory":
        write_inventory(args.family, args.output, args.repository)
    else:
        collect(
            args.family,
            args.inventory,
            args.results_root,
            args.output,
            args.require_subgroups,
            args.repository,
        )


if __name__ == "__main__":
    main()
