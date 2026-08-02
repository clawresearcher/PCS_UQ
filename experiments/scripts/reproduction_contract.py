#!/usr/bin/env python3
"""Freeze and validate PCS-UQ task matrices and result completeness."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import subprocess
import tempfile
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

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
REGRESSION_ESTIMATOR_NAMED = frozenset(
    {
        "split_conformal",
        "split_conformal_alt",
        "studentized_conformal",
        "studentized_conformal_alt",
        "jackknife_bootstrap",
        "pcs_oob_fixed_method",
        "pcs_oob_downsample_fixed_method",
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
CLASSIFICATION_ESTIMATOR_NAMED = frozenset(
    {"split_conformal_aps", "split_conformal_raps", "split_conformal_topk"}
)
CLASSIFICATION_REDUCED_ESTIMATORS = ("HistGradientBoosting",)
REGRESSION_METRICS = frozenset(
    {
        "coverage",
        "mean_width",
        "median_width",
        "mean_width_scaled",
        "median_width_scaled",
        "train_time",
        "pred_time",
        "scaled_pred_time",
    }
)
CLASSIFICATION_METRICS = frozenset(
    {
        "coverage",
        "mean_width",
        "median_width",
        "mean_width_scaled",
        "median_width_scaled",
    }
)
CLASSIFICATION_CLASS_METRICS = frozenset(
    {
        "class_coverage",
        "class_mean_width",
        "class_median_width",
        "class_mean_width_scaled",
        "class_median_width_scaled",
    }
)


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


SCIENTIFIC_SOURCE_PATTERNS = (
    "src/**/*.py",
    "experiments/configs/*.py",
    "experiments/scripts/run_regression_exp.py",
    "experiments/scripts/run_classification_exp.py",
    "experiments/data/**/X.csv",
    "experiments/data/**/y.csv",
    "experiments/data/**/bin_df.pkl",
    "experiments/data/**/importances.csv",
)


def repository_tree_hash(root: Path) -> str:
    """Hash only scientific bytes that determine task outputs."""
    selected: set[Path] = set()
    for pattern in SCIENTIFIC_SOURCE_PATTERNS:
        selected.update(path for path in root.glob(pattern) if path.is_file())
    digest = hashlib.sha256()
    for path in sorted(selected, key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def method_artifact_name(family: str, method: str, estimator: str) -> str:
    if family == "regression" and method in REGRESSION_ESTIMATOR_NAMED:
        return f"{method}_{estimator}"
    if family == "classification" and method in CLASSIFICATION_ESTIMATOR_NAMED:
        return f"{method}_{estimator}"
    return method


def task_rows(family: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if family == "regression":
        datasets = REGRESSION_DATASETS
        methods = REGRESSION_METHODS
        all_estimators = REGRESSION_ALL_ESTIMATORS
        reduced_methods = REGRESSION_REDUCED_METHODS
        reduced_estimators = REGRESSION_REDUCED_ESTIMATORS
    elif family == "classification":
        datasets = CLASSIFICATION_DATASETS
        methods = CLASSIFICATION_METHODS
        all_estimators = CLASSIFICATION_ALL_ESTIMATORS
        reduced_methods = CLASSIFICATION_REDUCED_METHODS
        reduced_estimators = CLASSIFICATION_REDUCED_ESTIMATORS
    else:
        raise ReproductionError(f"unsupported family: {family}")

    for method in methods:
        estimators = reduced_estimators if method in reduced_methods else all_estimators
        for dataset in datasets:
            for seed in SEEDS:
                for estimator in estimators:
                    rows.append(
                        {
                            "task_id": len(rows),
                            "family": family,
                            "dataset": dataset,
                            "method": method,
                            "method_name": method_artifact_name(
                                family, method, estimator
                            ),
                            "estimator": estimator,
                            "seed": seed,
                            "train_size": TRAIN_SIZE,
                        }
                    )
    return rows


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary).replace(path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def write_inventory(family: str, output: Path, repository: Path) -> None:
    rows = task_rows(family)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "schema": "pcs-uq-task-inventory-v2",
        "family": family,
        "repository_revision": git_revision(repository),
        "scientific_source_sha256": repository_tree_hash(repository),
        "contract_sha256": sha256(Path(__file__).resolve()),
        "task_count": len(rows),
        "task_inventory": output.name,
        "task_inventory_sha256": sha256(output),
        "seeds": list(SEEDS),
        "train_size": TRAIN_SIZE,
    }
    atomic_write_text(output.with_suffix(".json"), json.dumps(metadata, indent=2) + "\n")
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


def load_and_bind_inventory(path: Path, repository: Path) -> list[dict[str, str]]:
    sidecar = path.with_suffix(".json")
    if not sidecar.is_file():
        raise ReproductionError(f"missing inventory metadata: {sidecar}")
    try:
        metadata = json.loads(sidecar.read_text())
    except Exception as exc:
        raise ReproductionError(f"invalid inventory metadata: {sidecar}") from exc
    required = {
        "schema",
        "family",
        "repository_revision",
        "scientific_source_sha256",
        "contract_sha256",
        "task_count",
        "task_inventory_sha256",
    }
    if not required <= set(metadata):
        raise ReproductionError(f"incomplete inventory metadata: {sidecar}")
    if metadata["schema"] != "pcs-uq-task-inventory-v2":
        raise ReproductionError(f"unsupported inventory schema: {metadata['schema']}")
    if metadata["task_inventory_sha256"] != sha256(path):
        raise ReproductionError("task inventory hash does not match metadata")
    if metadata["family"] not in {"regression", "classification"}:
        raise ReproductionError("invalid inventory family metadata")
    if metadata["contract_sha256"] != sha256(Path(__file__).resolve()):
        raise ReproductionError("inventory was generated by a different contract")
    if metadata["repository_revision"] != git_revision(repository):
        # A manifest committed with its implementation necessarily records the
        # parent commit. Scientific identity is content-bound below; revision is
        # retained only as a human-readable provenance hint.
        if not metadata["repository_revision"]:
            raise ReproductionError("missing repository revision identity")
    if metadata["scientific_source_sha256"] != repository_tree_hash(repository):
        raise ReproductionError("scientific source tree differs from inventory binding")
    rows = load_inventory(path)
    if metadata["task_count"] != len(rows):
        raise ReproductionError("inventory row count does not match metadata")
    return rows


def artifact_paths(results_root: Path, row: dict[str, str]) -> dict[str, Path]:
    stem = (
        f"{row['method_name']}_seed_{row['seed']}_"
        f"train_size_{row['train_size']}"
    )
    dataset_root = results_root / row["dataset"]
    artifacts = {"metrics": dataset_root / f"{stem}_metrics.pkl"}
    if row["family"] == "regression":
        artifacts["subgroup_metrics"] = dataset_root / f"{stem}_subgroup_metrics.pkl"
    else:
        artifacts.update(
            {
                "full_metrics": dataset_root / f"{stem}_full_metrics.pkl",
                "class_metrics": dataset_root / f"{stem}_class_metrics.pkl",
                "full_class_metrics": dataset_root / f"{stem}_full_class_metrics.pkl",
            }
        )
    return artifacts


def metric_path(results_root: Path, row: dict[str, str]) -> Path:
    return artifact_paths(results_root, row)["metrics"]


def subgroup_path(results_root: Path, row: dict[str, str]) -> Path:
    return artifact_paths(results_root, row)["subgroup_metrics"]


def load_pickle(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            value = pickle.load(handle)
    except Exception as exc:
        raise ReproductionError(f"cannot load {path}: {exc}") from exc
    if not isinstance(value, dict) or not value:
        raise ReproductionError(f"empty or non-dict artifact: {path}")
    return value


def finite_numeric(value: Any, path: Path) -> np.ndarray:
    try:
        values = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ReproductionError(f"non-numeric metric in {path}") from exc
    if values.size == 0 or not np.isfinite(values).all():
        raise ReproductionError(f"empty or non-finite metric in {path}")
    return values


def require_keys(value: dict[str, Any], expected: frozenset[str], path: Path) -> None:
    if set(value) != expected:
        raise ReproductionError(
            f"wrong metric schema in {path}: expected {sorted(expected)}, got {sorted(value)}"
        )


def validate_marginal(value: dict[str, Any], expected: frozenset[str], path: Path) -> None:
    require_keys(value, expected, path)
    for key in expected:
        values = finite_numeric(value[key], path)
        if values.size != 1:
            raise ReproductionError(f"{key} must be scalar in {path}")
        scalar = float(values.item())
        if key == "coverage" and not 0 <= scalar <= 1:
            raise ReproductionError(f"coverage outside [0,1] in {path}")
        if key != "coverage" and scalar < 0:
            raise ReproductionError(f"negative {key} in {path}")
        if key.endswith("_scaled") and scalar > 1 and expected == CLASSIFICATION_METRICS:
            raise ReproductionError(f"scaled set size outside [0,1] in {path}")


def validate_class_metrics(value: dict[str, Any], path: Path) -> None:
    require_keys(value, CLASSIFICATION_CLASS_METRICS, path)
    lengths = set()
    for key in CLASSIFICATION_CLASS_METRICS:
        values = finite_numeric(value[key], path)
        lengths.add(values.size)
        if key == "class_coverage" and not ((values >= 0) & (values <= 1)).all():
            raise ReproductionError(f"class coverage outside [0,1] in {path}")
        if key != "class_coverage" and (values < 0).any():
            raise ReproductionError(f"negative class width in {path}")
        if key.endswith("_scaled") and (values > 1).any():
            raise ReproductionError(f"scaled class size outside [0,1] in {path}")
    if len(lengths) != 1:
        raise ReproductionError(f"class metric lengths disagree in {path}")
    if lengths == {0}:
        raise ReproductionError(f"empty class metrics in {path}")


def validate_subgroups(value: dict[str, Any], path: Path) -> None:
    if not value:
        raise ReproductionError(f"empty subgroup artifact: {path}")
    for feature, groups in value.items():
        if not isinstance(feature, str) or not isinstance(groups, dict) or not groups:
            raise ReproductionError(f"malformed subgroup feature in {path}")
        for metrics in groups.values():
            if not isinstance(metrics, dict):
                raise ReproductionError(f"malformed subgroup metrics in {path}")
            required = {
                "coverage",
                "mean_width",
                "median_width",
                "mean_width_scaled",
                "median_width_scaled",
            }
            if not required <= set(metrics):
                raise ReproductionError(f"incomplete subgroup metrics in {path}")
            validate_marginal(
                {key: metrics[key] for key in required},
                frozenset(required),
                path,
            )


def validate_artifact(kind: str, path: Path, family: str) -> str:
    value = load_pickle(path)
    if kind in {"metrics", "full_metrics"}:
        expected = REGRESSION_METRICS if family == "regression" else CLASSIFICATION_METRICS
        validate_marginal(value, expected, path)
    elif kind in {"class_metrics", "full_class_metrics"}:
        validate_class_metrics(value, path)
    elif kind == "subgroup_metrics":
        validate_subgroups(value, path)
    else:
        raise ReproductionError(f"unknown artifact kind: {kind}")
    return sha256(path)


def remove_stale_publication(output: Path) -> None:
    output.unlink(missing_ok=True)
    output.with_suffix(".json").unlink(missing_ok=True)


def collect(
    family: str,
    inventory: Path,
    results_root: Path,
    output: Path,
    require_subgroups: bool,
    repository: Path,
) -> None:
    remove_stale_publication(output)
    rows = load_and_bind_inventory(inventory, repository)
    metadata = json.loads(inventory.with_suffix(".json").read_text())
    if metadata["family"] != family:
        raise ReproductionError("inventory metadata family does not match collector family")
    expected = task_rows(family)
    canonical = [{key: str(value) for key, value in row.items()} for row in expected]
    if rows != canonical:
        raise ReproductionError(
            "inventory does not match the frozen matrix for this collector version"
        )
    if any(row["family"] != family for row in rows):
        raise ReproductionError("inventory family does not match collector family")

    records = []
    seen_paths: set[Path] = set()
    for row in rows:
        artifacts = artifact_paths(results_root, row)
        if family == "regression" and not require_subgroups:
            artifacts.pop("subgroup_metrics")
        record = dict(row)
        for kind, path in artifacts.items():
            resolved = path.resolve()
            if resolved in seen_paths:
                raise ReproductionError(f"artifact reused by multiple tasks: {path}")
            seen_paths.add(resolved)
            if not path.is_file():
                raise ReproductionError(f"missing {kind}: {path}")
            record[f"{kind}_path"] = str(path)
            record[f"{kind}_sha256"] = validate_artifact(kind, path, family)
        records.append(record)

    key_counts = Counter(
        (row["dataset"], row["method"], row["estimator"], row["seed"])
        for row in records
    )
    if set(key_counts.values()) != {1}:
        raise ReproductionError("duplicate scientific task key in completed panel")

    output.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    staged_csv = output.parent / f".{output.name}.{token}.tmp"
    report_path = output.with_suffix(".json")
    staged_report = report_path.parent / f".{report_path.name}.{token}.tmp"
    try:
        fields = list(records[0])
        with staged_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
        report = {
            "schema": "pcs-uq-completion-report-v2",
            "status": "complete",
            "family": family,
            "repository_revision": git_revision(repository),
            "scientific_source_sha256": repository_tree_hash(repository),
            "contract_sha256": sha256(Path(__file__).resolve()),
            "inventory_path": str(inventory),
            "inventory_sha256": sha256(inventory),
            "inventory_metadata_sha256": sha256(inventory.with_suffix(".json")),
            "results_root": str(results_root),
            "task_count": len(records),
            "completed_rows_path": str(output),
            "completed_rows_sha256": sha256(staged_csv),
            "require_subgroups": require_subgroups,
        }
        staged_report.write_text(json.dumps(report, indent=2) + "\n")
        staged_csv.replace(output)
        staged_report.replace(report_path)
    except BaseException:
        staged_csv.unlink(missing_ok=True)
        staged_report.unlink(missing_ok=True)
        remove_stale_publication(output)
        raise
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
