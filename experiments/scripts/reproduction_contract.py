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

IDENTITY_SCHEMA = "pcs-uq-identity-v1"
INVENTORY_FIELDS = (
    "task_id",
    "family",
    "dataset",
    "method",
    "method_name",
    "estimator",
    "seed",
    "train_size",
)
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
REGRESSION_REDUCED_ESTIMATORS = ("__candidate_pool__",)
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
CLASSIFICATION_REDUCED_ESTIMATORS = ("__candidate_pool__",)
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


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def typed_hash(domain: str, value: Any) -> str:
    preimage = domain.encode("ascii") + b"\0" + canonical_json(value)
    return f"sha256:{hashlib.sha256(preimage).hexdigest()}"


def file_hash(path: Path) -> str:
    return f"sha256:{sha256(path)}"


def resolved_universe(repository: Path, universe_id: str) -> dict[str, Any]:
    manifest = json.loads(
        (repository / "experiments/manifests/astra_universes.json").read_text()
    )
    try:
        decisions = manifest["universes"][universe_id]
    except KeyError as exc:
        raise ReproductionError(f"unknown universe: {universe_id}") from exc
    return {
        "schema": IDENTITY_SCHEMA,
        "analysis": manifest["analysis"],
        "source": manifest["source"],
        "decisions": decisions,
    }


def universe_identity(repository: Path, universe_id: str) -> tuple[str, dict[str, Any]]:
    value = resolved_universe(repository, universe_id)
    return typed_hash("astra-universe-v1", value), value


def task_identity(row: dict[str, str], inventory_hash: str) -> str:
    coordinates = {key: row[key] for key in INVENTORY_FIELDS}
    return typed_hash(
        "pcs-uq-task-v1",
        {"inventory_hash": inventory_hash, "coordinates": coordinates},
    )


def git_revision(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


SCIENTIFIC_SOURCE_PATTERNS = (
    "src/**/*.py",
    "experiments/configs/*.py",
    "experiments/manifests/astra_universes.json",
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


def atomic_pickle_dump(value: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary).replace(path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


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
        "identity_schema": IDENTITY_SCHEMA,
        "universe_id": "current_repository",
        "universe_hash": universe_identity(repository, "current_repository")[0],
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
        "identity_schema",
        "universe_id",
        "universe_hash",
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
    if "parent_inventory" in metadata:
        parent = Path(metadata["parent_inventory"])
        if not parent.is_absolute():
            parent = repository / parent
        if sha256(parent) != metadata.get("parent_inventory_sha256"):
            raise ReproductionError("parent inventory hash does not match selection metadata")
    expected_universe_hash, _ = universe_identity(repository, metadata["universe_id"])
    if metadata["identity_schema"] != IDENTITY_SCHEMA:
        raise ReproductionError("inventory uses an unsupported identity schema")
    if metadata["universe_hash"] != expected_universe_hash:
        raise ReproductionError("inventory universe hash does not match ASTRA decisions")
    rows = load_inventory(path)
    if metadata["task_count"] != len(rows):
        raise ReproductionError("inventory row count does not match metadata")
    return rows


def artifact_provenance_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".provenance.json")


def output_id_for(family: str, kind: str) -> str:
    if family == "regression":
        return (
            "regression_seed_metrics"
            if kind == "metrics"
            else "regression_subgroup_metrics"
        )
    return "classification_seed_metrics"


def provenance_payload(
    row: dict[str, str],
    inventory: Path,
    repository: Path,
    kind: str,
) -> dict[str, Any]:
    metadata = json.loads(inventory.with_suffix(".json").read_text())
    output_id = output_id_for(row["family"], kind)
    return {
        "schema": "pcs-uq-artifact-provenance-v2",
        "output_id": output_id,
        "task_hash": task_identity(row, metadata["task_inventory_sha256"]),
        "task": {key: row[key] for key in INVENTORY_FIELDS},
        "inventory_hash": f"sha256:{metadata['task_inventory_sha256']}",
        "contract_hash": f"sha256:{metadata['contract_sha256']}",
        "scientific_source_hash": f"sha256:{metadata['scientific_source_sha256']}",
        "universe_id": metadata["universe_id"],
        "universe_hash": metadata["universe_hash"],
        "producer_revision": git_revision(repository),
    }


def write_artifact_provenance(
    row: dict[str, str],
    inventory: Path,
    repository: Path,
    paths: dict[str, Path],
) -> None:
    for kind, path in paths.items():
        payload = provenance_payload(row, inventory, repository, kind)
        record = {
            **payload,
            "artifact_kind": kind,
            "artifact_hash": file_hash(path),
        }
        atomic_write_text(
            artifact_provenance_path(path),
            json.dumps(record, indent=2) + "\n",
        )


def verify_artifact_provenance(
    kind: str,
    path: Path,
    row: dict[str, str],
    inventory: Path,
    repository: Path,
) -> dict[str, Any]:
    sidecar = artifact_provenance_path(path)
    if not sidecar.is_file():
        raise ReproductionError(f"missing producer provenance: {sidecar}")
    try:
        observed = json.loads(sidecar.read_text())
    except Exception as exc:
        raise ReproductionError(f"invalid producer provenance: {sidecar}") from exc
    expected = {
        **provenance_payload(row, inventory, repository, kind),
        "artifact_kind": kind,
        "artifact_hash": file_hash(path),
    }
    if observed != expected:
        raise ReproductionError(f"producer provenance mismatch: {sidecar}")
    return observed


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
    """Explicit cleanup helper; collection failures never delete prior valid evidence."""
    output.unlink(missing_ok=True)
    output.with_suffix(".json").unlink(missing_ok=True)


def build_collection_manifest(
    family: str,
    rows: list[dict[str, str]],
    metadata: dict[str, Any],
    repository: Path,
    require_subgroups: bool,
    observed_members: list[dict[str, str]],
) -> dict[str, Any]:
    expected_members = []
    kinds = (
        ("metrics", "subgroup_metrics")
        if family == "regression" and require_subgroups
        else ("metrics",)
        if family == "regression"
        else ("metrics", "full_metrics", "class_metrics", "full_class_metrics")
    )
    for row in rows:
        for kind in kinds:
            expected_members.append(
                {
                    "output_id": output_id_for(family, kind),
                    "task_hash": task_identity(row, metadata["task_inventory_sha256"]),
                    "artifact_kind": kind,
                }
            )
    expected_members = sorted(
        expected_members,
        key=lambda item: (item["output_id"], item["task_hash"], item["artifact_kind"]),
    )
    observed_members = sorted(
        observed_members,
        key=lambda item: (item["output_id"], item["task_hash"], item["artifact_kind"]),
    )
    universe_hash, universe = universe_identity(repository, metadata["universe_id"])
    return {
        "schema": "pcs-uq-output-collection-v2",
        "output_ids": sorted({output_id_for(family, kind) for kind in kinds}),
        "universe_hash": universe_hash,
        "universe": universe,
        "inventory_hash": f"sha256:{metadata['task_inventory_sha256']}",
        "scientific_source_hash": f"sha256:{metadata['scientific_source_sha256']}",
        "contract_hash": f"sha256:{metadata['contract_sha256']}",
        "expected_members": expected_members,
        "observed_members": observed_members,
        "validation": {
            "status": "complete" if len(expected_members) == len(observed_members) else "incomplete",
            "expected_count": len(expected_members),
            "observed_count": len(observed_members),
            "omitted": [
                item for item in expected_members if not any(
                    member["output_id"] == item["output_id"]
                    and member["task_hash"] == item["task_hash"]
                    and member["artifact_kind"] == item["artifact_kind"]
                    for member in observed_members
                )
            ],
        },
    }


def collect(
    family: str,
    inventory: Path,
    results_root: Path,
    output: Path,
    require_subgroups: bool,
    repository: Path,
) -> None:
    rows = load_and_bind_inventory(inventory, repository)
    metadata = json.loads(inventory.with_suffix(".json").read_text())
    if metadata["family"] != family:
        raise ReproductionError("inventory metadata family does not match collector family")
    if "parent_inventory" not in metadata:
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
            artifact_hash = validate_artifact(kind, path, family)
            provenance = verify_artifact_provenance(
                kind, path, row, inventory, repository
            )
            record[f"{kind}_path"] = str(path)
            record[f"{kind}_sha256"] = artifact_hash
            record[f"{kind}_task_hash"] = provenance["task_hash"]
            record[f"{kind}_output_id"] = provenance["output_id"]
        records.append(record)

    key_counts = Counter(
        (row["dataset"], row["method"], row["estimator"], row["seed"])
        for row in records
    )
    if set(key_counts.values()) != {1}:
        raise ReproductionError("duplicate scientific task key in completed panel")

    observed_members = []
    for record in records:
        for kind in artifact_paths(results_root, record):
            if f"{kind}_sha256" not in record:
                continue
            observed_members.append(
                {
                    "output_id": record[f"{kind}_output_id"],
                    "task_hash": record[f"{kind}_task_hash"],
                    "artifact_kind": kind,
                    "artifact_hash": f"sha256:{record[f'{kind}_sha256']}",
                }
            )
    collection_manifest = build_collection_manifest(
        family,
        rows,
        metadata,
        repository,
        require_subgroups,
        observed_members,
    )
    validation = collection_manifest["validation"]
    if validation["status"] != "complete" or validation["omitted"]:
        raise ReproductionError("observed artifacts do not match expected collection")
    universe_hash = collection_manifest["universe_hash"]
    collection_hash = typed_hash("astra-collection-v1", collection_manifest)

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
            "schema": "pcs-uq-completion-report-v3",
            "status": "complete",
            "family": family,
            "universe_id": metadata["universe_id"],
            "universe_hash": universe_hash,
            "collection_hash": collection_hash,
            "collection_manifest": collection_manifest,
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
        raise
    print(f"PCS_UQ_COMPLETE family={family} tasks={len(records)}")


def verify_completion_report(
    report_path: Path,
    completed_rows_path: Path,
    family: str,
    repository: Path,
) -> dict[str, Any]:
    try:
        report = json.loads(report_path.read_text())
    except Exception as exc:
        raise ReproductionError(f"invalid completion report: {report_path}") from exc
    if report.get("schema") != "pcs-uq-completion-report-v3":
        raise ReproductionError("unsupported completion report schema")
    if report.get("status") != "complete" or report.get("family") != family:
        raise ReproductionError("completion report status/family mismatch")
    if report.get("completed_rows_sha256") != sha256(completed_rows_path):
        raise ReproductionError("completed rows differ from completion report")

    inventory = Path(report.get("inventory_path", ""))
    if not inventory.is_file():
        raise ReproductionError("completion inventory is unavailable")
    if report.get("inventory_sha256") != sha256(inventory):
        raise ReproductionError("completion inventory bytes differ")
    if report.get("inventory_metadata_sha256") != sha256(inventory.with_suffix(".json")):
        raise ReproductionError("completion inventory metadata differs")
    rows = load_and_bind_inventory(inventory, repository)
    metadata = json.loads(inventory.with_suffix(".json").read_text())
    if report.get("task_count") != len(rows):
        raise ReproductionError("completion task count differs from inventory")
    if report.get("scientific_source_sha256") != repository_tree_hash(repository):
        raise ReproductionError("completion source binding differs")
    if report.get("contract_sha256") != sha256(Path(__file__).resolve()):
        raise ReproductionError("completion contract binding differs")

    with completed_rows_path.open(newline="") as handle:
        completed = list(csv.DictReader(handle))
    if len(completed) != len(rows):
        raise ReproductionError("completed rows do not match inventory cardinality")
    canonical_rows = [
        {key: str(value) for key, value in row.items()}
        for row in rows
    ]
    for expected, observed in zip(canonical_rows, completed, strict=True):
        if any(observed.get(key) != expected[key] for key in INVENTORY_FIELDS):
            raise ReproductionError("completed rows do not match inventory coordinates")

    observed_members = []
    for record in completed:
        for kind in ("metrics", "subgroup_metrics", "full_metrics", "class_metrics", "full_class_metrics"):
            hash_key = f"{kind}_sha256"
            if hash_key not in record or not record[hash_key]:
                continue
            output_id = record.get(f"{kind}_output_id")
            task_hash = record.get(f"{kind}_task_hash")
            if output_id != output_id_for(family, kind) or not task_hash:
                raise ReproductionError("completed member identity is malformed")
            observed_members.append(
                {
                    "output_id": output_id,
                    "task_hash": task_hash,
                    "artifact_kind": kind,
                    "artifact_hash": f"sha256:{record[hash_key]}",
                }
            )

    expected_manifest = build_collection_manifest(
        family,
        rows,
        metadata,
        repository,
        bool(report.get("require_subgroups")),
        observed_members,
    )
    if report.get("collection_manifest") != expected_manifest:
        raise ReproductionError("completion manifest is not canonical for its inventory")
    if report.get("collection_hash") != typed_hash("astra-collection-v1", expected_manifest):
        raise ReproductionError("collection hash does not match canonical manifest")
    expected_universe_hash, _ = universe_identity(repository, report["universe_id"])
    if report.get("universe_hash") != expected_universe_hash:
        raise ReproductionError("completion report universe differs from ASTRA")
    return report


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
