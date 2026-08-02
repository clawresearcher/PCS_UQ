"""Tests for fail-closed PCS-UQ task inventory and collection."""

from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from experiments.scripts import reproduction_contract as contract


def test_frozen_task_counts_names_and_unique_keys() -> None:
    regression = contract.task_rows("regression")
    classification = contract.task_rows("classification")
    assert len(regression) == 12_580
    assert len(classification) == 1_020
    assert regression[0]["method_name"] == "split_conformal_XGBoost"
    assert regression[80]["method_name"] == "split_conformal_MLP"
    assert classification[0]["method_name"] == "split_conformal_aps_LogisticRegression"
    for rows in (regression, classification):
        keys = {
            (row["dataset"], row["method"], row["estimator"], row["seed"])
            for row in rows
        }
        artifact_names = {
            (row["dataset"], row["method_name"], row["seed"]) for row in rows
        }
        assert len(keys) == len(rows)
        assert len(artifact_names) == len(rows)


def write_inventory(
    path: Path,
    rows: list[dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    monkeypatch.setattr(contract, "git_revision", lambda root: "sha256:test")
    monkeypatch.setattr(contract, "repository_tree_hash", lambda root: "sha256:tree")
    metadata = {
        "schema": "pcs-uq-task-inventory-v2",
        "family": rows[0]["family"],
        "repository_revision": "sha256:test",
        "scientific_source_sha256": "sha256:tree",
        "contract_sha256": contract.sha256(Path(contract.__file__)),
        "task_count": len(rows),
        "task_inventory_sha256": contract.sha256(path),
    }
    path.with_suffix(".json").write_text(json.dumps(metadata))


def marginal_metrics(family: str) -> dict[str, float]:
    value = {
        "coverage": 0.9,
        "mean_width": 1.0,
        "median_width": 1.0,
        "mean_width_scaled": 0.1,
        "median_width_scaled": 0.1,
    }
    if family == "regression":
        value.update({"train_time": 1.0, "pred_time": 0.1, "scaled_pred_time": 0.01})
    return value


def class_metrics() -> dict[str, np.ndarray]:
    return {
        "class_coverage": np.array([0.8, 1.0]),
        "class_mean_width": np.array([1.0, 1.2]),
        "class_median_width": np.array([1.0, 1.0]),
        "class_mean_width_scaled": np.array([0.5, 0.6]),
        "class_median_width_scaled": np.array([0.5, 0.5]),
    }


def write_task_artifacts(root: Path, row: dict[str, object]) -> None:
    string_row = {key: str(value) for key, value in row.items()}
    artifacts = contract.artifact_paths(root, string_row)
    for kind, path in artifacts.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if kind in {"metrics", "full_metrics"}:
            value = marginal_metrics(string_row["family"])
        elif kind in {"class_metrics", "full_class_metrics"}:
            value = class_metrics()
        else:
            value = {
                "feature": {
                    "group": {
                        "coverage": 0.9,
                        "mean_width": 1.0,
                        "median_width": 1.0,
                        "mean_width_scaled": 0.1,
                        "median_width_scaled": 0.1,
                    }
                }
            }
        with path.open("wb") as handle:
            pickle.dump(value, handle)


def test_collect_fails_closed_and_removes_stale_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = contract.task_rows("classification")[:2]
    monkeypatch.setattr(contract, "task_rows", lambda family: rows)
    inventory = tmp_path / "tasks.csv"
    write_inventory(inventory, rows, monkeypatch)
    completed = tmp_path / "complete.csv"
    completed.write_text("stale")
    completed.with_suffix(".json").write_text("stale")
    with pytest.raises(contract.ReproductionError, match="missing metrics"):
        contract.collect(
            "classification", inventory, tmp_path / "results", completed, False, Path(".")
        )
    assert not completed.exists()
    assert not completed.with_suffix(".json").exists()


def test_collect_certifies_distinct_complete_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = contract.task_rows("classification")[:2]
    monkeypatch.setattr(contract, "task_rows", lambda family: rows)
    inventory = tmp_path / "tasks.csv"
    write_inventory(inventory, rows, monkeypatch)
    results = tmp_path / "results"
    for row in rows:
        write_task_artifacts(results, row)
    completed = tmp_path / "complete.csv"
    contract.collect("classification", inventory, results, completed, False, Path("."))
    report = json.loads(completed.with_suffix(".json").read_text())
    assert report["status"] == "complete"
    assert report["task_count"] == 2
    with completed.open(newline="") as handle:
        records = list(csv.DictReader(handle))
    assert len({record["metrics_path"] for record in records}) == 2
    assert all("full_class_metrics_sha256" in record for record in records)


def test_collect_rejects_invalid_metric_domain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = contract.task_rows("classification")[:1]
    monkeypatch.setattr(contract, "task_rows", lambda family: rows)
    inventory = tmp_path / "tasks.csv"
    write_inventory(inventory, rows, monkeypatch)
    results = tmp_path / "results"
    write_task_artifacts(results, rows[0])
    metrics = contract.artifact_paths(
        results, {key: str(value) for key, value in rows[0].items()}
    )["metrics"]
    value = marginal_metrics("classification")
    value["coverage"] = 2.0
    with metrics.open("wb") as handle:
        pickle.dump(value, handle)
    with pytest.raises(contract.ReproductionError, match="coverage outside"):
        contract.collect(
            "classification", inventory, results, tmp_path / "complete.csv", False, Path(".")
        )


def test_collect_rejects_inventory_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canonical = contract.task_rows("classification")[:2]
    rows = [dict(row) for row in canonical]
    monkeypatch.setattr(contract, "task_rows", lambda family: canonical)
    rows[0]["seed"] = 999
    inventory = tmp_path / "tasks.csv"
    write_inventory(inventory, rows, monkeypatch)
    with pytest.raises(contract.ReproductionError, match="frozen matrix"):
        contract.collect(
            "classification",
            inventory,
            tmp_path / "results",
            tmp_path / "complete.csv",
            False,
            Path("."),
        )
