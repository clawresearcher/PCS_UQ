"""Tests for fail-closed PCS-UQ task inventory and collection."""

from __future__ import annotations

import csv
import pickle
from pathlib import Path

import pytest

from experiments.scripts import reproduction_contract as contract


def test_frozen_task_counts_and_unique_keys() -> None:
    regression = contract.task_rows("regression")
    classification = contract.task_rows("classification")
    assert len(regression) == 12_580
    assert len(classification) == 1_020
    for rows in (regression, classification):
        keys = {
            (row["dataset"], row["method"], row["estimator"], row["seed"])
            for row in rows
        }
        assert len(keys) == len(rows)


def write_inventory(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_collect_fails_on_first_missing_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = contract.task_rows("classification")[:2]
    monkeypatch.setattr(contract, "task_rows", lambda family: rows)
    inventory = tmp_path / "tasks.csv"
    write_inventory(inventory, rows)
    with pytest.raises(contract.ReproductionError, match="missing result"):
        contract.collect(
            "classification",
            inventory,
            tmp_path / "results",
            tmp_path / "complete.csv",
            False,
            Path("."),
        )
    assert not (tmp_path / "complete.csv").exists()
    assert not (tmp_path / "complete.json").exists()


def test_collect_certifies_exact_complete_panel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = contract.task_rows("classification")[:2]
    monkeypatch.setattr(contract, "task_rows", lambda family: rows)
    monkeypatch.setattr(contract, "git_revision", lambda root: "sha256:test")
    inventory = tmp_path / "tasks.csv"
    write_inventory(inventory, rows)
    results = tmp_path / "results"
    for row in rows:
        path = contract.metric_path(
            results, {key: str(value) for key, value in row.items()}
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump({"coverage": 0.9, "mean_width": 1.0}, handle)
    completed = tmp_path / "complete.csv"
    contract.collect(
        "classification", inventory, results, completed, False, Path(".")
    )
    assert completed.is_file()
    assert completed.with_suffix(".json").is_file()


def test_collect_rejects_inventory_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canonical = contract.task_rows("classification")[:2]
    rows = [dict(row) for row in canonical]
    monkeypatch.setattr(contract, "task_rows", lambda family: canonical)
    rows[0]["seed"] = 999
    inventory = tmp_path / "tasks.csv"
    write_inventory(inventory, rows)
    with pytest.raises(contract.ReproductionError, match="frozen matrix"):
        contract.collect(
            "classification",
            inventory,
            tmp_path / "results",
            tmp_path / "complete.csv",
            False,
            Path("."),
        )
