#!/usr/bin/env python3
"""Aggregate only artifacts authenticated by a complete PCS-UQ report."""

from __future__ import annotations

import argparse
import csv
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from experiments.scripts.reproduction_contract import (
    ReproductionError,
    atomic_pickle_dump,
    artifact_paths,
    file_hash,
    verify_completion_report,
)


def aggregate_dicts(values: list[dict[str, Any]]) -> dict[str, float]:
    keys = set(values[0])
    if any(set(value) != keys for value in values):
        raise ReproductionError("metric schemas differ across seeds")
    result: dict[str, float] = {}
    for key in sorted(keys):
        array = np.asarray([value[key] for value in values], dtype=float)
        if not np.isfinite(array).all():
            raise ReproductionError(f"non-finite metric during aggregation: {key}")
        result[f"{key}_mean"] = float(np.mean(array))
        result[f"{key}_std"] = float(np.std(array))
    return result


def aggregate_subgroups(values: list[dict[str, Any]]) -> dict[str, Any]:
    features = set(values[0])
    if any(set(value) != features for value in values):
        raise ReproductionError("subgroup feature schemas differ across seeds")
    result = {}
    for feature in sorted(features):
        groups = set(values[0][feature])
        if any(set(value[feature]) != groups for value in values):
            raise ReproductionError(f"subgroup bins differ for {feature}")
        result[feature] = {
            group: aggregate_dicts([value[feature][group] for value in values])
            for group in sorted(groups, key=str)
        }
    return result


def load_authenticated(
    completed_rows: Path,
    report: Path,
    family: str,
    kind: str,
    repository: Path,
    seeds_per_cell: int = 10,
) -> tuple[dict[tuple[str, str, str], list[dict[str, Any]]], dict[str, Any]]:
    report_value = verify_completion_report(report, completed_rows, family, repository)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    with completed_rows.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        path = Path(row[f"{kind}_path"])
        expected_hash = f"sha256:{row[f'{kind}_sha256']}"
        if file_hash(path) != expected_hash:
            raise ReproductionError(f"artifact changed after completion: {path}")
        with path.open("rb") as handle:
            value = pickle.load(handle)
        grouped[(row["dataset"], row["method"], row["estimator"])].append(value)
    if any(len(values) != seeds_per_cell for values in grouped.values()):
        raise ReproductionError(
            f"strict aggregate requires exactly {seeds_per_cell} seeds per cell"
        )
    return grouped, report_value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("regression", "classification"), required=True)
    parser.add_argument("--completed-rows", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subgroups", action="store_true")
    parser.add_argument("--repository", type=Path, default=Path("."))
    args = parser.parse_args()

    kind = "subgroup_metrics" if args.subgroups else "metrics"
    grouped, completion = load_authenticated(
        args.completed_rows, args.report, args.family, kind, args.repository
    )
    result: dict[str, dict[tuple[str, str], Any]] = defaultdict(dict)
    for (dataset, method, estimator), values in sorted(grouped.items()):
        aggregate = aggregate_subgroups(values) if args.subgroups else aggregate_dicts(values)
        result[dataset][(method, estimator)] = aggregate
    payload = {
        "schema": "pcs-uq-strict-aggregate-v1",
        "family": args.family,
        "artifact_kind": kind,
        "collection_hash": completion["collection_hash"],
        "universe_hash": completion["universe_hash"],
        "results": dict(result),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_pickle_dump(payload, args.output)
    print(
        f"PCS_UQ_AGGREGATE_OK family={args.family} kind={kind} cells={len(grouped)}"
    )


if __name__ == "__main__":
    main()
