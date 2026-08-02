#!/usr/bin/env python3
"""Select a deterministic task subset and preserve its provenance binding."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from experiments.scripts.reproduction_contract import (
    IDENTITY_SCHEMA,
    atomic_write_text,
    git_revision,
    load_and_bind_inventory,
    repository_tree_hash,
    sha256,
    universe_identity,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--estimator", required=True)
    args = parser.parse_args()

    rows = load_and_bind_inventory(args.inventory, args.repository)
    selected = [
        row
        for row in rows
        if row["dataset"] == args.dataset
        and row["method"] == args.method
        and row["estimator"] == args.estimator
    ]
    selected = [dict(row, task_id=str(index)) for index, row in enumerate(selected)]
    if not selected:
        raise SystemExit("selection matched no tasks")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=selected[0].keys(), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(selected)

    parent = json.loads(args.inventory.with_suffix(".json").read_text())
    universe_hash, _ = universe_identity(args.repository, parent["universe_id"])
    metadata = {
        "schema": "pcs-uq-task-inventory-v2",
        "family": parent["family"],
        "selection": {
            "dataset": args.dataset,
            "method": args.method,
            "estimator": args.estimator,
        },
        "parent_inventory": str(args.inventory),
        "parent_inventory_sha256": sha256(args.inventory),
        "repository_revision": git_revision(args.repository),
        "scientific_source_sha256": repository_tree_hash(args.repository),
        "contract_sha256": parent["contract_sha256"],
        "task_count": len(selected),
        "task_inventory_sha256": sha256(args.output),
        "seeds": sorted({int(row["seed"]) for row in selected}),
        "train_size": parent["train_size"],
        "identity_schema": IDENTITY_SCHEMA,
        "universe_id": parent["universe_id"],
        "universe_hash": universe_hash,
    }
    atomic_write_text(
        args.output.with_suffix(".json"),
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
    )
    print(f"PCS_UQ_SELECTION_OK tasks={len(selected)} sha256={sha256(args.output)}")


if __name__ == "__main__":
    main()
