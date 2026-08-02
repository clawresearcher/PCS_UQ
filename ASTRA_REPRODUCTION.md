# PCS-UQ reproduction status

This fork now has an executable, fail-closed starting point. It does **not** yet
claim to reproduce the paper.

## Executed locally

A fresh Python 3.11 virtual environment installed `requirements.txt`, after which
two inexpensive real tasks completed, while a real PCS-OOB task reached the
1,000-bootstrap fit path and was stopped after ten minutes at 39/1,000 models:

- regression: Parkinsons, split conformal, OLS, seed 777;
- regression: Parkinsons, PCS-OOB, XGBoost, seed 777 (timeout, no artifact);
- classification: Yeast, APS, logistic regression, seed 777.

The metrics, hashes, and one sanity comparison against the deleted historical
regression aggregate are recorded in `experiments/reproduction/local_pilot.json`.
A one-seed check is not a paper reproduction verdict.

A complete authenticated ten-seed Airfoil split-conformal/XGBoost panel also ran
for seeds 777-786. Its strict collection hash is
`sha256:9f6f4565b7d0332815e4e3054c3dba3ae74b24f08ec0498f530f8e62fb7ba639`.
The current aggregate matches historical coverage exactly; width statistics differ
only around `1e-6`. See
`experiments/reproduction/pilot_airfoil_split_xgb_comparison.json`. This remains a
mixed-protocol comparison because the current runner is uncapped while the
historical run used the paper-era 5,000-row cap.

## Frozen task matrices

The current repository submitters expand to:

- 12,580 regression tasks in `experiments/manifests/regression_tasks.csv`;
- 1,020 classification tasks in `experiments/manifests/classification_tasks.csv`.

These are intentionally the repository's current matrices, not a claim that every
cell belongs in the paper's displayed method roster. The paper-facing subset and
historical/current implementation differences remain explicit ASTRA decisions.

Generate them deterministically with:

```bash
python experiments/scripts/reproduction_contract.py inventory \
  --family regression --output experiments/manifests/regression_tasks.csv
python experiments/scripts/reproduction_contract.py inventory \
  --family classification --output experiments/manifests/classification_tasks.csv
```

Run one frozen row locally with:

```bash
python -m experiments.scripts.run_manifest_task \
  --inventory experiments/manifests/regression_tasks.csv --task-id 0
```

On Slurm, submit the exact array rather than rebuilding the shell matrix:

```bash
export PCS_UQ_TASKS=experiments/manifests/regression_tasks.csv
sbatch --array=0-12579 experiments/scripts/submit_manifest_array.sh

export PCS_UQ_TASKS=experiments/manifests/classification_tasks.csv
sbatch --array=0-1019 experiments/scripts/submit_manifest_array.sh
```

The array task exits nonzero when Python fails. The legacy submit scripts were also
changed to `set -euo pipefail`, so their final `echo` can no longer hide failure.

## Completion gates

Do not aggregate or compare figures until these commands succeed:

```bash
python experiments/scripts/reproduction_contract.py collect \
  --family regression \
  --inventory experiments/manifests/regression_tasks.csv \
  --results-root experiments/results/reg_max \
  --require-subgroups \
  --output experiments/reproduction/regression_completed_rows.csv

python experiments/scripts/reproduction_contract.py collect \
  --family classification \
  --inventory experiments/manifests/classification_tasks.csv \
  --results-root experiments/results/class_max \
  --output experiments/reproduction/classification_completed_rows.csv
```

The collector fails on the first missing, duplicate, reused, unauthenticated,
unreadable, malformed, out-of-domain, empty, or non-finite artifact. Each artifact
must carry a producer provenance sidecar bound to the ASTRA output ID, task,
inventory, contract, scientific-source hash, universe hash, kind, and exact bytes.
The manifest runner refuses all pre-existing result or sidecar paths rather than
relabeling them as a fresh execution. Only a complete panel gets a completed-row
table and matching hash-bound JSON report. Strict consumers reconstruct expected
membership from the bound inventory, validate the complete CSV, and rehash every
member before reading. A failed collection preserves any prior valid publication;
new publications use unique staged paths.

Output hashes are scoped by materialization rather than attached blindly to the
family-level ASTRA output declaration. See `ASTRA_OUTPUT_IDENTITY.md` for the
artifact-hash + universe-hash model and canonical multiverse collection hashes.

## Historical comparison warning

Paper-era aggregate pickles are absent from the current repository. They are
available at immutable commit `c06235de065095b8f9bf568c8f515e81a485b87b`
(the parent of deletion commit `93c4aced`) and are registered in `astra.yaml` as
comparison-only external evidence. Current
regression code writes uncapped runs to `reg_max`; those bytes are not assumed to
be equivalent to the historical capped artifacts.

No SCF job has been started from this machine because cluster credentials are not
available here.
