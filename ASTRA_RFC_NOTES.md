# ASTRA registration notes

`astra.yaml` describes the paper-facing tabular PCS-UQ pipeline at the fork's
starting revision, `aagarwal1996/PCS_UQ@7d666b1e4b5940d4ad884d86a60ea1f053b45d75`.

## Paper authority

The scientific rationale and reporting contract were checked against
**PCS-UQ: Uncertainty Quantification via the Predictability-Computability-
Stability Framework**, arXiv:2505.08784v3, especially Sections 3-6 and
Appendices S2-S4. The pinned arXiv source archive has SHA-256
`35e6af0511027124515be93dd8041a676765fc65fe9bb100350d2fc9bfb6cc1e`.

The paper explains the registered decisions:

- prediction-checking excludes poorly predictive algorithms before uncertainty
  is assessed;
- bootstrap datasets represent inter-sample and algorithmic instability;
- OOB predictions avoid withholding a fixed PCS calibration split;
- multiplicative calibration widens locally uncertain cases proportionally;
- `k=1` and `B=1000` were selected using simulations and five pilot datasets;
- conformal baselines are globally oracle-selected by average test performance,
  which deliberately gives them information unavailable in practice;
- 17 regression and 6 multiclass datasets use 80/20 outer splits, 90% target
  coverage, and ten seeds;
- subgroup definitions use important categorical features, natural numerical
  breaks, or quartiles when no natural break exists;
- classification set size is normalized by the number of classes.

## Executable boundary

The committed processed matrices and subgroup artifacts are the reproduction
boundary. `experiments/notebooks/download_process_data_regression.ipynb` records
historical acquisition and cleaning, but it is stateful, includes manual choices,
and references mutable services. A clean rerun of that notebook is not claimed to
reproduce the committed bytes.

The graph follows:

1. `experiments/data/regression/data_*/{X.csv,y.csv,bin_df.pkl,importances.csv}`
   and the selected classification directories;
2. seeded outer train/test splits, training-row caps, method-specific fitting and
   calibration;
3. per-seed marginal, subgroup, classwise, and empty-set-policy pickles;
4. available-case legacy aggregation under `experiments/results/*/aggregated_results/`;
5. notebook-rendered marginal, subgroup, classification, and ablation figures.

## Historical results and current execution

Paper-era aggregate pickles are absent from the current tree; they survive only in
Git history immediately before deletion commit `93c4aced` and are registered as
comparison-only evidence through the `historical_results` source. Current runners
write to `reg_max`, `class_max`, and family-specific ablation directories.

The fork now freezes the current shell matrices (12,580 regression and 1,020
classification tasks), runs those exact rows through a manifest array, and refuses
to emit completion reports for incomplete/corrupt panels. See
`ASTRA_REPRODUCTION.md`. A two-task local pilot has run successfully, but neither
the full matrices nor the paper figures have been reproduced.

## Known paper/code gaps

Registration does not imply that the current repository reproduces arXiv v3.
Before making that claim, tomorrow's cluster run must resolve these differences:

- paper v3 reports J+aB and removes Majority Vote for simplicity, while the
  current regression plotting notebook uses Majority Vote and does not directly
  render J+aB;
- paper subgroup figures likewise report J+aB, while historical notebook
  selections may use Majority Vote;
- current regression jobs write to `experiments/results/reg_max/` and use the
  full outer-training partition; paper-era aggregates now exist only in Git
  history before `93c4aced` and came from an earlier capped implementation;
- paper classification and current fork code both configure PCS with 1,000
  bootstraps, but older committed result pickles may predate that implementation
  state and require provenance checks before comparison;
- the aggregation scripts catch missing files and average available seeds; a
  reproduction must separately require every expected dataset/method/seed cell;
- paper figures are notebook outputs with manual cell state and selections, not
  deterministic batch renderers;
- deep-learning experiments and the modified theoretical PCS procedure have no
  executable pipeline in this repository and are outside this registration.

## Draft ASTRA layer

The root record uses approved draft ASTRA issue #52 and PR #58 semantics:

- in-file named universes and multiverses;
- `artifact#multiverse` projection;
- concrete output `target` paths.

Released `astra-tools` may reject those fields. That expected rejection is not a
scientific failure and must not be hidden. Parse the YAML, audit graph references,
exercise real repository commands where feasible, and report released CLI errors
verbatim.
