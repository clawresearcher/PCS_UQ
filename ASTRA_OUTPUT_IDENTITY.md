# Output identity across universes

An ASTRA output declaration is a **family of potential artifacts**, not one blob.
A universe fixes the decisions needed to materialize one member of that family.
Therefore a single `output.hash` is generally wrong whenever an output can exist in
more than one universe.

## Proposed identity model

For a materialized artifact `A` in universe `U`:

```text
artifact_hash(A) = sha256(canonical output bytes)
universe_hash(U) = sha256(canonical resolved decision map)
instance_key     = (output_id, universe_hash, projection coordinates)
```

The artifact hash remains purely content-addressed and globally reusable. The
universe hash identifies the scientific choice context. `output_id` is only the
project-local semantic role. A mutable path or URL is only a location.

A materialization manifest should therefore look like:

```yaml
output: regression_seed_metrics
universe:
  id: current_repository
  hash: sha256:...
  decisions:
    seed_policy: paper_777_786
    training_cap: uncapped_current
artifact:
  hash: sha256:...
  location: experiments/results/reg_max/...
provenance:
  inputs:
    regression_data: sha256:...
  implementation: sha256:...
```

If two universes produce byte-identical output, they intentionally share the same
`artifact.hash`, while retaining two instance records with different universe
hashes. That is deduplication, not identity collapse: same bytes do not prove the
same method or interpretation.

## Collections and multiverses

A projected output over many universes should not pretend to have one ordinary
artifact hash. It is a collection with a canonical manifest:

```text
collection_hash = sha256(sorted canonical records of
  (universe_hash, projection coordinates, artifact_hash))
```

The collection hash changes when membership or any member content changes. It does
not hash mutable locations. Completeness belongs in this manifest too: expected
coordinates, observed coordinates, and the fail-closed validation report.

## Relationships

Relationships should point to content hashes (or collection hashes), with optional
universe context:

```yaml
rel:
  - predicate: almost_same_content
    object: sha256:...
    universe: sha256:...
  - predicate: same_method
    object: sha256:...
  - predicate: same_implementation
    object: sha256:...
  - predicate: extends
    object: sha256:...
  - predicate: supersedes
    object: sha256:...
  - predicate: contradicts
    object: sha256:...
```

These predicates are assertions backed by provenance, not consequences of hash
equality. `almost_same_content` also needs a declared comparison function and
threshold; it cannot be inferred from SHA-256.

## Rule for this fork

- Per-task pickle hashes identify output bytes.
- The task inventory plus resolved scientific-source hash identifies the current
  execution universe/projection contract.
- A completion report is a collection manifest over every expected task key and
  artifact hash.
- Paper-era capped and current uncapped outputs never share an instance identity,
  even if a particular pickle happens to be byte-identical.
- Multi-universe projections remain unmaterialized until they have explicit member
  manifests; ASTRA declarations alone do not receive fabricated output hashes.
