# Output identity across universes

An ASTRA output declaration is a family of potential artifacts, not one blob. A
universe fixes the decisions for one logical output slot. This fork uses the
following implemented identity model.

## Typed canonical hashes

Every structured identity is canonical JSON encoded as UTF-8 with sorted keys,
compact separators, preserved Unicode, rejected NaN/Infinity, and an explicit
schema version. Hash domains are separated:

```text
H(domain, value) = sha256(ASCII(domain) || NUL || canonical_json(value))
```

The current domains are `astra-universe-v1`, `pcs-uq-task-v1`, and
`astra-collection-v1`. Artifact files use `sha256` over the exact file bytes; this
identifies the serialized blob, not a format-independent scientific value.

## Logical slots and materializations

```text
universe_hash = H("astra-universe-v1", {
  schema, analysis ID, immutable source origin/base revision, complete decision map
})

task_hash = H("pcs-uq-task-v1", {
  inventory_hash, complete task coordinates
})

logical_slot = (output_id, universe_hash, task_hash, artifact_kind)
materialization = (logical_slot, artifact_hash, provenance)
```

A logical slot is not itself a materialization identity. Reruns with different
bytes occupy the same slot but have different materializations. Producer
provenance binds each pickle to its ASTRA output ID, task, inventory, contract,
scientific-source hash, universe hash, artifact kind, exact artifact hash, and
producer revision.

Two universes may produce byte-identical files. Those files share an artifact hash
but retain distinct slots and provenance. Equal bytes never imply equal method or
interpretation.

## Complete collections

The completion report embeds a location-free canonical collection manifest:

```text
collection_hash = H("astra-collection-v1", {
  schema,
  output_ids,
  universe_hash and resolved universe,
  inventory/scientific-source/contract hashes,
  expected members: sorted (output_id, task_hash, artifact_kind),
  observed members: sorted (output_id, task_hash, artifact_kind, artifact_hash),
  validation: status, expected_count, observed_count, omissions
})
```

Mutable filesystem paths live only in the paired completed-row CSV. They are not
part of `collection_hash`. Strict consumers reconstruct expected membership from
the bound inventory and output-ID rules, compare the entire CSV against that
inventory, and rehash every member before reading it. A fabricated or smaller
self-reported collection therefore cannot certify itself.

The CSV/report pair cannot be renamed as one atomic filesystem operation. The JSON
report authenticates the CSV hash, and consumers must verify the pair. A crash
between publication renames is detectable and cannot pass strict aggregation.

## Multiverses

A multiverse collection uses the same collection construction, but expected and
observed members additionally include each member's `universe_hash`. No hash is
assigned to a declaration or hypothetical Cartesian product. This repository's
broad regression and ablation sensitivity spaces remain unmaterialized because the
current runners do not parameterize every declared decision or provide
collision-free storage for that product.

`experiments/manifests/astra_universes.json` is the generated runtime projection
of the ASTRA decisions plus immutable repository-origin metadata. `astra.yaml`
remains the authoring authority; release verification regenerates and compares the
projection before publishing evidence.

## Relationships

Relationships address a materialization reference, not a bare blob hash:

```yaml
rel:
  - predicate: same_method
    object:
      output: regression_seed_metrics
      universe_hash: sha256:...
      task_hash: sha256:...
      artifact_hash: sha256:...
```

Predicates such as `same_method`, `same_implementation`, `extends`, `supersedes`,
and `contradicts` are provenance assertions. `almost_same_content` additionally
requires a named comparison function, version, parameters, and threshold; it
cannot be inferred from SHA-256.

## Current scope

- Current regression/classification manifest tasks receive producer-bound
  provenance sidecars and can enter strict completion collections.
- Historical paper-era pickles are comparison evidence at an immutable Git commit,
  not authenticated outputs of the current universe.
- Classification primary/full and marginal/classwise pickles remain distinct
  artifact kinds inside one task bundle and receive distinct content hashes.
- Full sensitivity multiverses have no artifact or collection hashes until an
  executable constrained inventory materializes them.
