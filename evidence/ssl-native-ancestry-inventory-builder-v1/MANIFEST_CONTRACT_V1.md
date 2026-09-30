# Explicit inventory manifest

This input manifest authorizes metadata derivation only. It is not a prefit,
final-selection, ancestry-review or numerical-access approval. ROOT supplies an
exact endpoint list; the CLI never enumerates run outputs or selects a best seed.

```json
{
  "kind": "native_research_inventory_manifest_v1",
  "purpose": "LOCAL_METADATA_DERIVATION_ONLY",
  "execution": "ROOT_LOCAL_COMPLETED_METADATA_AUDIT",
  "evidence_kind": "REAL_TRAIN_DEVELOPMENT_FIT",
  "device": "cpu",
  "owner_session_id": "ROOT_SESSION_ID",
  "output_path": "ABSOLUTE_NEW_OUTPUT_DIRECTORY",
  "split_path": "ABSOLUTE_ROOT/configs/native_ssl_split_v1.json",
  "train": {
    "input_path": "ABSOLUTE_ORIGINAL_TRAIN_NPZ",
    "report_path": "ABSOLUTE_ORIGINAL_TRAIN_REPORT_JSON",
    "identity_cohort_path": "ABSOLUTE_BOUND_COMPLETE_TRAIN_IDENTITY_COHORT_JSON"
  },
  "endpoints": {
    "EXPLICIT_ENDPOINT_NAME": {
      "directory": "ABSOLUTE_COMPLETED_RUN_DIRECTORY",
      "kind": "native_band_replication_ssl_weights_only_inference_v2",
      "method": "shared_ssl",
      "seed": 13,
      "mode": "core_frozen_readout",
      "parent": null,
      "config_path": "ABSOLUTE_ORIGINAL_REVIEWED_CONFIG_JSON",
      "review_path": "ABSOLUTE_ORIGINAL_DISTINCT_PREFIT_REVIEW_JSON"
    }
  },
  "references": {},
  "bindings": {
    "EVERY_ABSOLUTE_REQUIRED_SOURCE_AND_INPUT_FILE": "EXACT_SHA256"
  }
}
```

The example is a schema illustration, not an executable manifest or proposed
scientific recipe. Every directory must contain its actual `run.json`,
`membership.json`, `inference.pt`, `selected_encoder.pt` and `scalers.json`.
All these files, original reviews/configs, every original embedded binding, the
TRAIN corpus/report/cohort, split, CLI and `required_sources()` closure require
exact current SHA256 bindings. Manifest and original identity paths are absolute
regular files without symlinks/reparse points. No provenance command or import is
executed. Original source paths must remain available for this ROOT audit;
portable inference is a separate API and unaffected by this requirement.

The original identity cohort contains `role: train`, exact `input_npz_sha256`,
`split_sha256`, all unique ordered `[deployment, row_id]` pairs and exact
`identities` objects (`archive_id`, `deployment_id`, `site_id`, `source_ids`).
Original report totals and per-source totals, every full fitted membership and
the historical TR18593 reservation must agree. The original 82-receipt cohort
schema is accepted without inventing a mandatory `complete` field; an explicit
false value, missing rows, duplicates or partial count fails. Numerical NPZ
contents are never decoded to repair incomplete provenance.

Supported neural kinds are original V1, Band V1 and Band replication V2 inference
kinds. Band V1 remains seed7; V2 admits only 7/13/23. The saved config must exactly
match its immutable source schema. Modes are `core_frozen_readout`,
`frozen_readout`, `full_finetune` and `direct_end_to_end`. Every inherited endpoint
names an exact supplied parent; hashes in the actual supervised ancestry must
resolve to its completed report and selected artifact. Frozen encoders retain
tensor/buffer and selected-file byte identity. Full-finetuned states require
actual encoder tensor differences and preserve supervised ancestry. Fresh direct
parents cannot acquire an SSL label. A downstream factory that does not admit a
supervised selected parent remains unsupported.

Reference entries use exactly `kind`, `report_path`, `config_path`,
`statistics_path`, `review_path`. Supported reference kinds are `persistence`,
`seasonal24`, `lightgbm15_utf8` and `external_chronos_unknown`. Conventional
config fields are exactly `method`, `history`, `seed`, `feature_schema`.
Statistics fields are exactly `fit_role`, `input_npz_sha256`, `normalization`.
Normalization is `NO_FITTED_NORMALIZATION`; feature schema is
`native_primary_context_source_offsets_v1` for no-fit controls, or
`native_context_mask_metadata_summary_v1` for LightGBM. No-fit statistics use
`fit_role: none` and null input hash. LightGBM uses `fit_role: train`, exact TRAIN
hash and the actual fifteen named booster hashes from its completed report.
These are metadata identities, not newly fitted statistics. Unknown external
config is exactly `kind: external_chronos_unknown` and
`external_model_identity`; its statistics are `ancestry: UNKNOWN_EXTERNAL`.
It receives no clean-local ancestry receipt or supported neural loader claim.

The generated assessment split receipt is an identity transport of the original
split, not a new partition. Generated assessment cohort/fit receipts bind its
actual hash consistently and separately retain `original_split_sha256`. Original
run reviews and original TRAIN cohort remain immutable. All outputs are fresh,
exclusive JSON files. Selection receipts say `NOT_FINAL_SELECTION`; proposed
freeze inputs contain no owner confirmation or approval status. A distinct review
is required before using this inventory in a final freeze or assessment.

ROOT commands, after integration:

```text
python -B -m pytest tests/unit/test_native_ancestry_inventory.py tests/integration/test_native_ancestry_inventory.py --import-mode=importlib -p no:cacheprovider -q
python -B tools/prepare_native_research_inventory.py --manifest ROOT_BOUND_MANIFEST.json --output ROOT_NEW_INVENTORY_DIRECTORY
```

The actual completed-weight command is ROOT-only NOT_RUN in builder. Its process
must remain CPU-only and below 22 GiB RSS. There is no optimizer, model
construction, RNG reset, corpus decoding, booster inference or scientific fit.
