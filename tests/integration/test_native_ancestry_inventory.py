"""SYNTHETIC_CORRECTNESS_ONLY: real CPU safe codecs and private Unicode metadata."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

BUILDER = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "native_inventory_synthetic_support",
    BUILDER / "evidence/ssl-native-ancestry-inventory-builder-v1/synthetic_fixture.py",
)
support = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = support
SPEC.loader.exec_module(support)


def fixture():
    return support.Fixture().add("root").finish()


def mutate_artifact(f, transform):
    path = f.root / "root/inference.pt"
    payload = torch.load(path, weights_only=True, map_location="cpu")
    transform(payload)
    torch.save(payload, path)
    manifest = json.loads(f.manifest.read_bytes())
    manifest["bindings"][str(path)] = support.digest(path)
    f.rewrite(f.manifest, manifest)
    run_path = f.root / "root/run.json"
    run = json.loads(run_path.read_bytes())
    run["inference_sha256"] = support.digest(path)
    f.rewrite(run_path, run, rebind=True)


def test_unicode_durable_multigeneration_inventory_and_recursive_contract():
    f = (
        support.Fixture()
        .add("ssl")
        .add("frozen", mode="frozen_readout", parent="ssl")
        .add("fine", mode="full_finetune", parent="frozen")
        .add("direct", method="direct", mode="direct_end_to_end")
        .add("random", method="random_frozen")
        .finish()
    )
    rng = torch.get_rng_state().clone()
    result = f.candidate.derive_inventory(f.manifest, f.output)
    assert torch.equal(rng, torch.get_rng_state())
    assert result["numeric_corpus_decoded"] is False
    assert result["status"] == "DERIVED_METADATA_NOT_FINAL_SELECTION"
    assert len(result["models"]) == 5
    cohort = json.loads((f.output / "train_cohort.json").read_bytes())
    assert cohort["rows"] == f.rows
    for name, spec in result["models"].items():
        ancestry = json.loads(Path(spec["ancestry_path"]).read_bytes())
        assert ancestry["kind"] == "native_assessment_ancestry_v1"
        assert ancestry["completeness"] == "COMPLETE_LOCAL_ANCESTRY"
        assert len(ancestry["fit_inputs"]) == 1
        fit = ancestry["fit_inputs"][0]
        manifest = json.loads(Path(fit["manifest_path"]).read_bytes())
        for key in ("input", "cohort", "statistics", "config", "split"):
            assert manifest[key + "_sha256"] == support.digest(Path(fit[key + "_path"]))
        assert cohort["split_sha256"] == support.digest(Path(fit["split_path"]))
        selection = json.loads(Path(spec["selection_path"]).read_bytes())
        assert selection["status"] == "NOT_FINAL_SELECTION"
        assert "owner_freeze_confirmed" not in selection
        if name == "fine":
            assert "Supervised full-finetuned" in ancestry["feature_training_kind"]
            assert ancestry["encoder_parent_relation"] == "SUPERVISED_ENCODER_TENSORS_DIFFER"
        if name == "random":
            assert "Untrained random" in ancestry["feature_training_kind"]
    assert (
        json.loads((f.output / "proposed-freeze-inputs.json").read_bytes())["status"]
        == "PROPOSED_NOT_FINAL_SELECTION"
    )
    with pytest.raises(FileExistsError):
        f.candidate.derive_inventory(f.manifest, f.output)


@pytest.mark.parametrize(
    "kind,seed",
    [
        ("native_ssl_weights_only_inference_v1", 7),
        ("native_ssl_weights_only_inference_v1", 13),
        ("native_ssl_weights_only_inference_v1", 23),
        ("native_band_ssl_weights_only_inference_v1", 7),
        ("native_band_replication_ssl_weights_only_inference_v2", 7),
        ("native_band_replication_ssl_weights_only_inference_v2", 13),
        ("native_band_replication_ssl_weights_only_inference_v2", 23),
    ],
)
def test_typed_versions_and_admitted_seeds(kind, seed):
    f = support.Fixture().add("root", kind=kind, seed=seed).finish()
    result = f.candidate.derive_inventory(f.manifest, f.output)
    assert result["models"]["root"]["kind"] == kind
    assert result["models"]["root"]["seed"] == seed


@pytest.mark.parametrize(
    "filename",
    [
        "config.json",
        "scalers.json",
        "inference.pt",
        "membership.json",
        "selected_encoder.pt",
        "run.json",
        "review.json",
    ],
)
def test_stale_endpoint_is_rejected_before_any_safe_decode(filename, monkeypatch):
    f = fixture()
    path = f.root / "root" / filename
    with path.open("ab") as stream:
        stream.write(b"tampered")
    monkeypatch.setattr(
        f.candidate, "_safe_load", lambda p: pytest.fail("Stale gate reached safe tensor decode")
    )
    with pytest.raises(ValueError, match="Stale"):
        f.candidate.derive_inventory(f.manifest, f.output)
    assert not f.output.exists()


@pytest.mark.parametrize(
    "change",
    [
        "missing_source",
        "wrong_seed",
        "wrong_v1_seed",
        "missing_parent",
        "cycle",
        "promoted_report",
        "partial_cohort",
        "duplicate_cohort",
        "reserved_site",
        "foreign_archive",
    ],
)
def test_metadata_semantic_guards_before_tensor_decode(change, monkeypatch):
    f = fixture()
    manifest = json.loads(f.manifest.read_bytes())
    if change == "missing_source":
        del manifest["bindings"][
            str(BUILDER / "src/marine_echo/evaluation/native_ancestry_inventory.py")
        ]
    elif change in {"wrong_seed", "wrong_v1_seed"}:
        manifest["endpoints"]["root"]["seed"] = 99 if change == "wrong_seed" else 13
        if change == "wrong_v1_seed":
            manifest["endpoints"]["root"]["kind"] = "native_band_ssl_weights_only_inference_v1"
    elif change == "missing_parent":
        manifest["endpoints"]["root"]["parent"] = "absent"
    elif change == "cycle":
        manifest["endpoints"]["root"]["parent"] = "root"
    f.rewrite(f.manifest, manifest)
    if change == "promoted_report":
        path = f.root / "root/run.json"
        run = json.loads(path.read_bytes())
        run["evidence_kind"] = "REAL_TRAIN_DEVELOPMENT_FIT"
        f.rewrite(path, run, rebind=True)
    elif change in {"partial_cohort", "duplicate_cohort"}:
        cohort = json.loads(f.cohort.read_bytes())
        cohort["rows"] = (
            cohort["rows"][:2] if change == "partial_cohort" else [cohort["rows"][0]] * 3
        )
        f.rewrite(f.cohort, cohort, rebind=True)
    elif change == "reserved_site":
        split = json.loads(f.split.read_bytes())
        split["sources"][0]["site"] = split["sources"][2]["site"]
        f.rewrite(f.split, split, rebind=True)
    elif change == "foreign_archive":
        path = f.root / "root/membership.json"
        member = json.loads(path.read_bytes())
        member["train_archive_sha256"][0] = "c" * 64
        f.rewrite(path, member, rebind=True)
        run_path = f.root / "root/run.json"
        run = json.loads(run_path.read_bytes())
        run["membership_sha256"] = support.digest(path)
        f.rewrite(run_path, run, rebind=True)
    monkeypatch.setattr(
        f.candidate, "_safe_load", lambda p: pytest.fail("Invalid metadata reached tensor decoding")
    )
    with pytest.raises(ValueError):
        f.candidate.derive_inventory(f.manifest, f.output)
    assert not f.output.exists()


@pytest.mark.parametrize(
    "change",
    [
        "selected_mismatch",
        "bad_kind",
        "extra_metadata",
        "nonfinite",
        "bad_scaler",
        "bad_config",
        "historical_ancestry",
    ],
)
def test_safe_embedded_artifact_guards(change):
    f = fixture()

    def transform(payload):
        if change == "selected_mismatch":
            payload["model"]["encoder.projection.weight"] += 1
        elif change == "bad_kind":
            payload["kind"] = "native_ssl_resume_v1"
        elif change == "extra_metadata":
            payload["unsafe_command"] = "never execute provenance"
        elif change == "nonfinite":
            payload["model"]["encoder.projection.weight"][0, 0] = float("nan")
        elif change == "bad_scaler":
            payload["scalers"]["channel_std"][0] = 0
        elif change == "bad_config":
            payload["config"]["latent"] = 128
        elif change == "historical_ancestry":
            payload["supervised_ancestry"] = {"historical_initial_weights": True}

    mutate_artifact(f, transform)
    with pytest.raises(ValueError):
        f.candidate.derive_inventory(f.manifest, f.output)
    assert not f.output.exists()


@pytest.mark.parametrize(
    "mode,change",
    [
        ("frozen_readout", "encoder"),
        ("frozen_readout", "head"),
        ("full_finetune", "encoder"),
        ("full_finetune", "parent"),
    ],
)
def test_frozen_and_full_parent_tensor_proofs(mode, change):
    f = support.Fixture().add("root").add("child", mode=mode, parent="root").finish()
    path = f.root / "child/inference.pt"
    artifact = torch.load(path, weights_only=True, map_location="cpu")
    if change == "parent":
        artifact["supervised_ancestry"]["ancestor_run_sha256"] = "d" * 64
    elif change == "head":
        artifact["model"]["readout.weight"] = f.data["root"]["artifact"]["model"]["readout.weight"]
    elif mode == "frozen_readout":
        artifact["model"]["encoder.projection.weight"] += 2
    else:
        artifact["model"]["encoder.projection.weight"] = f.data["root"]["encoder"][
            "projection.weight"
        ]
        selected_path = f.root / "child/selected_encoder.pt"
        selected = torch.load(selected_path, weights_only=True, map_location="cpu")
        selected["encoder"]["projection.weight"] = artifact["model"]["encoder.projection.weight"]
        torch.save(selected, selected_path)
        manifest = json.loads(f.manifest.read_bytes())
        manifest["bindings"][str(selected_path)] = support.digest(selected_path)
        f.rewrite(f.manifest, manifest)
    torch.save(artifact, path)
    manifest = json.loads(f.manifest.read_bytes())
    manifest["bindings"][str(path)] = support.digest(path)
    f.rewrite(f.manifest, manifest)
    run_path = f.root / "child/run.json"
    run = json.loads(run_path.read_bytes())
    run["inference_sha256"] = support.digest(path)
    run["selected_encoder_sha256"] = support.digest(f.root / "child/selected_encoder.pt")
    run["supervised_ancestry"] = artifact["supervised_ancestry"]
    f.rewrite(run_path, run, rebind=True)
    with pytest.raises(ValueError):
        f.candidate.derive_inventory(f.manifest, f.output)


def test_safe_codec_flags_no_rng_or_dtype_changes(monkeypatch):
    f = fixture()
    calls = []
    original = torch.load

    def checked(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "load", checked)
    state, dtype = torch.get_rng_state().clone(), torch.get_default_dtype()
    f.candidate.derive_inventory(f.manifest, f.output)
    assert len(calls) == 2
    assert all(c["weights_only"] is True and c["map_location"] == "cpu" for c in calls)
    assert torch.equal(state, torch.get_rng_state())
    assert dtype == torch.get_default_dtype()


class UnsafeMetadata:
    pass


def test_unsafe_pickle_never_uses_unrestricted_load():
    f = fixture()
    mutate_artifact(f, lambda payload: payload.update(extra=UnsafeMetadata()))
    with pytest.raises(Exception, match="[Ww]eights only|[Ww]eights_only|Unsupported global"):
        f.candidate.derive_inventory(f.manifest, f.output)
    assert not f.output.exists()


def test_v2_frozen_preserves_typed_v1_selected_parent():
    f = (
        support.Fixture()
        .add("v1", kind="native_band_ssl_weights_only_inference_v1")
        .add(
            "v2",
            kind="native_band_replication_ssl_weights_only_inference_v2",
            mode="frozen_readout",
            parent="v1",
        )
        .finish()
    )
    result = f.candidate.derive_inventory(f.manifest, f.output)
    assert result["models"]["v1"]["kind"] == "native_band_ssl_weights_only_inference_v1"
    assert result["models"]["v2"]["parent_names"] == ["v1"]
    assert support.digest(f.root / "v2/selected_encoder.pt") == support.digest(
        f.root / "v1/selected_encoder.pt"
    )


@pytest.mark.parametrize("method", ["masked_ssl", "permuted_ssl", "cf_jepa"])
def test_non_jepa_control_and_cf_labels_remain_explicit(method):
    f = support.Fixture().add("root", method=method).finish()
    result = f.candidate.derive_inventory(f.manifest, f.output)
    label = result["models"]["root"]["feature_training_kind"]
    assert label == f.candidate.LABELS[method]


def reference(f, name, kind):
    directory = f.root / name
    directory.mkdir()
    config, stats = directory / "config.json", directory / "stats.json"
    review, report = directory / "review.json", directory / "result.json"
    if kind == "external_chronos_unknown":
        support.write(
            config, {"kind": kind, "external_model_identity": "SYNTHETIC_EXTERNAL_UNKNOWN"}
        )
        support.write(stats, {"ancestry": "UNKNOWN_EXTERNAL"})
        support.write(review, {"status": "NOT_APPROVED_SYNTHETIC_FIXTURE"})
        support.write(
            report, {"evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY", "ancestry": "UNKNOWN_EXTERNAL"}
        )
    else:
        method = "lightgbm" if kind == "lightgbm15_utf8" else kind
        support.write(
            config,
            {
                "method": method,
                "history": 96,
                "seed": 7,
                "feature_schema": "native_context_mask_metadata_summary_v1"
                if method == "lightgbm"
                else "native_primary_context_source_offsets_v1",
            },
        )
        support.write(
            stats,
            {
                "fit_role": "train" if method == "lightgbm" else "none",
                "input_npz_sha256": support.digest(f.train) if method == "lightgbm" else None,
                "normalization": "NO_FITTED_NORMALIZATION",
            },
        )
        support.write(
            review,
            {
                "status": "APPROVED_PREFIT",
                "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY",
                "reviewer_session_id": "SYNTHETIC_REFERENCE_REVIEWER",
                "implementer_session_id": "SYNTHETIC_IMPLEMENTER",
                "bindings": {
                    str(f.train): support.digest(f.train),
                    str(f.dev): support.digest(f.dev),
                },
            },
        )
        models = []
        if method == "lightgbm":
            for h in (1, 3, 6):
                for q in (0.05, 0.25, 0.5, 0.75, 0.95):
                    path = directory / f"h{h}_q{q:.2f}.txt"
                    path.write_text(
                        "SYNTHETIC_CORRECTNESS_ONLY: text identity; not a fitted booster",
                        encoding="utf-8",
                    )
                    models.append({"path": path.name, "sha256": support.digest(path)})
        support.write(
            report,
            {
                "status": "COMPLETED_DEVELOPMENT_REFERENCE",
                "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY",
                "method": method,
                "history": 96,
                "seed": 7,
                "tuning": "one_fixed_development_recipe_no_test_access",
                "train_sha256": support.digest(f.train),
                "dev_sha256": support.digest(f.dev),
                "review_sha256": support.digest(review),
                "models": models,
            },
        )
    return {
        "kind": kind,
        "config_path": str(config),
        "statistics_path": str(stats),
        "report_path": str(report),
        "review_path": str(review),
    }


def test_reference_no_fit_local_boosters_and_unknown_external_are_distinct():
    f = support.Fixture().add("root")
    specs = {
        name: reference(f, name, kind)
        for name, kind in (
            ("persistence", "persistence"),
            ("seasonal", "seasonal24"),
            ("lightgbm", "lightgbm15_utf8"),
            ("chronos", "external_chronos_unknown"),
        )
    }
    f.finish(specs)
    result = f.candidate.derive_inventory(f.manifest, f.output)
    assert result["references"]["persistence"]["locally_fitted"] is False
    assert result["references"]["seasonal"]["locally_fitted"] is False
    assert result["references"]["lightgbm"]["locally_fitted"] is True
    assert len(result["references"]["lightgbm"]["boosters"]) == 15
    assert result["references"]["chronos"]["clean_local_ancestry_guarantee"] is False
    assert "ancestry_path" not in result["references"]["chronos"]
    assert not (f.output / "chronos.reference-ancestry.json").exists()
    for name in ("persistence", "seasonal", "lightgbm"):
        ancestry = json.loads((f.output / f"{name}.reference-ancestry.json").read_bytes())
        assert bool(ancestry["fit_inputs"]) == (name == "lightgbm")


def test_unbound_parent_and_reference_changes_are_denied_before_weights(monkeypatch):
    f = support.Fixture().add("root").add("child", mode="full_finetune", parent="root")
    specs = {"reference": reference(f, "reference", "persistence")}
    f.finish(specs)
    path = f.root / "child/run.json"
    run = json.loads(path.read_bytes())
    run["supervised_ancestry"]["ancestor_encoder_sha256"] = "f" * 64
    f.rewrite(path, run, rebind=True)
    monkeypatch.setattr(
        f.candidate,
        "_safe_load",
        lambda p: pytest.fail("Forged local parent reached tensor loading"),
    )
    with pytest.raises(ValueError, match="ancestor"):
        f.candidate.derive_inventory(f.manifest, f.output)


def test_production_inventory_cannot_be_run_from_builder(monkeypatch):
    f = fixture()
    manifest = json.loads(f.manifest.read_bytes())
    manifest["evidence_kind"] = "REAL_TRAIN_DEVELOPMENT_FIT"
    f.rewrite(f.manifest, manifest)
    monkeypatch.setattr(
        f.candidate, "_safe_load", lambda p: pytest.fail("Builder decoded actual artifact")
    )
    expected = "ROOT-only" if BUILDER.name == "marine-jepa-vnext-builder" else "Actual immutable native split path required"
    with pytest.raises(ValueError, match=expected):
        f.candidate.derive_inventory(f.manifest, f.output)
