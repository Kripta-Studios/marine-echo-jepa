"""SYNTHETIC_CORRECTNESS_ONLY CPU metadata and admission tests."""

import importlib.util
import sys
from pathlib import Path

import pytest


def test_native_consecutive_source_intervals_accept_clock_jitter():
    from datetime import datetime, timedelta

    m = support.metadata()
    for record in m["intervals"].values():
        stamp = datetime.fromisoformat(record["timestamp"])
        record["timestamp"] = (
            stamp + timedelta(minutes=record["source_interval_index"] % 2)
        ).isoformat()
    for row in m["rows"]:
        row["cutoff"] = m["intervals"][row["context_ids"][-1][0]]["timestamp"]
    result = prefix.partitions(m, 7)
    assert result["fit_rows"] and result["suffix_rows"]


def test_structural_missing_target_preserves_suffix_issuance_without_fake_time():
    m = support.metadata(cutoffs=(0, 888))
    row = m["rows"][-1]
    missing = row["target_ids"][2]
    record = m["intervals"].pop(missing)
    row["target_ids"][2] = -1
    row["target_observed"][2] = False
    row["target_absence_refs"] = [None, None, "native-gap"]
    m["source_gaps"] = {
        "native-gap": {
            "complete": True,
            **{k: row[k] for k in ("deployment", "site", "archive", "configuration")},
            "reason": "SOURCE_GAP",
            "first_source_interval_index": record["source_interval_index"],
            "last_source_interval_index": record["source_interval_index"],
            "source_metadata_sha256": "a" * 64,
        }
    }
    result = prefix.partitions(m, 1)
    assert result["suffix_rows"] == [(row["deployment"], row["row_id"])]
    assert result["reserved_suffix_support"][0]["target_timestamps"][2] is None
    assert result["reserved_suffix_support"][0]["target_ids"][2] == -1
    assert result["suffix_support_status"] == "NOT_ASSESSABLE"


def test_direct_supervised_kind_frozen_encoder_fresh_head_replay():
    import torch

    c = prefix.PrefixConfig(
        method="direct", mode="scratch_direct", correctness_smoke=True, updates=4, cadence=1
    )
    b = prefix.core.Config(method="direct", width=8, latent=4, blocks=1, heads=2).to_dict()
    stats = {
        "channel_mean": [0.0] * 4,
        "channel_std": [1.0] * 4,
        "target_mean": [0.0] * 3,
        "target_std": [1.0] * 3,
    }
    scratch = prefix.prepare_model(c, b, None, stats)
    selected = {
        "kind": "native_downstream_supervised_encoder_v1",
        "config": b,
        "scalers": stats,
        "encoder": prefix.core.cpu_state(scratch.encoder),
        "supervised_ancestry": {
            "mode": "direct_end_to_end",
            "ssl_only": False,
            "supervised_updates": 3000,
            "selected_supervised_step": 750,
            "ancestor_encoder_sha256": None,
            "ancestor_run_sha256": None,
        },
    }
    frozen = prefix.prepare_model(
        prefix.PrefixConfig(**{**c.to_dict(), "mode": "frozen_readout"}), b, selected, stats
    )
    assert all(
        torch.equal(v, frozen.encoder.state_dict()[k]) for k, v in selected["encoder"].items()
    )
    assert all(
        torch.equal(v, frozen.head.state_dict()[k]) for k, v in scratch.head.state_dict().items()
    )


def test_mixed_or_unknown_split_roles_fail_before_numeric_decode(memory_case, monkeypatch):
    c = memory_case
    c.documents["split.txt"]["sources"].append(
        {"deployment": "synthetic-unknown", "archive_sha256": "b" * 64, "role": "unexpected"}
    )
    c.documents["train-intervals.json"]["split_sha256"] = prefix.sha(
        support.encoded(c.documents["split.txt"])
    )
    c.seal()
    monkeypatch.setattr(prefix.np, "load", lambda *a, **k: pytest.fail("Bad split decoded"))
    with pytest.raises(ValueError, match="split"):
        prefix.admit(c.manifest_path, c.review_path, c.output)


@pytest.mark.parametrize("damage", ["missing_index", "wrong_target_index", "large_gap"])
def test_native_interval_proof_rejects_missing_wrong_indices_and_gaps(damage):
    from datetime import datetime, timedelta

    m = support.metadata()
    row = m["rows"][0]
    if damage == "missing_index":
        del m["intervals"][row["context_ids"][0][0]]["source_interval_index"]
    elif damage == "wrong_target_index":
        m["intervals"][row["target_ids"][-1]]["source_interval_index"] += 1
    else:
        record = m["intervals"][row["context_ids"][10][0]]
        record["timestamp"] = (
            datetime.fromisoformat(record["timestamp"]) + timedelta(minutes=6)
        ).isoformat()
    with pytest.raises(ValueError):
        prefix.partitions(m, 7)


BUILDER = Path(__file__).resolve().parents[2]
MAIN = BUILDER.parent / "marine-echo-jepa"
sys.path.insert(0, str(MAIN / "src"))
SPEC = importlib.util.spec_from_file_location(
    "marine_echo.training.native_prefix_transfer",
    BUILDER / "src/marine_echo/training/native_prefix_transfer.py",
)
prefix = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prefix
SPEC.loader.exec_module(prefix)
prefix.torch.set_num_threads(1)


@pytest.mark.parametrize("days,end", [(1, "2026-01-06"), (7, "2026-01-12"), (30, "2026-02-04")])
def test_frozen_boundaries_and_common_suffix(days, end):
    start, stop, suffix = prefix.boundaries("2026-01-01", days)
    assert start.isoformat() == "2026-01-05T00:00:00"
    assert stop.isoformat() == end + "T00:00:00"
    assert suffix.isoformat() == "2026-02-11T00:00:00"


import copy
import io

import numpy as np

SUPPORT_SPEC = importlib.util.spec_from_file_location(
    "native_prefix_test_support",
    BUILDER / "evidence/ssl-prefix-native-config-builder-v3/prefix_test_support.py",
)
support = importlib.util.module_from_spec(SUPPORT_SPEC)
sys.modules[SUPPORT_SPEC.name] = support
SUPPORT_SPEC.loader.exec_module(support)


@pytest.fixture
def memory_case(monkeypatch):
    return support.case(prefix, monkeypatch, BUILDER)


def test_actual_main_helpers_are_imported_read_only():
    assert Path(prefix.core.__file__).resolve().is_relative_to(MAIN)
    assert (
        Path(sys.modules["marine_echo.models.native_temporal"].__file__)
        .resolve()
        .is_relative_to(MAIN)
    )
    assert Path(prefix.__file__).resolve().is_relative_to(BUILDER)


def test_package_initializers_are_bound_before_model_or_numeric_access(memory_case, monkeypatch):
    package = MAIN / "src/marine_echo/__init__.py"
    assert package.resolve() in prefix.required_sources()
    c = memory_case
    c.review["bindings"].pop(str(package.resolve()))
    c.fs.files[c.review_path] = support.encoded(c.review)
    monkeypatch.setattr(
        prefix.np, "load", lambda *a, **k: pytest.fail("Unbound package decoded arrays")
    )
    with pytest.raises(ValueError, match="binding"):
        prefix.admit(c.manifest_path, c.review_path, c.output)


@pytest.mark.parametrize(
    "days,expected", [(1, [0, 17]), (7, [0, 17, 24, 140]), (30, [0, 17, 24, 140, 360, 695, 696])]
)
def test_metadata_holes_boundaries_and_multiple_deployments(days, expected):
    m = support.metadata(("synthetic-a", "synthetic-b"))
    result = prefix.partitions(m, days)
    assert len(result["fit_rows"]) == len(expected) * 2
    for dep in ("synthetic-a", "synthetic-b"):
        assert {int(r.rsplit("-", 1)[1]) for d, r in result["fit_rows"] if d == dep} == set(
            expected
        )
    assert {int(r.rsplit("-", 1)[1]) for d, r in result["suffix_rows"]} == {888, 912}
    assert not set(result["fit_interval_ids"]) & set(result["suffix_interval_ids"])


def test_suffix_support_same_all_prefixes_and_no_numeric_boundary_dependence():
    m = support.metadata()
    results = [prefix.partitions(m, days) for days in (1, 7, 30)]
    assert len({r["suffix_support_sha256"] for r in results}) == 1
    assert results[0]["suffix_interval_ids"] == results[2]["suffix_interval_ids"]
    assert len(results[0]["fit_rows"]) < len(results[2]["fit_rows"])


@pytest.mark.parametrize(
    "change",
    [
        "missing_id",
        "wrong_time",
        "alias",
        "configuration",
        "165",
        "geometry",
        "incomplete",
        "duplicate",
        "timezone",
    ],
)
def test_raw_interval_proofs_reject_incomplete_alias_overlap_and_configuration(change):
    m = support.metadata()
    row = m["rows"][-1]
    identity = row["context_ids"][0][0]
    if change == "missing_id":
        del m["intervals"][identity]
    elif change == "wrong_time":
        m["intervals"][identity]["timestamp"] = m["rows"][0]["cutoff"]
    elif change == "alias":
        row["context_ids"][0][0] = m["rows"][0]["context_ids"][0][0]
    elif change == "configuration":
        m["intervals"][identity]["configuration"] = "different-config"
    elif change == "165":
        m["intervals"][identity]["pings"] = 165
    elif change == "geometry":
        m["intervals"][identity]["native_bounds_m"] = [0, 200]
    elif change == "incomplete":
        m["complete"] = False
    elif change == "duplicate":
        m["rows"].append(copy.deepcopy(row))
    elif change == "timezone":
        m["intervals"][identity]["timestamp"] += "+00:00"
    with pytest.raises((ValueError, KeyError)):
        prefix.partitions(m, 1)


def test_boundary_crossing_targets_not_admitted_even_when_one_horizon_inside():
    m = support.metadata(cutoffs=(18, 19, 888))
    result = prefix.partitions(m, 1)
    assert result["fit_rows"] == [], "Exclusive prefix end excludes +6h exactly midnight."


def test_distinct_ids_same_source_timestamp_rejected():
    m = support.metadata()
    first = next(iter(m["intervals"]))
    m["intervals"]["different-row-alias"] = copy.deepcopy(m["intervals"][first])
    with pytest.raises(ValueError, match="alias"):
        prefix.partitions(m, 1)


def test_admission_all_hashes_before_decode(memory_case, monkeypatch):
    monkeypatch.setattr(
        prefix.np, "load", lambda *a, **k: pytest.fail("Numeric decode during metadata admission")
    )
    monkeypatch.setattr(
        prefix.torch, "load", lambda *a, **k: pytest.fail("Tensor decode during metadata admission")
    )
    monkeypatch.setattr(
        prefix.torch, "manual_seed", lambda *a, **k: pytest.fail("RNG before admission")
    )
    got = prefix.admit(memory_case.manifest_path, memory_case.review_path, memory_case.output)
    assert len(got.partition["fit_rows"]) == 18
    assert got.config.correctness_smoke


@pytest.mark.parametrize(
    "key",
    [
        "raw.json",
        "prefix.npz",
        "dev.npz",
        "config.json",
        "stats.json",
        "split.txt",
        "protocol.txt",
        "selection.txt",
        "ancestry.json",
        "train.npz",
        "lock.txt",
        "sources.txt",
        "supervisor.txt",
    ],
)
def test_stale_bound_identity_rejected_before_decoding(memory_case, monkeypatch, key):
    case = memory_case
    case.fs.files[case.base / key] += b"tamper"
    monkeypatch.setattr(
        prefix.np, "load", lambda *a, **k: pytest.fail("Stale input reached decode")
    )
    with pytest.raises(ValueError, match="binding"):
        prefix.admit(case.manifest_path, case.review_path, case.output)


@pytest.mark.parametrize(
    "bad",
    [
        "self",
        "coordinator",
        "status",
        "role",
        "cell",
        "empty_bindings",
        "smoke_real",
        "suffix_numeric",
    ],
)
def test_wrong_review_or_synthetic_bypass_denied(memory_case, monkeypatch, bad):
    c = memory_case
    if bad == "self":
        c.review["reviewer_session_id"] = prefix.IMPLEMENTER_SESSION_ID
    elif bad == "coordinator":
        c.review["reviewer_session_id"] = c.manifest["coordinator_session_id"]
    elif bad == "status":
        c.review["status"] = "APPROVED_PREFIT"
    elif bad == "role":
        c.review["role"] = "development"
    elif bad == "cell":
        c.review["allowed_cells"][0]["seed"] = 13
    elif bad == "empty_bindings":
        c.review["bindings"] = {}
    elif bad == "smoke_real":
        c.manifest["evidence_kind"] = "REVIEWED_PREFIX_TRANSFER"
        c.seal()
    else:
        c.manifest["suffix_numeric_path"] = "forbidden.npz"
        c.seal()
    c.fs.files[c.review_path] = support.encoded(c.review)
    monkeypatch.setattr(prefix.np, "load", lambda *a, **k: pytest.fail("Invalid review decoded"))
    with pytest.raises(ValueError):
        prefix.admit(c.manifest_path, c.review_path, c.output)


@pytest.mark.parametrize(
    "bad",
    [
        "overlap_site",
        "overlap_archive",
        "overlap_deployment",
        "unknown",
        "historical",
        "cycle",
        "wrong_role",
        "cohort_mismatch",
    ],
)
def test_recursive_ancestry_denials_before_decode(memory_case, monkeypatch, bad):
    c = memory_case
    node = c.documents["ancestry.json"]
    if bad.startswith("overlap_"):
        field = bad.removeprefix("overlap_")
        member = node["inputs"][0]["members"][0]
        member[field] = next(iter(c.registry["sources"].values()))[field]
        c.documents["train-cohort.json"]["members"] = node["inputs"][0]["members"]
    elif bad == "unknown":
        node["complete"] = False
    elif bad == "historical":
        node["historical_initial_weights"] = True
    elif bad == "cycle":
        node["parents"] = ["ancestry.json"]
    elif bad == "wrong_role":
        node["role"] = "prefix"
    else:
        c.documents["train-cohort.json"]["members"] = [{"deployment": "unknown"}]
    c.seal()
    monkeypatch.setattr(prefix.np, "load", lambda *a, **k: pytest.fail("Bad ancestry decoded"))
    with pytest.raises(ValueError):
        prefix.admit(c.manifest_path, c.review_path, c.output)


def test_numeric_access_review_cannot_be_inherited_from_zero_shot(memory_case):
    c = memory_case
    c.numeric["allowed_uses"] = ["zero_shot"]
    c.fs.files[c.base / "numeric.json"] = support.encoded(c.numeric)
    c.review["bindings"][str(c.base / "numeric.json")] = prefix.sha(
        c.fs.files[c.base / "numeric.json"]
    )
    c.fs.files[c.review_path] = support.encoded(c.review)
    with pytest.raises(ValueError, match="scope"):
        prefix.admit(c.manifest_path, c.review_path, c.output)


def test_actual_decode_no_future_and_exact_native_support(memory_case, monkeypatch):
    c = memory_case
    admitted = prefix.admit(c.manifest_path, c.review_path, c.output)
    admitted.manifest["_base"] = str(c.base)
    getitem = np.lib.npyio.NpzFile.__getitem__
    visited = []

    def read(z, key):
        visited.append(key)
        assert key != "future"
        return getitem(z, key)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", read)
    data = prefix._decode(admitted, "prefix_npz", "prefix")
    assert "future" not in visited
    np.testing.assert_allclose(data["query"][..., 4] * 250, 230)
    assert data["target_observed"].sum(0).tolist() == [18] * 3


@pytest.mark.parametrize(
    "bad",
    [
        "row",
        "role",
        "target_date",
        "primary_missing",
        "query",
        "nonfinite_label",
        "geometry",
        "nonboolean",
    ],
)
def test_numeric_identity_geometry_support_rejected(memory_case, bad):
    c = memory_case
    if bad == "row":
        c.prefix_arrays["row_id"][0] = "wrong-row"
    elif bad == "role":
        c.prefix_arrays["corpus_role"] = np.asarray("final_test")
    elif bad == "target_date":
        c.prefix_arrays["target_dates"][0, 0] = "2026-09-01"
    elif bad == "primary_missing":
        c.prefix_arrays["context_observed"][0, 0, 0] = False
    elif bad == "query":
        c.prefix_arrays["query"][0, 0, 9] = 7
    elif bad == "nonfinite_label":
        c.prefix_arrays["targets"][0, 0] = np.nan
    elif bad == "geometry":
        c.prefix_arrays["metadata"][..., 4] = 0.8
    else:
        c.prefix_arrays["target_observed"] = c.prefix_arrays["target_observed"].astype(int)
    c.seal()
    admitted = prefix.admit(c.manifest_path, c.review_path, c.output)
    admitted.manifest["_base"] = str(c.base)
    with pytest.raises(ValueError):
        prefix._decode(admitted, "prefix_npz", "prefix")


def test_protected_collision_before_numeric_or_model_decode(memory_case, monkeypatch):
    c = memory_case
    c.fs.dirs.add(c.output)
    monkeypatch.setattr(prefix.np, "load", lambda *a, **k: pytest.fail("Collision decoded"))
    monkeypatch.setattr(
        prefix.torch, "load", lambda *a, **k: pytest.fail("Collision decoded weights")
    )
    with pytest.raises(FileExistsError):
        prefix.fit(c.manifest_path, c.review_path, c.output)


def test_fresh_heads_identical_modes_and_seed_specific_sampler():
    import torch

    config = prefix.PrefixConfig(
        method="direct", mode="scratch_direct", correctness_smoke=True, updates=4, cadence=1
    )
    backbone = prefix.core.Config(method="direct", width=8, latent=4, blocks=1, heads=2).to_dict()
    stats = {
        "channel_mean": [0.0] * 4,
        "channel_std": [1.0] * 4,
        "target_mean": [0.0] * 3,
        "target_std": [1.0] * 3,
    }
    scratch = prefix.prepare_model(config, backbone, None, stats)
    selected = {
        "kind": "native_downstream_supervised_encoder_v1",
        "supervised_ancestry": support.direct_ancestry(),
        "config": backbone,
        "scalers": stats,
        "encoder": prefix.core.cpu_state(scratch.encoder),
    }
    frozen_cfg = prefix.PrefixConfig(**{**config.to_dict(), "mode": "frozen_readout"})
    frozen = prefix.prepare_model(frozen_cfg, backbone, selected, stats)
    assert all(
        torch.equal(v, frozen.head.state_dict()[k]) for k, v in scratch.head.state_dict().items()
    )
    assert all(
        torch.equal(v, frozen.encoder.state_dict()[k]) for k, v in selected["encoder"].items()
    )
    for step in range(4):
        np.testing.assert_array_equal(
            prefix.sample_indices(20, config, step), prefix.sample_indices(20, frozen_cfg, step)
        )
    control13 = prefix.PrefixConfig(**{**config.to_dict(), "seed": 13})
    assert not np.array_equal(
        prefix.sample_indices(80, config, 0), prefix.sample_indices(80, control13, 0)
    )


def test_gradients_observed_labels_only_frozen_buffers_and_scratch():
    import torch

    torch.set_num_threads(1)
    c = prefix.PrefixConfig(
        method="direct", mode="scratch_direct", correctness_smoke=True, updates=4, cadence=1
    )
    backbone = prefix.core.Config(method="direct", width=8, latent=4, blocks=1, heads=2).to_dict()
    stats = {
        "channel_mean": [-50.0] * 4,
        "channel_std": [5.0] * 4,
        "target_mean": [-50.0] * 3,
        "target_std": [5.0] * 3,
    }
    data = support.arrays([], {}, "development", count=3)
    data["target_observed"][0, 1] = False
    data["targets"][0, 1] = np.nan
    scratch = prefix.prepare_model(c, backbone, None, stats)
    b = prefix._batch(data, np.arange(3), prefix._scalers(stats), "cpu", labels=True)
    p = scratch.forecast(b["x"], b["observed"], b["metadata"], b["query"], frozen=False)
    p.retain_grad()
    prefix.core.pinball(p, b["y"], b["y_observed"]).backward()
    assert p.grad[0, 1].abs().sum() == 0
    assert any(t.grad is not None and t.grad.abs().sum() > 0 for t in scratch.encoder.parameters())
    selected = {
        "kind": "native_downstream_supervised_encoder_v1",
        "supervised_ancestry": support.direct_ancestry(),
        "config": backbone,
        "scalers": stats,
        "encoder": prefix.core.cpu_state(scratch.encoder),
    }
    frozen = prefix.prepare_model(
        prefix.PrefixConfig(**{**c.to_dict(), "mode": "frozen_readout"}), backbone, selected, stats
    )
    before = prefix.core.cpu_state(frozen.encoder)
    p = frozen.forecast(b["x"], b["observed"], b["metadata"], b["query"])
    prefix.core.pinball(p, b["y"], b["y_observed"]).backward()
    assert all(t.grad is None and not t.requires_grad for t in frozen.encoder.parameters())
    assert all(torch.equal(v, frozen.encoder.state_dict()[k]) for k, v in before.items())


def test_safe_inference_codec_replay_no_rng_reset_targets_or_future():
    import torch

    c = prefix.PrefixConfig(
        method="direct", mode="scratch_direct", correctness_smoke=True, updates=4, cadence=1
    )
    backbone = prefix.core.Config(method="direct", width=8, latent=4, blocks=1, heads=2).to_dict()
    stats = {
        "channel_mean": [-50.0] * 4,
        "channel_std": [5.0] * 4,
        "target_mean": [-50.0] * 3,
        "target_std": [5.0] * 3,
    }
    model = prefix.prepare_model(c, backbone, None, stats).eval()
    artifact = {
        "kind": "native_prefix_transfer_inference_v1",
        "config": c.to_dict(),
        "backbone_config": backbone,
        "architecture": prefix.ARCHITECTURES[c.family],
        "model": prefix.core.cpu_state(model),
        "scalers": stats,
        "selected_step": 1,
        "zero_shot": False,
        "identities": {"synthetic": "a" * 64},
    }
    before = torch.get_rng_state().clone()
    inference = prefix.PrefixPredictor(io.BytesIO(prefix.encode_checkpoint(artifact)))
    assert torch.equal(before, torch.get_rng_state())
    data = support.arrays([], {}, "development", count=3)
    expected = prefix.predict(model, data, prefix._scalers(stats), c)
    data["metadata"][..., 4] = 230 / 250
    data["query"][..., 4] = 230 / 250
    expected = prefix.predict(model, data, prefix._scalers(stats), c)
    got = inference.forecast(data["x"], data["context_observed"], data["metadata"], data["query"])
    np.testing.assert_array_equal(got, expected)
    assert not inference.model.training
    assert all(not p.requires_grad and p.grad is None for p in inference.model.parameters())
    data["targets"][:] = 1e9
    np.testing.assert_array_equal(
        got,
        inference.forecast(data["x"], data["context_observed"], data["metadata"], data["query"]),
    )


@pytest.mark.parametrize(
    "bad", ["kind", "architecture", "missing_tensor", "wrong_shape", "scalers", "selected_step"]
)
def test_safe_artifact_rejects_incompatible_kinds_states_and_statistics(bad):
    import torch

    c = prefix.PrefixConfig(
        method="direct", mode="scratch_direct", correctness_smoke=True, updates=4, cadence=1
    )
    backbone = prefix.core.Config(method="direct", width=8, latent=4, blocks=1, heads=2).to_dict()
    model = prefix.prepare_model(c, backbone, None, {})
    artifact = {
        "kind": "native_prefix_transfer_inference_v1",
        "config": c.to_dict(),
        "backbone_config": backbone,
        "architecture": prefix.ARCHITECTURES[c.family],
        "model": prefix.core.cpu_state(model),
        "scalers": {
            "channel_mean": [0.0] * 4,
            "channel_std": [1.0] * 4,
            "target_mean": [0.0] * 3,
            "target_std": [1.0] * 3,
        },
        "selected_step": 1,
        "zero_shot": False,
        "identities": {"synthetic": "a" * 64},
    }
    if bad == "kind":
        artifact["kind"] = "native_ssl_weights_only_inference_v1"
    elif bad == "architecture":
        artifact["architecture"] = "wrong"
    elif bad == "missing_tensor":
        artifact["model"].pop(next(iter(artifact["model"])))
    elif bad == "wrong_shape":
        key = next(iter(artifact["model"]))
        artifact["model"][key] = torch.zeros(3)
    elif bad == "scalers":
        artifact["scalers"]["target_std"] = [0.0] * 3
    else:
        artifact["selected_step"] = 0
    with pytest.raises(ValueError):
        prefix.PrefixPredictor(io.BytesIO(prefix.encode_checkpoint(artifact)))


def test_real_recipe_four_dev_opportunities_and_no_ssl_full_finetune():
    config = prefix.PrefixConfig()
    config.validate("REVIEWED_PREFIX_TRANSFER")
    assert list(range(config.cadence, config.updates + 1, config.cadence)) == [
        500,
        1000,
        1500,
        2000,
    ]
    with pytest.raises(ValueError):
        prefix.PrefixConfig(updates=3000).validate("REVIEWED_PREFIX_TRANSFER")
    with pytest.raises(ValueError):
        prefix.PrefixConfig(mode="full_finetune").validate("REVIEWED_PREFIX_TRANSFER")


def test_selection_cannot_name_target_site_even_when_hash_admitted(memory_case):
    c = memory_case
    c.documents["selection.txt"] = {"role": "prefix", "method": "target_site_selection"}
    c.seal()
    with pytest.raises(ValueError, match="selection"):
        prefix.admit(c.manifest_path, c.review_path, c.output)


def test_different_split_membership_not_only_a_frozen_filename(memory_case):
    c = memory_case
    c.documents["split.txt"] = {
        "sources": [{"deployment": "different-site", "archive_sha256": "wrong", "role": "test"}]
    }
    c.seal()
    with pytest.raises(ValueError, match="split"):
        prefix.admit(c.manifest_path, c.review_path, c.output)


@pytest.mark.parametrize(
    "key", ["selected.pt", "parent-inference.pt", "membership.json", "parent-run.json"]
)
def test_stale_selected_parent_denied_before_arrays(monkeypatch, key):
    c = support.frozen_case(prefix, monkeypatch, BUILDER)
    c.fs.files[c.base / key] += b"tamper"
    monkeypatch.setattr(
        prefix.np, "load", lambda *a, **k: pytest.fail("Stale selected ancestry decoded arrays")
    )
    with pytest.raises(ValueError):
        prefix.admit(c.manifest_path, c.review_path, c.output)


@pytest.mark.parametrize("bad", ["kind", "config", "scalers", "tensor", "bindings"])
def test_safe_selected_mismatch_denied_before_numeric_load(monkeypatch, bad):
    c = support.frozen_case(prefix, monkeypatch, BUILDER)
    artifact = copy.deepcopy(c.selected)
    if bad == "kind":
        artifact["kind"] = "native_ssl_resume_v1"
    elif bad == "config":
        artifact["config"]["method"] = "shared_ssl"
    elif bad == "scalers":
        artifact["scalers"]["channel_mean"][0] = -999
    elif bad == "tensor":
        artifact["encoder"].pop(next(iter(artifact["encoder"])))
    else:
        artifact["bindings"] = {}
    raw = prefix.encode_checkpoint(artifact)
    c.fs.files[c.base / "selected.pt"] = raw
    for v in c.documents["ancestry.json"]["artifacts"]:
        if v["path"] == "selected.pt":
            v["sha256"] = prefix.sha(raw)
    c.seal()
    monkeypatch.setattr(
        prefix.np,
        "load",
        lambda *a, **k: pytest.fail("Unsafe selected tensor decoded numerical arrays"),
    )
    monkeypatch.setattr(
        prefix.torch.optim,
        "AdamW",
        lambda *a, **k: pytest.fail("Unsafe encoder initialized optimizer"),
    )
    with pytest.raises(ValueError):
        prefix.fit(c.manifest_path, c.review_path, c.output)


def test_deeper_recursive_parent_overlap_cannot_hide_behind_clean_immediate_ancestor(memory_case):
    c = memory_case
    parent = copy.deepcopy(c.documents["ancestry.json"])
    parent["inputs"][0]["members"][0]["site"] = "synthetic-site-site"
    c.documents["parent-cohort.json"] = {
        "role": "train",
        "complete": True,
        "members": parent["inputs"][0]["members"],
    }
    parent["inputs"][0]["cohort"] = "parent-cohort.json"
    c.documents["bad-parent.json"] = parent
    c.documents["ancestry.json"]["parents"] = ["bad-parent.json"]
    c.seal()
    with pytest.raises(ValueError, match="overlaps"):
        prefix.admit(c.manifest_path, c.review_path, c.output)


def test_postprefix_numeric_rows_rejected_without_boundary_or_mask_adjustments(memory_case):
    c = memory_case
    row = next(r for r in c.registry["rows"] if r["row_id"].endswith("-888"))
    c.prefix_arrays["row_id"][0] = row["row_id"]
    c.prefix_arrays["target_observed"][0] = False
    c.seal()
    admitted = prefix.admit(c.manifest_path, c.review_path, c.output)
    admitted.manifest["_base"] = str(c.base)
    with pytest.raises(ValueError, match="row/cohort"):
        prefix._decode(admitted, "prefix_npz", "prefix")


def test_independent_daily_hierarchy_unequal_counts():
    # Independently expected score: (mean horizon means at site A + site B) /2.
    predictions, targets, masks, dates, deployments = [], [], [], [], []
    for dep, counts in (("A", (18, 24, 30)), ("B", (18, 18, 18))):
        for h, count in enumerate(counts):
            for day, value in (("2026-01-01", h + 1), ("2026-01-02", (h + 1) * 3)):
                for _ in range(count):
                    predictions.append(np.zeros((3, 5)))
                    targets.append([value] * 3)
                    masks.append([j == h for j in range(3)])
                    dates.append([day] * 3)
                    deployments.append(dep)
    result = prefix.native_scores(
        np.asarray(predictions),
        np.asarray(targets),
        np.asarray(masks),
        np.asarray(dates),
        np.asarray(deployments),
        minimum_daily_rows=18,
    )
    # Positive error against zero: average quantile=.5, average day multiplier=2,
    # hence horizon scores1,2,3 and equal horizon/site mean2, despite unequal rows.
    assert result["primary_pinball_db"] == pytest.approx(2)
    assert result["aggregation"] == "equal_target_source_date_then_horizon_then_deployment"


def test_caller_uses_actual_main_source_hashes_and_no_scientific_self_review(memory_case):
    c = memory_case
    admission = prefix.admit(c.manifest_path, c.review_path, c.output)
    for p in prefix.required_sources():
        assert admission.identities[str(p)] == prefix.sha(p.read_bytes())
    assert c.review["reviewer_session_id"] != prefix.IMPLEMENTER_SESSION_ID
    assert c.review["evidence_kind"] == prefix.EVIDENCE


def test_missing_root_declared_suffix_masks_cannot_prove_fixed_support():
    m = support.metadata()
    for r in m["rows"]:
        r.pop("target_observed", None)
    with pytest.raises(ValueError, match="support"):
        prefix.partitions(m, 1)


def test_actual_source_centres_need_not_be_clock_hour_boundaries():
    from datetime import datetime, timedelta

    m = support.metadata()
    for interval in m["intervals"].values():
        interval["timestamp"] = (
            datetime.fromisoformat(interval["timestamp"]) + timedelta(minutes=30)
        ).isoformat()
    for row in m["rows"]:
        row["cutoff"] = (datetime.fromisoformat(row["cutoff"]) + timedelta(minutes=30)).isoformat()
    result = prefix.partitions(m, 1)
    assert len(result["fit_rows"]) == 2
    assert result["boundaries"]["synthetic-site"][2] == "2026-01-05T00:00:00"


@pytest.mark.parametrize(
    "family,method",
    [
        ("core", "direct"),
        ("band", "direct"),
        ("core", "shared_ssl"),
        ("core", "permuted_ssl"),
        ("core", "masked_ssl"),
        ("core", "random_frozen"),
    ],
)
def test_typed_parent_admission_before_tensor_array_or_rng(monkeypatch, family, method):
    c = support.frozen_case(prefix, monkeypatch, BUILDER, family=family, method=method)
    monkeypatch.setattr(
        prefix.np, "load", lambda *a, **k: pytest.fail("Metadata admission decoded numeric data")
    )
    monkeypatch.setattr(
        prefix.torch, "load", lambda *a, **k: pytest.fail("Metadata admission decoded tensors")
    )
    monkeypatch.setattr(
        prefix.torch,
        "manual_seed",
        lambda *a, **k: pytest.fail("Metadata admission initialized RNG"),
    )
    admission = prefix.admit(c.manifest_path, c.review_path, c.output)
    assert admission.config.method == method
    assert admission.identities[str(c.base / "parent-inference.pt")] == prefix.sha(
        c.fs.files[c.base / "parent-inference.pt"]
    )


@pytest.mark.parametrize(
    "bad",
    [
        "status",
        "self",
        "config",
        "review_hash",
        "source",
        "membership",
        "sampling",
        "lineage",
        "parent",
        "missing_config",
        "missing_review",
        "missing_inputs",
        "original_protocol",
    ],
)
def test_direct_typed_metadata_forgery_denied_before_decode(monkeypatch, bad):
    c = support.frozen_case(prefix, monkeypatch, BUILDER)
    review, run = c.documents["parent-review.json"], c.documents["parent-run.json"]
    if bad == "status":
        review["status"] = "APPROVED_PREFIT"
    elif bad == "self":
        review["reviewer_session_id"] = prefix.IMPLEMENTER_SESSION_ID
    elif bad == "config":
        c.documents["parent-config.json"]["method"] = "shared_ssl"
    elif bad == "review_hash":
        run["review_sha256"] = "f" * 64
    elif bad == "source":
        p = str(Path(prefix.core.__file__).resolve())
        run["bindings"].pop(p)
        review["bindings"].pop(p, None)
    elif bad in ("membership", "sampling"):
        membership = prefix._json(c.fs.files[c.base / "membership.json"])
        if bad == "membership":
            membership["train_deployments"][0] = "synthetic-site"
        else:
            membership["sequence"][0]["indices"] = [999]
            membership["sequence_sha256"] = prefix.sha(
                prefix.json.dumps(
                    membership["sequence"], sort_keys=True, separators=(",", ":")
                ).encode()
            )
        raw = support.encoded(membership)
        c.fs.files[c.base / "membership.json"] = raw
        run["membership_sha256"] = prefix.sha(raw)
        for item in c.documents["ancestry.json"]["artifacts"]:
            if item["path"] == "membership.json":
                item["sha256"] = prefix.sha(raw)
    elif bad == "lineage":
        run["supervised_ancestry"]["ssl_only"] = True
    elif bad == "parent":
        run["core_config"] = {**c.backbone, "method": "shared_ssl"}
    elif bad == "missing_config":
        c.manifest.pop("parent_config")
    elif bad == "missing_review":
        c.manifest.pop("parent_review")
    elif bad == "missing_inputs":
        c.manifest.pop("parent_inputs")
    else:
        p = str(c.base / "downstream-protocol.txt")
        run["bindings"].pop(p)
        review["bindings"].pop(p, None)
    c.seal()
    monkeypatch.setattr(
        prefix.np, "load", lambda *a, **k: pytest.fail("Forged parent decoded arrays")
    )
    monkeypatch.setattr(
        prefix.torch, "load", lambda *a, **k: pytest.fail("Forged parent decoded tensors")
    )
    with pytest.raises((ValueError, KeyError)):
        prefix.admit(c.manifest_path, c.review_path, c.output)


@pytest.mark.parametrize(
    "reason",
    [
        "SOURCE_GAP",
        "MISSING_SOURCE_INTERVAL",
        "CONFIGURATION_BOUNDARY",
        "PING_QUARANTINE",
        "PARTIAL_SOURCE_INTERVAL",
        "DUPLICATE_ROW",
        "MISSING_CHANNEL",
    ],
)
def test_native_structural_absence_reasons_no_fabricated_source_stamp(reason):
    m = support.metadata(cutoffs=(0, 888))
    support.absent_target(m, m["rows"][0], 2, reason)
    support.absent_target(m, m["rows"][1], 1, reason)
    results = [prefix.partitions(m, d) for d in (1, 7, 30)]
    assert all(len(r["fit_rows"]) == len(r["suffix_rows"]) == 1 for r in results)
    assert len({r["suffix_support_sha256"] for r in results}) == 1
    assert results[0]["reserved_suffix_support"][0]["target_timestamps"][1] is None
    assert -1 not in results[0]["fit_interval_ids"]
    assert results[0]["suffix_support_status"] == "NOT_ASSESSABLE"


@pytest.mark.parametrize(
    "bad",
    [
        "observed",
        "no_gap",
        "wrong_index",
        "unknown_reason",
        "ping",
        "config",
        "hash",
        "false_sentinel",
    ],
)
def test_missing_identity_cannot_hide_observed_label_or_unexplained_boundary(bad):
    m = support.metadata(cutoffs=(0, 888))
    row = m["rows"][0]
    ref = support.absent_target(m, row, 2)
    gap = m["source_gaps"][ref]
    if bad == "observed":
        row["target_observed"][2] = True
    elif bad == "no_gap":
        m["source_gaps"] = {}
    elif bad == "wrong_index":
        gap["first_source_interval_index"] += 20
    elif bad == "unknown_reason":
        gap["reason"] = "not_a_native_gap"
    elif bad == "ping":
        gap["reason"] = "PING_QUARANTINE"
        gap["quarantined_pings"] = [150]
    elif bad == "config":
        gap["reason"] = "CONFIGURATION_BOUNDARY"
        gap["next_configuration"] = row["configuration"]
    elif bad == "hash":
        gap["source_metadata_sha256"] = "unknown"
    else:
        row["target_ids"][2] = -2
    with pytest.raises(ValueError):
        prefix.partitions(m, 1)


def test_masked_known_out_of_prefix_target_never_moves_boundaries():
    m = support.metadata(cutoffs=(18, 888))
    m["rows"][0]["target_observed"][2] = False
    assert prefix.partitions(m, 1)["fit_rows"] == []


@pytest.mark.parametrize("seed", [13, 23])
def test_band_replication_remains_unassigned(seed):
    with pytest.raises(ValueError, match="Band seed7"):
        prefix.PrefixConfig(family="band", seed=seed).validate("REVIEWED_PREFIX_TRANSFER")
    prefix.PrefixConfig(seed=seed).validate("REVIEWED_PREFIX_TRANSFER")


@pytest.mark.parametrize("bad", ["batch_rows", "step", "batch_size", "batch_hash"])
def test_actual_direct_sampler_receipt_identity_is_verified_before_decode(monkeypatch, bad):
    c = support.frozen_case(prefix, monkeypatch, BUILDER)
    membership = prefix._json(c.fs.files[c.base / "membership.json"])
    entry = membership["sequence"][0]
    if bad == "batch_rows":
        entry["row_ids"][0] = "unknown-row"
    elif bad == "step":
        entry["step"] = 1000
    elif bad == "batch_size":
        entry["indices"] = entry["indices"][:1]
    else:
        entry["sha256"] = "f" * 64
    membership["sequence_sha256"] = prefix.sha(
        prefix.json.dumps(membership["sequence"], sort_keys=True, separators=(",", ":")).encode()
    )
    raw = support.encoded(membership)
    c.fs.files[c.base / "membership.json"] = raw
    c.documents["parent-run.json"]["membership_sha256"] = prefix.sha(raw)
    for item in c.documents["ancestry.json"]["artifacts"]:
        if item["path"] == "membership.json":
            item["sha256"] = prefix.sha(raw)
    c.seal()
    monkeypatch.setattr(
        prefix.np, "load", lambda *a, **k: pytest.fail("Bad sampling receipt decoded arrays")
    )
    with pytest.raises(ValueError, match="sampling"):
        prefix.admit(c.manifest_path, c.review_path, c.output)


def test_actual_split_schema_final_test_metadata_admitted_without_numeric_access(
    memory_case, monkeypatch
):
    c = memory_case
    split = c.documents["split.txt"]
    split["schema_version"] = "native_acoustic_ssl_v1"
    for source in split["sources"]:
        if source["role"] == "test":
            source["role"] = "final_test"
    c.documents["train-intervals.json"]["split_sha256"] = prefix.sha(support.encoded(split))
    c.seal()
    monkeypatch.setattr(
        prefix.np, "load", lambda *a, **k: pytest.fail("Split metadata parsed numeric arrays")
    )
    assert prefix.admit(c.manifest_path, c.review_path, c.output).partition["fit_rows"]


@pytest.mark.parametrize(
    "bad", ["schema", "alias", "mixed", "duplicate", "archive_roles", "train_final"]
)
def test_actual_split_conflicting_or_fitted_final_roles_denied(memory_case, monkeypatch, bad):
    c = memory_case
    s = c.documents["split.txt"]
    s["schema_version"] = "native_acoustic_ssl_v1"
    for source in s["sources"]:
        if source["role"] == "test":
            source["role"] = "final_test"
    if bad == "schema":
        s["schema_version"] = "different_version"
    elif bad == "alias":
        s["sources"][0]["role"] = "test"
    elif bad == "mixed":
        s["sources"].append({"deployment": "alias", "archive_sha256": "alias", "role": "test"})
    elif bad == "duplicate":
        s["sources"].append(copy.deepcopy(s["sources"][0]))
    elif bad == "archive_roles":
        s["sources"][-1]["archive_sha256"] = s["sources"][0]["archive_sha256"]
    else:
        s["sources"][-1]["role"] = "final_test"
    c.documents["train-intervals.json"]["split_sha256"] = prefix.sha(support.encoded(s))
    c.seal()
    monkeypatch.setattr(
        prefix.np, "load", lambda *a, **k: pytest.fail("Conflicting split decoded arrays")
    )
    with pytest.raises(ValueError):
        prefix.admit(c.manifest_path, c.review_path, c.output)


def test_native_configuration_map_preserves_150_and_180_in_one_deployment():
    raw, _ = support.native_configuration_metadata()
    result = prefix.partitions(raw, 7)
    assert len(raw["sources"]) == 1
    assert {
        r["configuration"]
        for r in raw["rows"]
        if (r["deployment"], r["row_id"]) in result["fit_rows"]
    } == {"native-150", "native-180"}
    assert len(result["suffix_rows"]) == 2
    assert result["boundaries"]["synthetic-site"][2:] == [
        "2026-01-05T00:00:00",
        "2026-01-12T00:00:00",
        "2026-02-11T00:00:00",
    ]


def test_positive_nominal_absence_is_not_an_observed_interval_identity():
    raw, _ = support.native_configuration_metadata(cutoffs=(21, 140, 888))
    row = raw["rows"][0]
    assert row["target_ids"][1] > 0 and row["target_observed"][1] is False
    result = prefix.partitions(raw, 7)
    assert (row["deployment"], row["row_id"]) in result["fit_rows"]
    assert all(isinstance(identity, str) for identity in result["fit_interval_ids"])
    assert row["target_ids"][1] not in result["fit_interval_ids"]


@pytest.mark.parametrize("days", [1, 7, 30])
def test_native_configurations_keep_common_suffix_and_actual_native_bounds(days):
    raw, _ = support.native_configuration_metadata()
    result = prefix.partitions(raw, days)
    support_rows = result["reserved_suffix_support"]
    assert len(support_rows) == 2
    assert {r["deployment"] for r in support_rows} == {"synthetic-site"}
    assert {r["issued_configuration"] for r in support_rows} == {"native-180"}
    assert all(all(r["target_available"]) for r in support_rows)
    assert raw["configuration_map"]["sources"]["synthetic-site"]["configurations"]["native-180"][
        "channel_bounds_m"
    ] == [[0, 230], [0, 225], [0, 230], [0, 230]]
    assert result["boundaries"]["synthetic-site"][-1] == "2026-02-11T00:00:00"


@pytest.mark.parametrize(
    "damage",
    [
        "configuration",
        "geometry",
        "processing",
        "pings",
        "segment",
        "lifetime",
        "site",
        "archive",
        "source_hash",
        "quarantine",
        "row",
        "context_mask",
        "unknown_mode",
        "future_clock",
        "future_after_break",
    ],
)
def test_native_configuration_proof_rejects_conflicting_actual_metadata(damage):
    raw, _ = support.native_configuration_metadata(cutoffs=(21, 140, 888))
    row = raw["rows"][1]
    record = raw["intervals"][row["context_ids"][10][1]]
    entry = raw["configuration_map"]["sources"][row["deployment"]]
    if damage == "configuration":
        record["configuration"] = "native-150"
    elif damage == "geometry":
        record["native_bounds_m"] = [0, 200]
    elif damage == "processing":
        record["processing_id_or_unknown"] = "different-native-processing"
    elif damage == "pings":
        record["pings"] = 150
    elif damage == "segment":
        entry["segments"][2]["first_source_interval_index"] += 1
    elif damage == "lifetime":
        entry["end"] = "2026-02-28T00:00:00"
    elif damage == "site":
        entry["site"] = "fake-site"
    elif damage == "archive":
        record["archive"] = "fake-archive"
    elif damage == "source_hash":
        entry["source_metadata_sha256"] = "unbound"
    elif damage == "quarantine":
        record["pings"] = 165
    elif damage == "row":
        row["configuration"] = "native-150"
    elif damage == "context_mask":
        row["context_observed"][10][1] = False
    elif damage == "unknown_mode":
        raw["configuration_mode"] = "single_deployment_guess"
    elif damage == "future_clock":
        from datetime import datetime, timedelta

        future = raw["intervals"][row["future_chain_ids"][1]]
        future["timestamp"] = (
            datetime.fromisoformat(future["timestamp"]) + timedelta(minutes=10)
        ).isoformat()
    else:
        raw["rows"][0]["future_chain_ids"][-1] = row["future_chain_ids"][0]
    with pytest.raises((ValueError, KeyError)):
        prefix.partitions(raw, 7)


@pytest.mark.parametrize(
    "damage",
    [
        "observed_label",
        "metadata",
        "ping",
        "processing",
        "mask",
        "wrong_request",
        "raw_id",
        "no_receipt",
        "wrong_break",
        "future_available",
    ],
)
def test_nominal_unavailable_ids_cannot_hide_observed_labels_or_source_flags(damage):
    raw, _ = support.native_configuration_metadata(cutoffs=(21, 140, 888))
    row = raw["rows"][0]
    if damage == "observed_label":
        row["target_observed"][1] = row["target_slot_observed"][1][0] = True
    elif damage == "metadata":
        row["target_slot_metadata"][1][0][0] = 38000 / 455000
    elif damage == "ping":
        row["target_slot_ping_counts"][1][0] = 165
    elif damage == "processing":
        row["target_slot_processing_id_or_unknown"][1][0] = "actual-transition-processing"
    elif damage == "mask":
        row["target_slot_observed"][1][1] = True
    elif damage == "wrong_request":
        row["target_ids"][1] += 1
    elif damage == "raw_id":
        row["target_ids"][1] = row["context_ids"][-1][0]
    elif damage == "no_receipt":
        raw["source_gaps"] = {}
    elif damage == "wrong_break":
        raw["source_gaps"][row["future_absence_ref"]]["break_source_interval_index"] += 1
    else:
        row["target_slot_metadata"][2] = copy.deepcopy(raw["rows"][1]["target_slot_metadata"][2])
        row["target_slot_processing_id_or_unknown"][2] = ["native-process-150"] * 4
        row["target_slot_ping_counts"][2] = [150] * 4
    with pytest.raises(ValueError):
        prefix.partitions(raw, 7)


def test_minus_one_and_nominal_requests_preserve_same_support_without_fake_timestamp():
    raw, _ = support.native_configuration_metadata(cutoffs=(21, 140, 888))
    initial = prefix.partitions(raw, 7)
    for row in raw["rows"]:
        row["target_ids"] = [-1 if type(v) is int else v for v in row["target_ids"]]
    explicit = prefix.partitions(raw, 7)
    assert initial["fit_rows"] == explicit["fit_rows"]
    assert initial["fit_interval_ids"] == explicit["fit_interval_ids"]
    assert initial["reserved_suffix_support"] == explicit["reserved_suffix_support"]


def test_available_masked_target_retains_actual_timestamp_not_structural_absence():
    raw, _ = support.native_configuration_metadata(cutoffs=(0, 888))
    row = raw["rows"][0]
    row["target_observed"][0] = row["target_slot_observed"][0][0] = False
    record = raw["intervals"][row["target_ids"][0]]
    record["observed"], record["qc"] = False, "INVALID_OR_SENTINEL"
    result = prefix.partitions(raw, 1)
    assert row["target_ids"][0] in result["fit_interval_ids"]
    assert row["future_chain_ids"][0] == row["target_ids"][0]


def test_legacy_single_configuration_requires_explicit_synthetic_compatibility():
    raw = support.metadata()
    prefix.partitions(raw, 1)
    raw.pop("configuration_mode")
    with pytest.raises(ValueError, match="explicit synthetic"):
        prefix.partitions(raw, 1)


def test_native_configuration_clock_jitter_keeps_actual_source_centres():
    from datetime import datetime, timedelta

    raw, _ = support.native_configuration_metadata()
    for record in raw["intervals"].values():
        record["timestamp"] = (
            datetime.fromisoformat(record["timestamp"])
            + timedelta(minutes=record["source_interval_index"] % 2)
        ).isoformat()
    for row in raw["rows"]:
        row["cutoff"] = raw["intervals"][row["context_ids"][-1][0]]["timestamp"]
    result = prefix.partitions(raw, 7)
    assert result["fit_rows"] and result["suffix_rows"]
    assert result["boundaries"]["synthetic-site"][-1] == "2026-02-11T00:00:00"


def test_reader_actual_nominal_ids_and_constant_context_configuration_semantics():
    # SYNTHETIC_CORRECTNESS_ONLY; inspect only generated metadata/IDs/masks.
    from marine_echo.data.native_ssl_corpus import NativeSlot, issue_windows

    slots = []
    for index in range(330):
        pings = 150 if index < 150 else 165 if index <= 160 else 180
        slots.append(
            NativeSlot(
                index,
                np.datetime64("2026-01-01T00:00") + np.timedelta64(index, "h"),
                np.zeros(4, np.float32),
                np.full(4, pings != 165, bool),
                np.asarray(
                    [[0, 230], [0, 225 if pings == 180 else 230], [0, 230], [0, 230]], float
                ),
                (f"processing-{pings}",) * 4,
                (pings,) * 4,
                "SYNTHETIC_CORRECTNESS_ONLY-one-deployment",
                "synthetic-archive",
                ("PARTIAL_SOURCE_INTERVAL" if pings == 165 else "OBSERVED_CENSORING_UNKNOWN",) * 4,
            )
        )
    issued = issue_windows(slots, history=96)
    before = int(np.flatnonzero(issued["cutoff"] == 149)[0])
    assert issued["future_ids"][before].min() == 150
    assert not issued["future_observed"][before].any()
    assert not issued["future_metadata"][before].any()
    assert np.all(issued["future_ping_counts"][before] == 0)
    assert np.all(issued["future_processing_id_or_unknown"][before] == "UNKNOWN")
    assert {int(p[0, 0]) for p in issued["ping_counts"]} == {150, 180}
    assert len(set(issued["deployment"].tolist())) == 1
    assert not any(150 <= int(c) < 256 for c in issued["cutoff"])
    assert np.allclose(issued["metadata"][:, 0, 4] * 250, 230)


def test_unobserved_secondary_165_values_are_quarantined_without_dropping_primary():
    raw, _ = support.native_configuration_metadata(cutoffs=(0, 888))
    row = raw["rows"][0]
    record = raw["intervals"][row["context_ids"][10][2]]
    record.update(observed=False, pings=165, qc="PARTIAL_SOURCE_INTERVAL")
    row["context_observed"][10][2] = False
    result = prefix.partitions(raw, 1)
    assert (row["deployment"], row["row_id"]) in result["fit_rows"]
    row["context_observed"][10][2] = True
    with pytest.raises(ValueError):
        prefix.partitions(raw, 1)


def test_missing_secondary_channel_has_no_invented_identity_or_timestamp():
    raw, _ = support.native_configuration_metadata(cutoffs=(0, 888))
    row = raw["rows"][0]
    identity = row["context_ids"][10][2]
    record = raw["intervals"].pop(identity)
    row["context_ids"][10][2], row["context_observed"][10][2] = None, False
    row["context_absence_refs"] = [[None] * 4 for _ in range(96)]
    row["context_absence_refs"][10][2] = "actual-missing-secondary"
    entry = raw["configuration_map"]["sources"][row["deployment"]]
    raw.setdefault("source_gaps", {})["actual-missing-secondary"] = {
        "complete": True,
        **{k: row[k] for k in ("deployment", "site", "archive", "configuration")},
        "channel": 2,
        "reason": "MISSING_CHANNEL",
        "first_source_interval_index": record["source_interval_index"],
        "last_source_interval_index": record["source_interval_index"],
        "source_metadata_sha256": entry["source_metadata_sha256"],
    }
    result = prefix.partitions(raw, 1)
    assert identity not in result["fit_interval_ids"]
    assert (row["deployment"], row["row_id"]) in result["fit_rows"]


def test_bound_clock_claim_cannot_mask_a_known_observed_native_interval():
    from datetime import datetime, timedelta

    raw, _ = support.native_configuration_metadata(cutoffs=(0, 17, 888))
    row = raw["rows"][0]
    cutoff = raw["intervals"][row["context_ids"][-1][0]]["source_interval_index"]
    ref = "fake-clock-break"
    row["future_chain_ids"] = [None] * 9
    row["future_absence_ref"] = ref
    row["target_ids"] = [cutoff + h for h in (1, 3, 6)]
    row["target_observed"] = [False] * 3
    row["target_absence_refs"] = [ref] * 3
    row["target_slot_metadata"] = np.zeros((3, 4, 10)).tolist()
    row["target_slot_observed"] = np.zeros((3, 4), bool).tolist()
    row["target_slot_processing_id_or_unknown"] = [["UNKNOWN"] * 4 for _ in range(3)]
    row["target_slot_ping_counts"] = [[0] * 4 for _ in range(3)]
    entry = raw["configuration_map"]["sources"][row["deployment"]]
    raw.setdefault("source_gaps", {})[ref] = {
        "complete": True,
        **{k: row[k] for k in ("deployment", "site", "archive", "configuration")},
        "first_source_interval_index": cutoff + 1,
        "last_source_interval_index": cutoff + 9,
        "break_source_interval_index": cutoff + 1,
        "reason": "SOURCE_GAP",
        "previous_timestamp": row["cutoff"],
        "break_timestamp": (
            datetime.fromisoformat(row["cutoff"]) + timedelta(minutes=68)
        ).isoformat(),
        "source_metadata_sha256": entry["source_metadata_sha256"],
    }
    with pytest.raises(ValueError, match="actual raw timestamp"):
        prefix.partitions(raw, 7)
