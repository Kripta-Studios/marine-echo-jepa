"""SYNTHETIC_CORRECTNESS_ONLY: native metadata derivation, never public data."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

BUILDER = Path(__file__).resolve().parents[2]
MAIN = BUILDER.parent / "marine-echo-jepa"
sys.path.insert(0, str(MAIN / "src"))
SPEC = importlib.util.spec_from_file_location(
    "native_transfer_candidate", BUILDER / "src/marine_echo/data/native_transfer_corpus.py"
)
candidate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = candidate
SPEC.loader.exec_module(candidate)

from marine_echo.data.native_ssl_corpus import NativeSlot, issue_windows


def slots(n=1200, deployment="SYNTHETIC_A", changes=None):
    changes = changes or {}
    result = []
    for i in range(n):
        pings = 150 if i < 500 else 180
        observed = np.ones(4, bool)
        qc = ["OBSERVED_CENSORING_UNKNOWN"] * 4
        counts, processing = [pings] * 4, ["SYNTHETIC_PROCESS"] * 4
        if i in changes:
            channel, reason = changes[i]
            observed[channel], qc[channel] = False, reason
            if reason == "MISSING_CHANNEL":
                counts[channel], processing[channel] = 0, "UNKNOWN"
            if reason == "PARTIAL_SOURCE_INTERVAL":
                counts[channel] = 165
        result.append(
            NativeSlot(
                i,
                np.datetime64("2020-01-01T00:00", "us") + np.timedelta64(i, "h"),
                np.asarray([-60, -61, -62, -63.0]) + i / 1000,
                observed,
                np.tile([0.0, 230.0], (4, 1)),
                tuple(processing),
                tuple(counts),
                deployment,
                "a" * 64,
                tuple(qc),
            )
        )
    return result


def source(deployment="SYNTHETIC_A"):
    return {
        "deployment": deployment,
        "site": "SYNTHETIC_SITE_" + deployment,
        "archive": "a" * 64,
        "start": "2020-01-01T00:00:00",
        "end": "2020-03-01T00:00:00",
        "native_bounds_m": [0, 230],
    }


def derive(items):
    # Small genuine reader windows at prespecified clocks, never value-selected.
    cuts = [
        *range(95, 125),
        195,
        199,
        200,
        201,
        249,
        299,
        495,
        499,
        600,
        700,
        800,
        *range(990, 1026),
    ]
    pieces = []
    for end, slot in enumerate(items):
        if slot.interval not in cuts or end < 95:
            continue
        part = issue_windows(items[end - 95 : end + 11])
        if len(part["x"]):
            pieces.append({k: v[:1] for k, v in part.items()})
    arrays = {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
    return arrays, candidate.derive_metadata(items, arrays, source(), "SYNTHETIC_METADATA.json")


def test_original_arrays_preserved_and_fixed_boundaries():
    items = slots()
    arrays, bundle = derive(items)
    original = issue_windows(items[:106])
    for key, value in original.items():
        np.testing.assert_array_equal(arrays[key][:1], value[:1])
    results = candidate.validate_partitions(bundle["registry"])
    for days, result in results.items():
        bounds = result["boundaries"]["SYNTHETIC_A"]
        assert bounds[2] == "2020-01-05T00:00:00"
        expected_end = f"2020-01-{5 + days:02d}T00:00:00" if days < 30 else "2020-02-04T00:00:00"
        assert bounds[3] == expected_end
        assert bounds[4] == "2020-02-11T00:00:00"
        assert not set(result["fit_interval_ids"]) & set(result["suffix_interval_ids"])
    assert results[1]["suffix_rows"] == results[30]["suffix_rows"]
    assert len(results[1]["fit_rows"]) < len(results[7]["fit_rows"]) < len(results[30]["fit_rows"])


@pytest.mark.parametrize(
    "reason", ["MISSING_CHANNEL", "DUPLICATE_ROW", "PARTIAL_SOURCE_INTERVAL", "INVALID_OR_SENTINEL"]
)
def test_secondary_missing_values_do_not_invent_ids_or_configurations(reason):
    arrays, bundle = derive(slots(changes={200: (1, reason)}))
    registry = bundle["registry"]
    row = next(r for r in registry["rows"] if r["cutoff"] == "2020-01-09T08:00:00.000000")
    if reason in {"MISSING_CHANNEL", "DUPLICATE_ROW"}:
        assert row["context_ids"][-1][1] is None
        assert row["context_absence_refs"][-1][1] in registry["source_gaps"]
    else:
        identity = row["context_ids"][-1][1]
        assert registry["intervals"][identity]["observed"] is False
    assert not arrays["observed"][list(arrays["cutoff"]).index(200), -1, 1]
    assert len(bundle["configuration_receipt"]["configurations"]) == 2
    candidate.validate_partitions(registry)


def test_positive_nominal_missing_target_and_165_quarantine():
    arrays, bundle = derive(slots(changes={250: (0, "PARTIAL_SOURCE_INTERVAL")}))
    registry = bundle["registry"]
    row = next(r for r in registry["rows"] if r["target_requested_indices"][0] == 250)
    assert row["target_ids"][0] == 250
    assert row["target_observed"][0] is False
    gap = registry["source_gaps"][row["target_absence_refs"][0]]
    assert gap["reason"] == "PING_QUARANTINE"
    assert gap["quarantined_pings"] == [165]
    assert not arrays["y_observed"][list(arrays["cutoff"]).index(249)].any()
    candidate.validate_partitions(registry)


def test_configuration_boundary_retains_same_deployment():
    _, bundle = derive(slots())
    configs = bundle["configuration_receipt"]["configurations"]
    assert {tuple(c["ping_counts"]) for c in configs.values()} == {(150,) * 4, (180,) * 4}
    gaps = bundle["registry"]["source_gaps"].values()
    assert any(g["reason"] == "CONFIGURATION_BOUNDARY" for g in gaps)
    assert set(bundle["registry"]["sources"]) == {"SYNTHETIC_A"}


def test_gap_not_squeezed_and_no_timestamp_invention():
    items = slots()
    del items[300]
    _, bundle = derive(items)
    registry = bundle["registry"]
    assert all(r["source_interval_index"] != 300 for r in registry["intervals"].values())
    assert any(g["reason"] == "MISSING_SOURCE_INTERVAL" for g in registry["source_gaps"].values())
    candidate.validate_partitions(registry)


def test_duplicate_interval_and_wrong_geometry_rejected():
    with pytest.raises(ValueError):
        issue_windows(slots(106) + slots(1))
    items = slots()
    items[0].bounds[0, 1] = 200
    with pytest.raises(ValueError, match="230"):
        derive(items)


def test_target_values_never_change_metadata_or_cohorts():
    items = slots()
    _, first = derive(items)
    for item in items:
        item.values[:] = 9999
    _, second = derive(items)
    assert first == second


def test_insufficient_suffix_is_not_assessable():
    items = slots(993)
    _, bundle = derive(items)
    assert (
        candidate.validate_partitions(bundle["registry"])[1]["suffix_support_status"]
        == "NOT_ASSESSABLE"
    )


@pytest.mark.parametrize("minutes", [5, -5])
def test_real_native_55_65_minute_jitter_retained(minutes):
    items = slots()
    items[200] = replace(items[200], timestamp=items[200].timestamp + np.timedelta64(minutes, "m"))
    _, bundle = derive(items)
    record = next(
        r
        for r in bundle["registry"]["intervals"].values()
        if r["source_interval_index"] == 200 and r["channel"] == 0
    )
    assert record["timestamp"] == str(items[200].timestamp)
    candidate.validate_partitions(bundle["registry"])


def test_clock_gap_has_actual_timestamps_and_no_hour_repair():
    items = slots()
    items[200] = replace(items[200], timestamp=items[200].timestamp + np.timedelta64(6, "m"))
    _, bundle = derive(items)
    gaps = [g for g in bundle["registry"]["source_gaps"].values() if g["reason"] == "SOURCE_GAP"]
    assert gaps
    assert gaps[0]["break_timestamp"] == str(items[200].timestamp)
    candidate.validate_partitions(bundle["registry"])


def test_observed_identity_cannot_be_replaced_by_missing_positive_request():
    _, bundle = derive(slots())
    registry = bundle["registry"]
    row = next(r for r in registry["rows"] if all(r["target_observed"]))
    row["target_ids"][0] = row["target_requested_indices"][0]
    with pytest.raises(ValueError):
        candidate.validate_partitions(registry)


def test_unknown_utc_and_all_metadata_acoustic_values_excluded():
    _, bundle = derive(slots())
    assert bundle["configuration_receipt"]["source_clock"] == "SOURCE_TIMEZONE_UNKNOWN"
    assert bundle["configuration_receipt"]["acoustic_values"] == "NOT_INCLUDED"
    assert "values" not in bundle["registry"]["intervals"].values()


@pytest.mark.parametrize(
    "blocker",
    [
        "gpu-owner.lock",
        "RUNNING_CUDA",
        "RUNNING_CPU_FIT",
        "record_reconciliation",
        "ledger_reconciliation",
        "all.pending",
        ".prefix-pending",
        ".band-pending",
        ".assessment-pending",
        ".reconciliation-pending",
        "malformed_ledger",
        "missing_ledger",
    ],
)
def test_root_only_fixture_rejects_actual_ownership_before_mutation(blocker, monkeypatch):
    import json
    import uuid

    spec = importlib.util.spec_from_file_location(
        "synthetic_owned_transfer_fixture",
        BUILDER / "evidence/ssl-native-transfer-corpus-builder-v1/root_supervised_fixture.py",
    )
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    private = (
        BUILDER
        / "evidence/ssl-native-transfer-corpus-builder-v1"
        / ("SYNTHETIC_CORRECTNESS_ONLY-ownership-" + str(uuid.uuid4()))
    )
    root = private / "marine-echo-jepa"
    (root / "orchestration").mkdir(parents=True)
    ledger = root / "orchestration/native_ssl_run_ledger_v1.json"
    document = {"runs": []}
    if blocker in {"RUNNING_CUDA", "RUNNING_CPU_FIT"}:
        document["runs"] = [{"status": blocker}]
    elif blocker == "record_reconciliation":
        document["runs"] = [{"status": "FAILED", "requires_reconciliation": True}]
    elif blocker == "ledger_reconciliation":
        document["requires_reconciliation"] = True
    if blocker != "missing_ledger":
        with ledger.open("x", encoding="utf-8") as stream:
            json.dump({"runs": "invalid"} if blocker == "malformed_ledger" else document, stream)
    if blocker == "gpu-owner.lock":
        lock = root / "evidence/ssl-builder-v1/gpu-owner.lock"
        lock.parent.mkdir(parents=True)
        lock.open("x").close()
    elif blocker.endswith("pending"):
        journal = (
            ledger.parent / blocker if blocker == "all.pending" else ledger.with_suffix(blocker)
        )
        journal.open("x").close()
    monkeypatch.setattr(fixture, "__file__", str(root / "evidence/fixture/fixture.py"))

    def forbidden(*args, **kwargs):
        pytest.fail("Ownership admission must precede spawning or output mkdir.")

    monkeypatch.setattr(fixture.subprocess, "Popen", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    with pytest.raises(RuntimeError, match="owner|ledger|journal|reconciliation"):
        fixture.main()


@pytest.mark.parametrize("helper", ["write_source_products", "_cohort"])
def test_production_transport_calls_match_verified_codec_interfaces(helper):
    """Validate unexecuted public worker wiring against real private-tested codecs."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(candidate.materialize))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == helper
    ]
    assert calls
    signature = inspect.signature(getattr(candidate, helper))
    for call in calls:
        signature.bind(
            *[object() for _ in call.args],
            **{keyword.arg: object() for keyword in call.keywords},
        )
