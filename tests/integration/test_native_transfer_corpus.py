"""SYNTHETIC_CORRECTNESS_ONLY private codecs and metadata gates; no public data."""

from __future__ import annotations

import csv
import importlib.util
import io
import json
import sys
import uuid
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

BUILDER = Path(__file__).resolve().parents[2]
MAIN = BUILDER.parent / "marine-echo-jepa"
sys.path.insert(0, str(MAIN / "src"))
SPEC = importlib.util.spec_from_file_location(
    "native_transfer_integration_candidate",
    BUILDER / "src/marine_echo/data/native_transfer_corpus.py",
)
candidate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = candidate
SPEC.loader.exec_module(candidate)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(candidate.encode_json(value))
    return str(path)


@pytest.fixture
def private_dir():
    path = (
        BUILDER
        / "evidence/ssl-native-transfer-corpus-builder-v1"
        / ("SYNTHETIC_CORRECTNESS_ONLY-" + str(uuid.uuid4()))
    )
    path.mkdir()
    return path


def tiny_archive(path, *, n=240, deployment="SYNTHETIC_A", offset=0, invalid_primary=None):
    with path.open("xb") as transport, zipfile.ZipFile(transport, "w") as archive:
        for frequency in ("038", "125", "200", "455"):
            output = io.StringIO()
            names = [
                "Interval",
                "Date_M",
                "Time_M",
                "Sv_mean",
                "Layer",
                "Layer_depth_min",
                "Layer_depth_max",
                "Ping_S",
                "Ping_E",
                "Process_ID",
            ]
            writer = csv.DictWriter(output, fieldnames=names)
            writer.writeheader()
            for i in range(n):
                if i == 145 and frequency == "125":
                    continue
                clock = datetime(2020, 1, 1) + timedelta(hours=i, minutes=offset)  # noqa: DTZ001 -- synthetic source timezone unknown
                writer.writerow(
                    dict(
                        zip(
                            names,
                            [
                                i,
                                clock.strftime("%Y%m%d"),
                                clock.strftime("%H:%M:%S"),
                                -999
                                if frequency == "038" and i == invalid_primary
                                else -60 - i / 1000,
                                1,
                                0,
                                230,
                                1,
                                150 if i < 180 else 180,
                                "SYNTHETIC_PROCESS",
                            ],
                            strict=True,
                        )
                    )
                )
            archive.writestr(f"AEON2_1_{frequency}_2020_01_60minFullDepth.csv", output.getvalue())
    return {
        "file_id": 1,
        "deployment": deployment,
        "site": "SYNTHETIC_SITE",
        "path": str(path),
        "archive_sha256": candidate.reader.sha256(path),
        "role": "final_test",
        "start_date": "20200101",
        "end_date_inclusive": "20200229",
        "complete_ping_counts": [150, 180],
        "quarantined_ping_counts": [165],
    }


@pytest.fixture
def policy_fixture(private_dir):
    """Metadata-only fake signed identities; never invokes production decode."""
    root = private_dir
    protocol = root / "docs/protocol.md"
    protocol.parent.mkdir()
    protocol.write_text("SYNTHETIC_CORRECTNESS_ONLY frozen protocol\n", encoding="utf-8")
    lock = root / "dependencies.txt"
    lock.write_text("SYNTHETIC_CORRECTNESS_ONLY no installed changes\n", encoding="utf-8")
    source = tiny_archive(root / "SYNTHETIC_CORRECTNESS_ONLY.zip")
    split_path = root / "configs/split.json"
    save(
        split_path,
        {
            "schema_version": "native_acoustic_ssl_v1",
            "support_history": 96,
            "protocol_path": "docs/protocol.md",
            "protocol_sha256": candidate.reader.sha256(protocol),
            "sources": [source],
        },
    )
    artifacts = {}
    for name in ("model", "readout", "scalers", "config", "completion", "ancestry"):
        path = root / ("SYNTHETIC_" + name + ".json")
        save(path, {"status": "COMPLETED", "evidence_kind": "REAL_TRAIN_DEVELOPMENT_FIT"})
        artifacts[name] = str(path)
    selection_path = root / "selection.json"
    save(
        selection_path,
        {
            "kind": "native_selection_freeze_v1",
            "selection_role": "development",
            "frozen_before_numeric_access": True,
            "final_test_numerical_model_selection": False,
            "owner_session_id": "SYNTHETIC_ROOT",
            "owner_freeze_confirmed": True,
            "frozen_at": "2026-09-29T00:00:00+00:00",
            "methods": {"SYNTHETIC_METHOD": {"status": "COMPLETED", **artifacts}},
        },
    )
    selection_review = root / "selection-review.json"
    save(
        selection_review,
        {
            "status": "APPROVED_SELECTION_FREEZE",
            "scope": "selection_freeze",
            "reviewer_session_id": "SYNTHETIC_DISTINCT_FREEZE_REVIEWER",
            "implementer_session_id": candidate.IMPLEMENTER,
            "reviewed_at": "2026-09-29T01:00:00+00:00",
            "bindings": {
                str(p): candidate.reader.sha256(p)
                for p in [selection_path, *map(Path, artifacts.values())]
            },
        },
    )
    output, receipt = root / "NEW_OUTPUT", root / "NEW_RECEIPT.json"
    identity_path = root / "output-identity.json"
    save(
        identity_path,
        {
            "kind": "native_transfer_output_identity_v1",
            "output_path": str(output),
            "receipt_path": str(receipt),
            "role": "final_test",
            "fit_authorized": False,
        },
    )
    identity = {
        "deployment": source["deployment"],
        "site": source["site"],
        "archive": source["archive_sha256"],
        "start": "2020-01-01T00:00:00",
        "end": "2020-03-01T00:00:00",
        "native_bounds_m": [0, 230],
    }
    manifest_path = root / "manifest.json"
    manifest = {
        "kind": "native_transfer_materialization_manifest_v1",
        "role": "final_test",
        "evidence_kind": candidate.EVIDENCE,
        "fit_authorized": False,
        "history": 96,
        "device": "cpu",
        "implementer_session_id": candidate.IMPLEMENTER,
        "coordinator_session_id": "SYNTHETIC_ROOT",
        "output_path": str(output),
        "receipt_path": str(receipt),
        "split": str(split_path),
        "protocol": str(protocol),
        "prefix_protocol": str(MAIN / "docs/adr/0021-native-prefix-transfer-assessment.md"),
        "dependency_lock": str(lock),
        "selection": str(selection_path),
        "supervisor": str(MAIN / "tools/native_reference_supervisor.py"),
        "resource_policy": candidate.RESOURCE_POLICY,
        "selection_review": str(selection_review),
        "output_identity": str(identity_path),
        "source_identities": {source["deployment"]: identity},
    }
    save(manifest_path, manifest)
    bindings = {
        str(p): candidate.reader.sha256(p)
        for p in [
            *candidate.required_sources(),
            manifest_path,
            split_path,
            protocol,
            lock,
            selection_path,
            selection_review,
            identity_path,
            Path(source["path"]),
            *map(Path, artifacts.values()),
            MAIN / "tools/native_reference_supervisor.py",
            MAIN / "docs/adr/0021-native-prefix-transfer-assessment.md",
        ]
    }
    review_path = root / "numeric-review.json"
    review = {
        "status": "APPROVED_NATIVE_NUMERIC_ACCESS",
        "scope": "native_transfer_materialization",
        "allowed_roles": ["final_test"],
        "evidence_kind": candidate.EVIDENCE,
        "fit_authorized": False,
        "reviewer_session_id": "SYNTHETIC_DISTINCT_NUMERIC_REVIEWER",
        "implementer_session_id": candidate.IMPLEMENTER,
        "reviewed_at": "2026-09-30T00:00:00+00:00",
        "output_path": str(output),
        "receipt_path": str(receipt),
        "frozen_selection_sha256": candidate.reader.sha256(selection_path),
        "source_identities": manifest["source_identities"],
        "bindings": bindings,
    }
    review["resource_policy"] = candidate.RESOURCE_POLICY
    save(review_path, review)
    return manifest_path, review_path, output, receipt


def rewrite(path, value):
    # Private synthetic fixture changes deliberately invalidate prior bindings.
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream)


def test_complete_metadata_gate_does_not_decode(policy_fixture, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Numeric parsing before gate")

    monkeypatch.setattr(zipfile, "ZipFile", forbidden)
    admitted = candidate.admit(*policy_fixture)
    assert admitted.manifest["role"] == "final_test"
    assert not policy_fixture[2].exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "APPROVED_PREFIT"),
        ("allowed_roles", ["development"]),
        ("reviewer_session_id", candidate.IMPLEMENTER),
        ("reviewer_session_id", "SYNTHETIC_ROOT"),
        ("evidence_kind", "SYNTHETIC_CORRECTNESS_ONLY"),
        ("fit_authorized", True),
        ("scope", "fitting"),
        ("frozen_selection_sha256", "0" * 64),
        ("reviewed_at", "2026-09-28T00:00:00+00:00"),
        ("reviewed_at", "2099-01-01T00:00:00+00:00"),
    ],
)
def test_wrong_review_denied_before_archive(policy_fixture, monkeypatch, field, value):
    review = json.loads(policy_fixture[1].read_text(encoding="utf-8"))
    review[field] = value
    rewrite(policy_fixture[1], review)
    monkeypatch.setattr(zipfile, "ZipFile", lambda *a, **k: pytest.fail("Decoded rejected source"))
    with pytest.raises(ValueError):
        candidate.materialize(*policy_fixture)
    assert not policy_fixture[2].exists()
    assert not policy_fixture[3].exists()


@pytest.mark.parametrize("choice", ["source", "cli", "prefix", "selection", "archive", "lock"])
def test_missing_required_bindings(policy_fixture, choice):
    review = json.loads(policy_fixture[1].read_text(encoding="utf-8"))
    manifest = json.loads(policy_fixture[0].read_text(encoding="utf-8"))
    match = {
        "source": str(Path(candidate.reader.__file__).resolve()),
        "cli": str(BUILDER / "tools/materialize_native_transfer_corpus.py"),
        "prefix": str(MAIN / "src/marine_echo/training/native_prefix_transfer.py"),
        "selection": manifest["selection"],
        "archive": next(k for k in review["bindings"] if k.endswith(".zip")),
        "lock": manifest["dependency_lock"],
    }[choice]
    del review["bindings"][match]
    rewrite(policy_fixture[1], review)
    with pytest.raises(ValueError):
        candidate.admit(*policy_fixture)
    assert not policy_fixture[2].exists()


def test_stale_archive_and_protected_output(policy_fixture):
    archive = next(
        Path(p)
        for p in json.loads(policy_fixture[1].read_text(encoding="utf-8"))["bindings"]
        if p.endswith(".zip")
    )
    with archive.open("ab") as stream:
        stream.write(b"SYNTHETIC_TAMPER")
    with pytest.raises(ValueError, match="Stale"):
        candidate.admit(*policy_fixture)


def test_synthetic_completed_weights_cannot_authorize_real_access(policy_fixture, monkeypatch):
    """All hashes updated, but a smoke endpoint is still semantically inadmissible."""
    manifest = json.loads(policy_fixture[0].read_bytes())
    selection = json.loads(Path(manifest["selection"]).read_bytes())
    completion = Path(selection["methods"]["SYNTHETIC_METHOD"]["completion"])
    rewrite(completion, {"status": "COMPLETED", "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY"})
    freeze_review_path = Path(manifest["selection_review"])
    freeze_review = json.loads(freeze_review_path.read_bytes())
    freeze_review["bindings"][str(completion)] = candidate.reader.sha256(completion)
    rewrite(freeze_review_path, freeze_review)
    review = json.loads(policy_fixture[1].read_bytes())
    for path in (completion, freeze_review_path):
        review["bindings"][str(path)] = candidate.reader.sha256(path)
    rewrite(policy_fixture[1], review)
    monkeypatch.setattr(
        zipfile, "ZipFile", lambda *a, **k: pytest.fail("Decoded source for synthetic endpoint")
    )
    with pytest.raises(ValueError, match="completed.*REAL|REAL.*completed"):
        candidate.admit(*policy_fixture)
    assert not policy_fixture[2].exists()
    policy_fixture[2].mkdir()
    with pytest.raises(FileExistsError):
        candidate.admit(*policy_fixture)


def refresh_bindings(fixture):
    manifest = json.loads(fixture[0].read_bytes())
    freeze_path = Path(manifest["selection_review"])
    freeze = json.loads(freeze_path.read_bytes())
    freeze["bindings"] = {p: candidate.reader.sha256(Path(p)) for p in freeze["bindings"]}
    rewrite(freeze_path, freeze)
    review = json.loads(fixture[1].read_bytes())
    review["bindings"] = {p: candidate.reader.sha256(Path(p)) for p in review["bindings"]}
    review["frozen_selection_sha256"] = review["bindings"][manifest["selection"]]
    rewrite(fixture[1], review)


def test_original_split_single_complete_ping_count_without_quarantine_field(policy_fixture):
    manifest = json.loads(policy_fixture[0].read_bytes())
    split_path = Path(manifest["split"])
    split = json.loads(split_path.read_bytes())
    split["sources"][0]["complete_ping_counts"] = [150]
    del split["sources"][0]["quarantined_ping_counts"]
    rewrite(split_path, split)
    refresh_bindings(policy_fixture)
    assert candidate.admit(*policy_fixture).split["selected_sources"][0][
        "complete_ping_counts"
    ] == [150]


def test_current_numeric_review_cannot_change_after_gate(policy_fixture):
    admission = candidate.admit(*policy_fixture)
    with policy_fixture[1].open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="approval changed"):
        candidate._check_bindings(admission)


def test_fixed_reference_completion_is_typed_and_frozen(policy_fixture):
    manifest = json.loads(policy_fixture[0].read_bytes())
    selection_path = Path(manifest["selection"])
    selection = json.loads(selection_path.read_bytes())
    entry = selection["methods"]["SYNTHETIC_METHOD"]
    entry["artifact_family"] = "native_reference"
    rewrite(
        Path(entry["completion"]),
        {
            "status": "COMPLETED_DEVELOPMENT_REFERENCE",
            "method": "persistence",
            "history": 96,
            "models": [],
            "tuning": "one_fixed_development_recipe_no_test_access",
        },
    )
    rewrite(selection_path, selection)
    refresh_bindings(policy_fixture)
    candidate.admit(*policy_fixture)
    entry["artifact_family"] = "native_neural"
    rewrite(selection_path, selection)
    refresh_bindings(policy_fixture)
    with pytest.raises(ValueError, match="completed"):
        candidate.admit(*policy_fixture)


def test_actual_reader_source_metadata_and_two_source_equivalence(private_dir):
    for index in (1, 2):
        source = tiny_archive(
            private_dir / f"SYNTHETIC_SOURCE_{index}.zip", n=240, deployment=f"SYNTHETIC_{index}"
        )
        items = candidate.reader.read_source(source)
        arrays = candidate.reader.issue_windows(items)
        clock = candidate.source_clock_metadata(source)
        identity = {
            "deployment": source["deployment"],
            "site": source["site"],
            "archive": source["archive_sha256"],
            "start": "2020-01-01T00:00:00",
            "end": "2020-03-01T00:00:00",
            "native_bounds_m": [0, 230],
        }
        directory = private_dir / f"SYNTHETIC_PRODUCT_{index}"
        bundle = candidate.derive_metadata(
            items, arrays, identity, directory / "source-metadata.json", clock_records=clock
        )
        assert "Sv_mean" not in json.dumps(bundle["configuration_receipt"])
        assert clock[145].get(1) is None
        products = candidate.write_source_products(
            directory, arrays, bundle, source, "a" * 64, evidence_kind="SYNTHETIC_CORRECTNESS_ONLY"
        )
        with np.load(directory / "zero-shot.npz", allow_pickle=False) as decoded:
            for key, original in arrays.items():
                np.testing.assert_array_equal(decoded[key], original)
                assert decoded[key].dtype == original.dtype
                assert decoded[key].tobytes(order="C") == original.tobytes(order="C")
            assert decoded["corpus_role"].item() == "final_test"
            assert (decoded["metadata"][:, 0, 4] * 250 == 230).all()
        metadata_bytes = (directory / "source-metadata.json").read_bytes()
        assert (
            candidate._hash(metadata_bytes)
            == bundle["configuration_map"]["sources"][source["deployment"]][
                "source_metadata_sha256"
            ]
        )
        for days in (1, 7, 30):
            with np.load(directory / f"prefix-{days}d.npz", allow_pickle=False) as prefix:
                assert "future" not in prefix.files
                assert prefix["evidence_kind"].item() == "SYNTHETIC_CORRECTNESS_ONLY"
                assert np.array_equal(prefix["targets"], prefix["y"])
                assert prefix["metadata"].shape[1:] == (4, 10)
        cohort = json.loads((directory / "zero-shot-cohort.json").read_bytes())
        assert cohort["rows"] == list(
            map(list, zip(arrays["deployment"].tolist(), arrays["row_id"].tolist(), strict=True))
        )
        assert cohort["identities"][0]["source_ids"] == [source["archive_sha256"]]
        assert len(products) > 12
        with pytest.raises(FileExistsError):
            candidate.write_source_products(
                directory,
                arrays,
                bundle,
                source,
                "a" * 64,
                evidence_kind="SYNTHETIC_CORRECTNESS_ONLY",
            )


def test_actual_masked_sentinel_keeps_real_target_identity(private_dir):
    source = tiny_archive(private_dir / "SYNTHETIC_SENTINEL.zip", n=150, invalid_primary=140)
    items = candidate.reader.read_source(source)
    arrays = candidate.reader.issue_windows(items)
    identity = {
        "deployment": source["deployment"],
        "site": source["site"],
        "archive": source["archive_sha256"],
        "start": "2020-01-01T00:00:00",
        "end": "2020-03-01T00:00:00",
        "native_bounds_m": [0, 230],
    }
    bundle = candidate.derive_metadata(
        items,
        arrays,
        identity,
        private_dir / "SYNTHETIC_METADATA.json",
        clock_records=candidate.source_clock_metadata(source),
    )
    row = next(r for r in bundle["registry"]["rows"] if r["target_requested_indices"][0] == 140)
    raw = bundle["registry"]["intervals"][row["target_ids"][0]]
    assert raw["observed"] is False
    assert raw["qc"] == "INVALID_OR_SENTINEL"
    assert raw["timestamp"] == str(items[140].timestamp)
    assert row["target_absence_refs"][0] is None
    position = list(arrays["cutoff"]).index(139)
    assert arrays["y"][position, 0] == 0
    assert not arrays["y_observed"][position, 0]
    candidate.validate_partitions(bundle["registry"])
