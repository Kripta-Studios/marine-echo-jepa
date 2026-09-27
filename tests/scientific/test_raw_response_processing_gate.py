"""Full raw processing must fail closed on review and resumable lineage."""

from pathlib import Path

import pytest

from tools.v2_process_raw_response import (
    ensure_no_orphans,
    registered_source_path,
    validate_existing_entry,
    validate_index_identity,
    validate_review,
)


def test_wrong_study_or_data_kind_cannot_resume() -> None:
    bound = {"contract": "a" * 64}
    base = {
        "data_kind": "REAL", "study_id": "raw_response_development_v1",
        "processing_run_id": "raw_response_full_v2_20260927", "bindings": bound, "days": {},
    }
    validate_index_identity(base, bound)
    for key, value in (
        ("data_kind", "FIXTURE"), ("study_id", "v2_candidate2"),
        ("processing_run_id", "pilot"), ("bindings", {}),
    ):
        tampered = {**base, key: value}
        with pytest.raises(ValueError):
            validate_index_identity(tampered, bound)


def test_review_must_bind_exact_bytes_and_calendar() -> None:
    bound = {"contract": "a" * 64}
    review = {
        "disposition": "APPROVE_RAW_RESPONSE_PROCESSING_IMPLEMENTATION",
        "reviewer_session": "/root/v2_reviewer",
        "bindings": bound,
        "approved_start": "2020-02-17",
        "approved_end_exclusive": "2020-04-15",
    }
    validate_review(review, bound, "2020-02-17", "2020-04-15")
    with pytest.raises(ValueError):
        validate_review({**review, "bindings": {}}, bound, "2020-02-17", "2020-04-15")
    with pytest.raises(ValueError):
        validate_review(review, bound, "2020-02-17", "2020-04-16")


def test_existing_entry_rejects_identity_and_path_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import tools.v2_process_raw_response as processor

    monkeypatch.setattr(processor, "ROOT", tmp_path)
    monkeypatch.setattr(processor, "EVIDENCE", tmp_path / "evidence")
    monkeypatch.setattr(processor, "OUTPUT", tmp_path / "processed")
    (tmp_path / "evidence").mkdir()
    (tmp_path / "processed").mkdir()
    day = "2020-02-17"
    bound = {"contract": "a" * 64}
    (tmp_path / "processed" / f"{day}.npz").write_bytes(b"real output placeholder")
    manifest = {
        "data_kind": "REAL", "study_id": "raw_response_development_v1",
        "processing_run_id": "raw_response_full_v2_20260927", "day": day,
        "bindings": bound, "shard_path": f"processed/{day}.npz",
        "shard_sha256": processor.sha(tmp_path / "processed" / f"{day}.npz"),
    }
    import json

    path = tmp_path / "evidence" / f"{day}.json"
    path.write_text(json.dumps(manifest))
    entry = {
        "manifest_path": f"evidence/{day}.json", "manifest_sha256": processor.sha(path),
        "shard_sha256": manifest["shard_sha256"],
    }
    validate_existing_entry(day, entry, bound)
    with pytest.raises(ValueError):
        validate_existing_entry(day, {**entry, "manifest_path": "elsewhere.json"}, bound)
    (tmp_path / "processed" / f"{day}.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError):
        validate_existing_entry(day, entry, bound)
    (tmp_path / "processed" / f"{day}.npz").write_bytes(b"real output placeholder")
    manifest["data_kind"] = "FIXTURE"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        validate_existing_entry(day, {**entry, "manifest_sha256": processor.sha(path)}, bound)


def test_orphan_refusal_and_raw_source_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import tools.v2_process_raw_response as processor

    monkeypatch.setattr(processor, "EVIDENCE", tmp_path / "evidence")
    monkeypatch.setattr(processor, "OUTPUT", tmp_path / "processed")
    processor.EVIDENCE.mkdir()
    processor.OUTPUT.mkdir()
    ensure_no_orphans("2020-02-17")
    (processor.OUTPUT / "2020-02-17.npz").write_bytes(b"orphan")
    with pytest.raises(RuntimeError):
        ensure_no_orphans("2020-02-17")
    with pytest.raises(ValueError):
        registered_source_path("../escape.01A")
    with pytest.raises(ValueError):
        registered_source_path("C:/escape.01A")
