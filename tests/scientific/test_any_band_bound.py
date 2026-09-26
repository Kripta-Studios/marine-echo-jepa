"""Synthetic scientific checks for the TRAIN-only 38 kHz necessary bound."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


def _module():
    path = Path(__file__).resolve().parents[2] / "tools/any_band_bound.py"
    spec = importlib.util.spec_from_file_location("any_band_bound", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _counts():
    return np.zeros((9600, 64), dtype=np.int16), np.full(9600, 60, dtype=np.int16)


def _set_horizons(valid: np.ndarray, cutoff: int, value: int = 48) -> None:
    for cell, offset in enumerate((0, 8, 20)):
        valid[cutoff + offset : cutoff + offset + 4, cell] = value


def test_exact_80_boundary_allows_different_cells_across_horizons() -> None:
    module = _module()
    valid, effective = _counts()
    _set_horizons(valid, 0)
    assert module.any_band_cutoffs(valid, effective) == [0]
    valid[8:12, 1] = 47
    assert module.any_band_cutoffs(valid, effective) == []
    valid[8:12, 1] = 48
    effective[8:12] = 61
    assert module.any_band_cutoffs(valid, effective) == []
    valid[8:12, 1] = 49
    assert module.any_band_cutoffs(valid, effective) == [0]


def test_cross_day_and_last_day_cutoffs_use_cutoff_utc_date() -> None:
    module = _module()
    valid, effective = _counts()
    _set_horizons(valid, 92)
    _set_horizons(valid, 99 * 96 + 72)
    cutoffs = module.any_band_cutoffs(valid, effective)
    assert cutoffs == [92, 99 * 96 + 72]
    assert module.cutoff_dates(cutoffs) == ["2020-02-17", "2020-05-26"]


def test_any_cell_is_a_generous_bound_for_every_nonnegative_weighted_band() -> None:
    module = _module()
    valid, effective = _counts()
    _set_horizons(valid, 0)
    weights = np.zeros(64)
    weights[0] = 0.2
    weights[1] = 0.3
    weights[2] = 0.5
    for offset in (0, 8, 20):
        cell_support = valid[offset : offset + 4].sum(axis=0) / effective[offset : offset + 4].sum()
        weighted_support = float(np.dot(weights, cell_support))
        assert weighted_support <= float(cell_support.max())
    assert module.any_band_cutoffs(valid, effective) == [0]
    assert any(
        np.dot(
            weights,
            valid[offset : offset + 4].sum(axis=0) / effective[offset : offset + 4].sum(),
        )
        < 0.8
        for offset in (0, 8, 20)
    )


def test_overall_day_minimum_status_boundary_is_exact() -> None:
    module = _module()
    assert module.bound_disposition(22) == ("D1_INELIGIBLE_ANY_38KHZ_BAND_SUPPORT_UPPER_BOUND", 89)
    assert module.bound_disposition(23) == ("NO_GLOBAL_INELIGIBILITY_CONCLUSION", 90)
    with pytest.raises(ValueError):
        module.bound_disposition(True)


@pytest.mark.parametrize(
    "field", ["float_valid", "negative", "above_observed", "wrong_effective", "frequency"]
)
def test_day_count_validation_fails_closed(field: str) -> None:
    module = _module()
    day = "2020-02-17"
    valid = np.zeros((96, 4, 64), dtype=np.int16)
    expected = np.full(96, 60, dtype=np.int16)
    observed = expected.copy()
    effective = expected.copy()
    frequency = np.array([38000, 125000, 200000, 455000])
    edges = np.arange(0, 130, 2)
    starts = np.datetime64(day, "ns") + np.arange(96) * np.timedelta64(15, "m")
    if field == "float_valid":
        valid = valid.astype(float)
    elif field == "negative":
        valid[0, 0, 0] = -1
    elif field == "above_observed":
        valid[0, 0, 0] = 48
        observed[0] = 47
    elif field == "wrong_effective":
        effective[0] = 59
    else:
        frequency = frequency[::-1]
    with pytest.raises(ValueError):
        module.validate_day_counts(
            day, valid, expected, observed, effective, starts, frequency, edges
        )


def test_published_map_requires_exact_acceptance_and_recomputed_content(tmp_path: Path) -> None:
    module = _module()
    report_path = tmp_path / module._MAP_REPORT
    report_path.parent.mkdir(parents=True)
    report = {
        "status": "TRAIN_QC_ONLY_NOT_BENCHMARK",
        "census_generation": "census-v2",
        "held_out_acoustic_payloads_processed": False,
        "completed_benchmark_runs": 0,
        "benchmark_eligible": False,
        "processed_train_calendar_days": 100,
        "daily": [{"valid_ping_count": [[0]]}],
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    acceptance = tmp_path / module._MAP_ACCEPTANCE
    acceptance.parent.mkdir(parents=True)
    acceptance.write_text(
        json.dumps(
            {
                "disposition": "ACCEPT_TRAIN_SUPPORT_MAP_RESULT",
                "reviewer_session": "/root/continuation_review",
                "support_map_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    module.verify_accepted_map(tmp_path, report)
    with pytest.raises(ValueError):
        module.verify_accepted_map(tmp_path, report | {"daily": []})
    report_path.write_text(report_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError):
        module.verify_accepted_map(tmp_path, report)


def test_full_synthetic_bound_requires_unchanged_accepted_complete_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    support_test_path = Path(__file__).resolve().parents[1] / "unit/test_train_support_map.py"
    spec = importlib.util.spec_from_file_location("support_map_fixture", support_test_path)
    assert spec is not None and spec.loader is not None
    support_test = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(support_test)
    support_module, _, _ = support_test._complete_synthetic_audit(tmp_path, monkeypatch)
    monkeypatch.setitem(sys.modules, "train_support_map", support_module)
    contract = tmp_path / module._CONTRACT
    contract.parent.mkdir(parents=True, exist_ok=True)
    contract.write_bytes((Path(__file__).resolve().parents[2] / module._CONTRACT).read_bytes())
    review = tmp_path / module._REVIEW
    review.parent.mkdir(parents=True, exist_ok=True)
    review.write_text(
        json.dumps(
            {
                "reviewer_session": "/root/continuation_review",
                "disposition": "APPROVE_ANY_BAND_BOUND_METHOD",
                "reviewed_contract_sha256": hashlib.sha256(contract.read_bytes()).hexdigest(),
                "reviewed_code_sha256": hashlib.sha256(
                    Path(module.__file__).read_bytes()
                ).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    published = support_module.audit_support_map(tmp_path)
    report_path = tmp_path / module._MAP_REPORT
    report_path.write_text(json.dumps(published), encoding="utf-8")
    acceptance = tmp_path / module._MAP_ACCEPTANCE
    acceptance.write_text(
        json.dumps(
            {
                "reviewer_session": "/root/continuation_review",
                "disposition": "ACCEPT_TRAIN_SUPPORT_MAP_RESULT",
                "support_map_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    result = module.audit_any_band_bound(tmp_path)
    assert result["status"] == "D1_INELIGIBLE_ANY_38KHZ_BAND_SUPPORT_UPPER_BOUND"
    assert result["overall_eligible_day_upper_bound"] == 67
    assert result["qualifying_cutoff_dates"] == []
    assert result["completed_benchmark_runs"] == 0
    assert not any("cell_id" in key or "ranking" in key for key in result)
    altered = json.loads(report_path.read_text(encoding="utf-8"))
    altered["daily"][0]["valid_ping_count"][0][0] = 1
    report_path.write_text(json.dumps(altered), encoding="utf-8")
    acceptance.write_text(
        json.dumps(
            {
                "reviewer_session": "/root/continuation_review",
                "disposition": "ACCEPT_TRAIN_SUPPORT_MAP_RESULT",
                "support_map_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="fully recomputed"):
        module.audit_any_band_bound(tmp_path)


def test_missing_method_review_stops_before_count_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: pytest.fail("Count NPZ opened"))
    with pytest.raises(FileNotFoundError):
        module.audit_any_band_bound(tmp_path)


def test_output_publication_is_exclusive_and_cleans_failed_serialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    output = tmp_path / module._OUTPUT
    output.parent.mkdir(parents=True)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(
        module,
        "audit_any_band_bound",
        lambda root: {"status": "SYNTHETIC_TEST_ONLY", "completed_benchmark_runs": 0},
    )
    with monkeypatch.context() as failed:

        def fail_after_partial(result, stream, **kwargs):
            stream.write('{"partial"')
            raise OSError("Synthetic write failure")

        failed.setattr(module.json, "dump", fail_after_partial)
        with pytest.raises(OSError, match="Synthetic write failure"):
            module.main()
    assert not output.exists()
    assert not list(output.parent.glob(".any_band_bound-*.tmp"))
    assert module.main() == 0
    saved = output.read_bytes()
    with pytest.raises(FileExistsError):
        module.main()
    assert output.read_bytes() == saved
    assert not list(output.parent.glob(".any_band_bound-*.tmp"))
