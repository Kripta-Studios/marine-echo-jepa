"""Publication and reviewed-input protection for the counts-only plot."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "support_plot", Path(__file__).resolve().parents[2] / "tools/plot_train_support_map.py"
)
assert SPEC and SPEC.loader
plot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot)


def test_plot_rejects_missing_result_review(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        plot.verified_report(tmp_path)


def test_plot_publication_failure_leaves_no_final_artifact(tmp_path: Path) -> None:
    def failing_renderer(path: Path) -> None:
        path.write_bytes(b"partial synthetic test image")
        raise OSError("simulated rendering failure")

    target = tmp_path / "figure"
    with pytest.raises(OSError, match="simulated"):
        plot.publish_figure(target, {}, failing_renderer)
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_plot_preserves_existing_artifact(tmp_path: Path) -> None:
    target = tmp_path / "figure"
    target.mkdir()
    marker = target / "retained.txt"
    marker.write_text("existing")
    with pytest.raises(FileExistsError):
        plot.publish_figure(target, {}, lambda path: path.write_bytes(b"replacement"))
    assert marker.read_text() == "existing"


def test_plot_manifest_failure_does_not_publish_png(tmp_path: Path) -> None:
    target = tmp_path / "figure"
    with pytest.raises(ValueError):
        plot.publish_figure(
            target, {"invalid": float("nan")}, lambda path: path.write_bytes(b"fixture")
        )
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("mutation", ["report", "bound_source", "plot_code"])
def test_plot_rejects_changed_reviewed_input(tmp_path: Path, mutation: str) -> None:
    bindings = {
        "code_sha256": "tools/train_support_map.py",
        "contract_sha256": "evidence/continuation/train_support_map_contract.json",
        "execution_report_sha256": "evidence/continuation/train_census_v2_execution.json",
        "strict_context_report_sha256": "evidence/continuation/train_census_v2_eligibility.json",
        "original_target_only_report_sha256": "evidence/continuation/target_only_bound.json",
    }
    report = {}
    for key, name in bindings.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic test lineage")
        report[key] = plot.digest(path)
    method = tmp_path / "orchestration/reviews/TRAIN_SUPPORT_MAP_20260927.json"
    method.parent.mkdir(parents=True)
    method.write_text(
        json.dumps(
            {
                "reviewer_session": "/root/continuation_review",
                "disposition": "APPROVE_TRAIN_SUPPORT_MAP_METHOD",
                "reviewed_code_sha256": report["code_sha256"],
                "reviewed_contract_sha256": report["contract_sha256"],
            }
        )
    )
    report["review_sha256"] = plot.digest(method)
    source = tmp_path / "evidence/continuation/train_support_map.json"
    source.write_text(json.dumps(report))
    review = {
        "reviewer_session": "/root/continuation_review",
        "disposition": "ACCEPT_TRAIN_SUPPORT_MAP_RESULT",
        "support_map_sha256": plot.digest(source),
        "reviewed_plot_code_sha256": plot.digest(Path(plot.__file__)),
    }
    review_path = method.with_name("TRAIN_SUPPORT_MAP_RESULT_20260927.json")
    review_path.write_text(json.dumps(review))
    assert plot.verified_report(tmp_path) == report
    if mutation == "report":
        source.write_text(json.dumps({**report, "fabricated_count": 48}))
    elif mutation == "bound_source":
        (tmp_path / bindings["execution_report_sha256"]).write_text("changed ledger")
    else:
        review["reviewed_plot_code_sha256"] = "0" * 64
        review_path.write_text(json.dumps(review))
    with pytest.raises(ValueError):
        plot.verified_report(tmp_path)


def test_plot_rename_failure_cleans_only_owned_stage(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "figure"

    def fail_rename(source, destination):
        target.mkdir()
        (target / "retained.txt").write_text("racing existing artifact")
        raise FileExistsError("publication race")

    monkeypatch.setattr(plot.os, "rename", fail_rename)
    with pytest.raises(FileExistsError):
        plot.publish_figure(target, {}, lambda path: path.write_bytes(b"fixture"))
    assert (target / "retained.txt").read_text() == "racing existing artifact"
    assert list(tmp_path.iterdir()) == [target]
