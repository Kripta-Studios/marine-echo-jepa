"""Render the complete reviewed count-support map with a fixed zero-to-one scale."""

import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_report(root: Path) -> dict:
    source = root / "evidence/continuation/train_support_map.json"
    result_review = json.loads(
        (root / "orchestration/reviews/TRAIN_SUPPORT_MAP_RESULT_20260927.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        result_review.get("reviewer_session") != "/root/continuation_review"
        or result_review.get("disposition") != "ACCEPT_TRAIN_SUPPORT_MAP_RESULT"
        or result_review.get("support_map_sha256") != digest(source)
        or result_review.get("reviewed_plot_code_sha256") != digest(Path(__file__))
    ):
        raise ValueError("Independent review of exact count report and plot code required")
    report = json.loads(source.read_text(encoding="utf-8"))
    bindings = {
        "code_sha256": "tools/train_support_map.py",
        "contract_sha256": "evidence/continuation/train_support_map_contract.json",
        "review_sha256": "orchestration/reviews/TRAIN_SUPPORT_MAP_20260927.json",
        "execution_report_sha256": "evidence/continuation/train_census_v2_execution.json",
        "strict_context_report_sha256": "evidence/continuation/train_census_v2_eligibility.json",
        "original_target_only_report_sha256": "evidence/continuation/target_only_bound.json",
    }
    for field, relative in bindings.items():
        if report.get(field) != digest(root / relative):
            raise ValueError(f"Reviewed support-map binding differs: {field}")
    method = json.loads((root / bindings["review_sha256"]).read_text(encoding="utf-8"))
    if (
        method.get("reviewer_session") != "/root/continuation_review"
        or method.get("disposition") != "APPROVE_TRAIN_SUPPORT_MAP_METHOD"
        or method.get("reviewed_code_sha256") != report["code_sha256"]
        or method.get("reviewed_contract_sha256") != report["contract_sha256"]
    ):
        raise ValueError("Exact approved support-map method required")
    return report


def publish_figure(output: Path, provenance: dict, render: Callable[[Path], None]) -> None:
    """Publish the PNG and manifest together by a no-replace Windows directory rename."""
    if output.exists() or output.is_symlink():
        raise FileExistsError("Preserve existing count-map figure")
    with tempfile.TemporaryDirectory(prefix=".support-figure-", dir=output.parent) as directory:
        staged = Path(directory)
        image = staged / "support.png"
        render(image)
        with image.open("r+b") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        record = {**provenance, "png_sha256": digest(image)}
        with (staged / "provenance.json").open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.rename(staged, output)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source = ROOT / "evidence/continuation/train_support_map.json"
    report = verified_report(ROOT)
    if (
        report["status"] != "TRAIN_QC_ONLY_NOT_BENCHMARK"
        or report["processed_train_calendar_days"] != 100
        or report["held_out_acoustic_payloads_processed"] is not False
        or report["frequency_hz"] != [38000, 125000, 200000, 455000]
        or report["range_edges_m"] != list(range(0, 130, 2))
    ):
        raise ValueError("Reviewed full TRAIN map required")
    days = report["daily"]
    dates = np.arange("2020-02-17", "2020-05-27", dtype="datetime64[D]").astype(str).tolist()
    if [day["date"] for day in days] != dates:
        raise ValueError("Exact complete TRAIN calendar required")
    valid = np.array([day["valid_ping_count"] for day in days])
    effective = np.array([day["effective_ping_count"] for day in days])
    if (
        valid.shape != (100, 4, 64)
        or effective.shape != valid.shape
        or np.any(effective <= 0)
        or np.any(valid < 0)
        or np.any(valid > effective)
    ):
        raise ValueError("Invalid daily counts")
    output = ROOT / "evidence/continuation/train-support-map-figure"
    figure, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True, constrained_layout=True)
    for channel, axis in enumerate(axes):
        raster = axis.imshow(
            (valid[:, channel] / effective[:, channel]).T,
            origin="upper",
            aspect="auto",
            extent=(-0.5, 99.5, 128, 0),
            vmin=0,
            vmax=1,
            interpolation="nearest",
            cmap="viridis",
        )
        axis.set_ylabel(f"{report['frequency_hz'][channel] / 1000:g} kHz\nRange (m)")
        axis.set_yticks([0, 32, 64, 96, 128])
    ticks = [0, 14, 28, 42, 56, 70, 84, 99]
    axes[-1].set_xticks(ticks, [dates[index] for index in ticks], rotation=25, ha="right")
    axes[-1].set_xlabel("Original TRAIN UTC date; every day and range cell retained")
    figure.colorbar(raster, ax=axes, label="Daily valid / effective ping count", shrink=0.85)
    figure.suptitle(
        "Complete TRAIN QC support map — not a benchmark or target selection\n"
        "Unchanged calibration/QC; no held-out acoustic data; fixed 0–1 color scale",
        fontsize=13,
    )
    record = {
        "scope": "Complete TRAIN daily count visualization only; not per-cell target eligibility",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "color_scale": [0, 1],
        "shape": [100, 4, 64],
        "all_dates_channels_ranges_shown": True,
    }
    try:
        publish_figure(output, record, lambda path: figure.savefig(path, dpi=150))
    finally:
        plt.close(figure)
    print(json.dumps(record))


if __name__ == "__main__":
    main()
