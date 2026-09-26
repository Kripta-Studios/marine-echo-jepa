"""Render the entire predetermined TRAIN day for independent QC review, never test."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main(date="2020-03-03", variant="v2"):
    allowed = {"2020-02-17", "2020-03-01", "2020-04-01", "2020-05-01"}
    if (
        (variant == "v2" and date != "2020-03-03")
        or (variant == "monthly-v1" and date not in allowed)
        or variant not in {"v2", "monthly-v1"}
    ):
        raise ValueError("Only prospectively registered TRAIN review days may be plotted")
    compact = date.replace("-", "")
    evidence = ROOT / (
        "evidence/continuation/train-candidate-v2"
        if variant == "v2"
        else f"evidence/continuation/train-candidate-{compact}-{variant}"
    )
    report = json.loads((evidence / "processing_manifest.json").read_text())
    if report["status"] != "TRAIN_CANDIDATE_PROCESSED_REQUIRES_QC_REVIEW":
        raise ValueError("A completed candidate is required for the TRAIN review page")
    output = ROOT / f"data/processed/candidate-train-{compact}-{variant}"
    start = np.datetime64(date, "ns")
    with np.load(ROOT / report["source_hours"][0]["candidate_file"], allow_pickle=False) as first:
        bins_per_channel = first["pre_noise_Sv"].shape[-1]
        reference_ranges = first["echo_range"][0, 0]
    raw_sum = np.zeros((96, bins_per_channel))
    raw_count = np.zeros((96, bins_per_channel), dtype=int)
    for row in report["source_hours"]:
        with np.load(ROOT / row["candidate_file"], allow_pickle=False) as hour:
            times = hour["ping_time"]
            if not ((times >= start) & (times < start + np.timedelta64(1, "D"))).all():
                raise ValueError("Only the predetermined TRAIN day may be plotted")
            bins = (
                (times - start).astype("timedelta64[ns]").astype("int64") // 900_000_000_000
            ).astype(int)
            values = hour["pre_noise_Sv"][0]
            ranges = hour["echo_range"][0, 0]
            if not np.allclose(ranges, reference_ranges):
                raise ValueError("Plot may not mix changed range geometries")
            valid = np.isfinite(values)
            np.add.at(raw_sum, bins, np.where(valid, 10 ** (values / 10), 0))
            np.add.at(raw_count, bins, valid)
    raw = np.full(raw_sum.shape, np.nan)
    np.divide(raw_sum, raw_count, out=raw, where=raw_count > 0)
    raw = 10 * np.log10(raw)
    with np.load(output / (date + ".npz"), allow_pickle=False) as daily:
        corrected = 10 * np.log10(daily["linear_sv"][:, 0])
        corrected[daily["valid_ping_count"][:, 0] == 0] = np.nan
        support = daily["primary_support"]
        observed = daily["observed_ping_count"]
        expected = daily["expected_ping_count"]
        denominator = daily["support_denominator_ping_count"]
    figure, axes = plt.subplots(4, 1, figsize=(13, 12), layout="constrained", sharex=True)
    panels = [
        (raw, ranges, "38 kHz pre-noise Sv: full recorded range", (0, 510)),
        (raw, ranges, "38 kHz pre-noise Sv: analysis vicinity", (0, 128)),
        (
            corrected,
            np.arange(1, 128, 2),
            "38 kHz noise-corrected candidate, masked invalid cells",
            (0, 128),
        ),
    ]
    for axis, (values, centres, title, limits) in zip(axes, panels):
        display = axis.pcolormesh(
            np.arange(96) * 0.25 + 0.125,
            centres,
            values.T,
            vmin=-100,
            vmax=-40,
            cmap="viridis",
            shading="nearest",
        )
        axis.set_ylim(*limits[::-1])
        axis.set_ylabel("Range from transducer (m)")
        axis.set_title(title, loc="left")
        axis.axhline(10, color="white", lw=0.6, ls="--")
        axis.axhline(100, color="white", lw=0.6, ls="--")
        figure.colorbar(display, ax=axis, label="Sv (dB re 1 m⁻¹)")
    axes[3].plot(np.arange(96) * 0.25 + 0.125, support, label="Primary range/time support")
    axes[3].plot(
        np.arange(96) * 0.25 + 0.125,
        observed / denominator,
        label="Observed/effective ping count",
        ls="--",
    )
    axes[3].axhline(0.8, color="black", lw=0.8, ls=":", label="Frozen 80% support threshold")
    axes[3].set_ylim(0, 1.05)
    axes[3].set_ylabel("Fraction")
    axes[3].legend(loc="lower right")
    axes[3].set_xlabel(f"Hour UTC on {date} (TRAIN only)")
    axes[3].set_xlim(0, 24)
    figure.suptitle(
        "Factory-calibrated TRAIN candidate — QC review pending\nFixed regional environmental assumption; bottom mask not applied",
        fontsize=14,
    )
    figure.savefig(evidence / "training-review.png", dpi=140)
    plt.close(figure)
    text = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Training-only acoustic QC review</title><style>body{{font:17px system-ui;max-width:1100px;margin:2rem auto;padding:1rem}}img{{width:100%}}code{{overflow-wrap:anywhere}}</style><h1>Training-only QC review: 3 March 2020</h1><p>The complete predetermined day is shown, including its missing interval. No validation, interval-calibration or test acoustic values are displayed. This is a candidate, not an approved corpus or forecast result.</p><p>Expected acquisitions: {int(expected.sum())}; observed: {int(observed.sum())}; quarter-hours with at least80% primary support: {int((support >= 0.8).sum())}/96. Support uses the larger of nominal expected and observed counts in each bin.</p><img src="training-review.png" alt="Full training-day raw and corrected echograms with missingness and support"><p>Physical method: serial55170 factory certificate; fixed TRAIN-derived temperature/salinity/pressure. Per-ping upstream background noise removal,10-sample range blocks,3dB SNR;10m proximal exclusion. Slant-range coordinates are not absolute depth. Partial saturation after onboard range averaging is unobservable. Bottom mask: NOT_APPLIED_BOTTOM_NOT_OBSERVED_OR_VALIDATED.</p><p>Reviewer must assess calibration, noise subtraction, geometry, any suspected bottom, gaps and whether additional prospective QC is required before corpus promotion. All source and code hashes are in processing_manifest.json.</p></html>"""
    text = text.replace("3 March 2020", date)
    (evidence / "training-review.html").write_text(text, encoding="utf-8")
    print(str(evidence / "training-review.html"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="2020-03-03")
    parser.add_argument("--variant", choices=["v2", "monthly-v1"], default="v2")
    args = parser.parse_args()
    main(args.date, args.variant)
