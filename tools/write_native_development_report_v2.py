"""Write an executed development report from the frozen twenty-method result."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    result_path = ROOT / "evidence/ssl-research-v1/development-comparison-result-v2.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    pairs = {row["method"]: row for row in result["comparisons"]}
    lines = [
        "# Native acoustic development comparison: twenty executed endpoints",
        "",
        ("Status: actual reconstruction exited 0 under distinct 150-binding admission review. "
        "Independent numerical reconstruction of this expanded matrix is NOT_RUN. "
        "The earlier eight-method reconstruction was independently verified."),
        "",
        ("All results concern one development deployment, AEON4_JOB:61937266, at its native "
        "0–225 m integrated 38-kHz product. Whole-site AEON2 final-test numerical values remain "
        "unopened. These are development comparisons; they do not establish transfer, "
        "JEPA value, SOTA or causal biological effects."),
        "",
        ("The saved outputs preserve the same 7,593 issued rows, targets, masks, source dates "
        "and query geometry. There are 7,581/7,557/7,521 observed targets at horizons 1/3/6; "
        "floor18 yields 316 eligible dates per horizon and 7,572/7,548/7,512 scored rows. "
        "No prediction-based filtering, support intersection, imputation, geometry renaming "
        "or test threshold tuning was performed. All twenty forecasts vary across issuance."),
        "",
        "| Endpoint | Development pinball (dB) | Difference vs LightGBM | Paired 95% interval (dB) |",
        "|---|---:|---:|---|",
    ]
    for name, value in sorted(
        result["methods"].items(), key=lambda item: item[1]["metrics"]["primary_pinball_db"]
    ):
        point = value["metrics"]["primary_pinball_db"]
        pair = pairs.get(name)
        interval = (
            "reference" if pair is None else f"[{pair['ci95_db'][0]:.6f}, {pair['ci95_db'][1]:.6f}]"
        )
        difference = "0" if pair is None else f"{pair['point_difference_db']:+.6f}"
        lines.append(f"| {name} | {point:.6f} | {difference} | {interval} |")
    lines += [
        "",
        ("Lower loss is better; a positive difference favours LightGBM. Intervals use the "
        "predeclared paired 2,000-draw bootstrap, seed20260929, 48 nominal source-calendar hours, "
        "with gaps and duplicate sampled-block multiplicities retained. They are conditional on "
        "this one deployment, not intervals over independent sites or verified UTC observations."),
        "",
        ("The shared frozen candidate scores0.785715, versus masked0.786842, random0.812868, "
        "permuted0.854690 and supervised features0.776118. Its3.34% improvement over random "
        "features is a development observation; supervised features perform better and the "
        "masked difference is small. Full shared fine-tuning0.772594 does not beat matched "
        "scratch supervised0.765185. Masked full fine-tuning0.691806 is stronger here."),
        "",
        ("CF frozen0.614422 and full fine-tuning0.541814 are the strongest local neural endpoints "
        "in this matrix, but both trail LightGBM0.505274. CF uses128-dimensional features and "
        "18,565 head parameters; shared methods use64 and5,189. These controls therefore "
        "do not isolate the effect of the CF SSL objective. Chronos2 zero-shot0.506916 is close "
        "to LightGBM; its external pretraining ancestry is unknown and it does not inherit "
        "the local fitted-ancestor exclusion guarantee."),
        "",
        ("Short probes use500 supervised updates and are development checkpoint-selection "
        "instruments. Strong frozen endpoints use fresh2,000-update readouts; full and "
        "scratch endpoints use3,000 updates. Four scheduled selection opportunities, matched "
        "supervision/sample streams where specified, warmup and cosine schedules are retained. "
        "Equal updates are not equal compute. SSL pretraining/probes, full downstream costs, "
        "startup/evaluation/saving/failures and serial ownership are accounted separately in "
        "the immutable run/resource receipts and active96-hour ledger. Three-seed replication "
        "is still executing; this matrix contains seed7 only."),
        "",
        ("The structural band-pooling proof uses synthetic inputs and does not establish the "
        "cause of these numerical gaps. Its proposed revision requires the pending ADR0020 "
        "owner budget clarification and source-prefit approval. No revised fit has occurred."),
        "",
        ("Remaining evidence: independent expanded numerical reconstruction, three-seed strong "
        "endpoints/direct reference, frozen pretest selection, held-out native230m comparisons, "
        "secondary-channel robustness, supported prefix adaptation, relocated model-only "
        "inference package and independent scientific/package review. Missing runs are not "
        "positive results. Software, experiment, JEPA-value and business gates remain separate."),
        "",
        f"Result SHA256: `{hashlib.sha256(result_path.read_bytes()).hexdigest()}`.",
        "",
        ("Sources: `orchestration/native_development_comparison_v2.json`, "
        "`evidence/ssl-research-v1/development-comparison-v2-review-final.json`, "
        "`evidence/ssl-research-v1/development-comparison-result-v2.json`, "
        "`evidence/ssl-research-v1/development-comparison-execution-v2.log`."),
        "",
    ]
    destination = ROOT / "evidence/ssl-research-v1/NATIVE_SSL_DEVELOPMENT_COMPARISON_V2.md"
    with destination.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    print(
        json.dumps(
            {
                "status": "REPORT_WRITTEN",
                "independent_expanded_numeric_review": "NOT_RUN",
                "path": str(destination),
                "methods": len(result["methods"]),
            }
        )
    )


if __name__ == "__main__":
    main()
