"""Freeze the complete seed7 development comparison after every endpoint closes."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "outputs/native_acoustic_ssl_v1"


def main():
    manifest = json.loads((ROOT / "orchestration/native_development_comparison_v1.json").read_text(encoding="utf-8"))
    extra = {f"{method}_short_probe": f"{method}_seed7_h96_reviewed"
             for method in ("masked_ssl", "permuted_ssl", "direct", "random_frozen")}
    extra["chronos2_zero_shot"] = "chronos2_h96_local"
    extra.update({f"{method}_{mode}": f"{method}_{mode}_seed7_h96"
                  for method, mode in (("cf_jepa", "frozen_readout"),
                                       ("cf_jepa", "full_finetune"),
                                       ("masked_ssl", "frozen_readout"),
                                       ("masked_ssl", "full_finetune"),
                                       ("permuted_ssl", "frozen_readout"),
                                       ("random_frozen", "frozen_readout"),
                                       ("direct", "frozen_readout"))})
    for name, directory in extra.items():
        parent = ROOT / BASE / directory
        report_path = parent / ("result.json" if name == "chronos2_zero_shot" else "run.json")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        expected = "COMPLETED_ZERO_SHOT_DEVELOPMENT" if name == "chronos2_zero_shot" else "COMPLETED"
        if report.get("status") != expected or not (parent / "predictions.npz").is_file():
            raise ValueError("Every scientific endpoint must actually be completed")
        manifest["methods"][name] = str((parent / "predictions.npz").resolve())
    destination = ROOT / "orchestration/native_development_comparison_v2.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"status": "MANIFEST_FROZEN_NOT_REVIEWED_OR_EXECUTED",
                      "methods": len(manifest["methods"]), "role": "development",
                      "reference": manifest["reference"], "path": str(destination)}))


if __name__ == "__main__":
    main()
