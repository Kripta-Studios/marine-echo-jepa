"""Record all completed CF strong endpoints and the retained native failure."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    ledger = json.loads(
        (ROOT / "orchestration/native_ssl_run_ledger_v1.json").read_text(encoding="utf-8")
    )
    models = []
    for seed in (7, 13, 23):
        for mode in ("frozen_readout", "full_finetune"):
            name = f"cf_jepa_{mode}_seed{seed}_h96"
            if seed == 13 and mode == "frozen_readout":
                name += "_runtime_retry01"
            folder = ROOT / "outputs/native_acoustic_ssl_v1" / name
            run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
            record = next(r for r in ledger["runs"] if r["id"] == name)
            if (
                run.get("status") != "COMPLETED"
                or run["config"]["seed"] != seed
                or run.get("mode") != mode
                or record.get("status") != "COMPLETED_REAL_DOWNSTREAM"
                or record["run_sha256"] != digest(folder / "run.json")
                or not record["resources_full_attempt"]["owned_tree_cleanup_verified"]
            ):
                raise ValueError("Actual strong endpoint identity or cleanup differs")
            models.append(
                {
                    "id": name,
                    "seed": seed,
                    "mode": mode,
                    "path": str(folder),
                    "run_sha256": digest(folder / "run.json"),
                    "inference_sha256": digest(folder / "inference.pt"),
                    "selected_encoder_sha256": digest(folder / "selected_encoder.pt"),
                    "selected_supervised_step": run["selected_supervised_step"],
                    "development_pinball_db": run["selected_daily_dev_pinball"],
                    "resources_full_attempt": record["resources_full_attempt"],
                    "feature_training": "SSL_PRETRAINED_FROZEN"
                    if mode == "frozen_readout"
                    else "SUPERVISED_FULL_FINETUNING_AFTER_SSL",
                }
            )
    result = {
        "status": "SIX_REAL_CF_STRONG_ENDPOINTS_VERIFIED",
        "models": models,
        "all_seeds_preserved": True,
        "seed_selection": "NONE",
        "test_access": "NOT_RUN",
        "native_seed13_failed_attempt": "Retained unchanged, charged164.14s, followed by one fresh-output same-recipe retry after bounded CUDA smoke",
        "seed13_queue_actual_exit_witness": "Root session45636 chunk179f95 exit0",
        "seed23_queue_actual_exit_witness": "Root session22909 chunk840387 exit0",
        "metric_scope": "Development runner reports; expanded independent numerical reconstruction pending",
        "scientific_claim": "NOT_ESTABLISHED",
    }
    with (ROOT / "evidence/ssl-research-v1/cf-strong-three-seed-completion-v1.json").open(
        "x", encoding="utf-8"
    ) as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "endpoints": len(models)}))


if __name__ == "__main__":
    main()
