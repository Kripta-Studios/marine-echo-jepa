"""Prepare closed-weight replay admission using hashes/JSON only."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    manifest_path = ROOT / "orchestration/native_latent_cpu_replay_v1.json"
    output = ROOT / "evidence/ssl-research-v1/Unicode-actual-latent-Álvaro-cpu-replay-v1"
    runs = [
        ("shared_ssl_seed7_h96_cuda0", "shared_ssl", 7),
        ("cf_jepa_seed7_h96_deterministic", "cf_jepa", 7),
        ("cf_jepa_seed13_h96_replication", "cf_jepa", 13),
        ("cf_jepa_seed23_h96_replication", "cf_jepa", 23),
    ]
    manifest = {
        "kind": "native_latent_cpu_replay_manifest_v1", "role": "development",
        "context_rows": 64, "fitting": False, "final_numeric_access": False,
        "runs": [{"directory": str(ROOT / "outputs/native_acoustic_ssl_v1" / name),
                  "method": method, "seed": seed} for name, method, seed in runs],
    }
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    software = ROOT / "evidence/ssl-research-v1/native-latent-software-review-final.json"
    review = json.loads(software.read_text(encoding="utf-8"))
    bindings = dict(review["bindings"])
    paths = [software, manifest_path, Path(__file__).resolve(),
             ROOT / "tools/execute_native_latent_cpu_replay_v1.py",
             ROOT / "data/processed/native_ssl_v1/development.npz",
             ROOT / "data/processed/native_ssl_v1/development.json",
             ROOT / "configs/native_ssl_split_v1.json",
             ROOT / "evidence/ssl-latent-builder-v1/root-durable-fixture-v1.log",
             ROOT / "evidence/ssl-latent-builder-v1/root_durable_fixture_check.py"]
    for run in manifest["runs"]:
        folder = Path(run["directory"])
        report = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        if (report.get("status") != "COMPLETED"
                or report.get("evidence_kind") != "REAL_TRAIN_DEVELOPMENT_FIT"
                or report["config"]["method"] != run["method"]
                or report["config"]["seed"] != run["seed"]
                or report.get("inference_sha256") != digest(folder / "inference.pt")):
            raise ValueError("Only exact original completed real weights admitted")
        paths.extend(folder / name for name in ("run.json", "inference.pt", "selected_encoder.pt", "membership.json"))
    for path in paths:
        bindings[str(path)] = digest(path)
    proposal = {
        "status": "PROPOSED_REPLAY_ADMISSION_NOT_APPROVAL", "bindings": bindings,
        "manifest_sha256": digest(manifest_path), "approved_output": str(output),
        "role": "development", "device": "cpu", "optimizer_updates": 0,
        "context_keys_only": ["x", "observed", "metadata", "query"],
        "final_numeric_access": False,
        "command": [str(ROOT / ".venv/Scripts/python.exe"), "-B",
                    "tools/execute_native_latent_cpu_replay_v1.py", "--manifest", str(manifest_path),
                    "--review", str(ROOT / "evidence/ssl-research-v1/native-latent-cpu-replay-review-final.json"),
                    "--output", str(output)],
    }
    path = ROOT / "orchestration/native_latent_cpu_replay_admission_v1.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(proposal, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": proposal["status"], "bindings": len(bindings),
                      "manifest": str(manifest_path), "public_numeric_decode": False}))


if __name__ == "__main__":
    main()
