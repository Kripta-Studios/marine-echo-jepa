"""Copy completed pretrained models and static inference source; no fitting or data decode."""

import ast
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_closure():
    package = ROOT / "src/marine_echo"
    pending = [package / "inference" / name for name in ("native_encoder.py", "native_latent.py", "native_acoustic.py")]
    found = set()
    while pending:
        path = pending.pop()
        if path in found:
            continue
        if not path.is_file() or path.is_symlink():
            raise ValueError("Static regular inference source required")
        found.add(path)
        parent = path.parent
        while parent.is_relative_to(package):
            initializer = parent / "__init__.py"
            if initializer.is_file():
                pending.append(initializer)
            parent = parent.parent
        for node in ast.walk(ast.parse(path.read_bytes())):
            names = []
            if isinstance(node, ast.Import):
                names = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module, *(node.module + "." + item.name for item in node.names)]
            for name in names:
                if name.startswith("marine_echo."):
                    candidate = package.joinpath(*name.split(".")[1:]).with_suffix(".py")
                    if candidate.is_file():
                        pending.append(candidate)
    return found


def main():
    folder = ROOT / "outputs/native_pretrained_model_snapshot_v1"
    archive = folder.with_suffix(".zip")
    receipt_path = ROOT / "evidence/ssl-research-v1/native-pretrained-model-snapshot-v1.json"
    if folder.exists() or archive.exists() or receipt_path.exists():
        raise FileExistsError("Preserve previous model snapshots")
    replay_path = ROOT / "evidence/ssl-research-v1/Unicode-actual-latent-Álvaro-cpu-replay-v1/completion.json"
    closeout_path = ROOT / "evidence/ssl-research-v1/native-latent-actual-cpu-replay-closeout-v1.json"
    closeout, replay = (json.loads(path.read_bytes()) for path in (closeout_path, replay_path))
    if closeout.get("actual_cli_exit_code") != 0 or closeout.get("completion_sha256") != digest(replay_path):
        raise ValueError("Actual original four-model CPU replay required")
    payloads, models, bindings = {}, [], {}
    for record in replay["records"]:
        parent = ROOT / "outputs/native_acoustic_ssl_v1" / record["run"]
        report = json.loads((parent / "run.json").read_bytes())
        if (report.get("status") != "COMPLETED" or report.get("evidence_kind") != "REAL_TRAIN_DEVELOPMENT_FIT"
                or report.get("inference_sha256") != digest(parent / "inference.pt")
                or record["artifact_sha256"] != digest(parent / "inference.pt")
                or record["selected_encoder_cpu_replay"] != "BITIDENTICAL"
                or record["saved_forward_head_cpu_replay"] != "BITIDENTICAL"):
            raise ValueError("Exact actual completed and replayed pretrained model required")
        for name in ("selected_encoder.pt", "inference.pt", "scalers.json", "membership.json", "run.json"):
            source = parent / name
            payloads["models/" + record["run"] + "/" + name] = source.read_bytes()
            bindings[str(source)] = digest(source)
        models.append({"id": record["run"], "method": record["method"], "seed": record["seed"],
                       "selected_encoder": "models/" + record["run"] + "/selected_encoder.pt",
                       "latent_inference": "models/" + record["run"] + "/inference.pt"})
    for source in source_closure():
        payloads[str(source.relative_to(ROOT)).replace("\\", "/")] = source.read_bytes()
        bindings[str(source)] = digest(source)
    provenance = {"provenance/native_split.json": ROOT / "configs/native_ssl_split_v1.json",
                  "provenance/original_cpu_replay.json": replay_path,
                  "provenance/original_cpu_replay_closeout.json": closeout_path,
                  "provenance/CF-JEPA-LICENSE.txt": ROOT / "external/cf-jepa-vnext/LICENSE",
                  "provenance/research_report_v1.md": ROOT / "docs/NATIVE_SSL_MODEL_RESEARCH_REPORT_V1.md",
                  "provenance/research_report_addendum_v2.md": ROOT / "docs/NATIVE_SSL_MODEL_RESEARCH_REPORT_ADDENDUM_V2.md"}
    for destination, source in provenance.items():
        payloads[destination] = source.read_bytes()
        bindings[str(source)] = digest(source)
    manifest = {"kind": "native_pretrained_model_snapshot_v1", "status": "INCOMPLETE_RESEARCH_SNAPSHOT",
                "models": models, "source_bindings": bindings, "fitting": False,
                "raw_acoustic_data_included": False, "held_out_results": "NOT_RUN",
                "package_cpu_relocation_test": "NOT_RUN", "independent_package_review": "NOT_RUN",
                "sota": "NOT_ESTABLISHED", "app_release": False,
                "cf_source_revision": "5d3d2fd1273c283fbfa03249c078619245e84033"}
    payloads["manifest.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    payloads["MODEL_CARD.md"] = b"""# Native acoustic pretrained research snapshot

This incomplete local snapshot contains actual TRAIN-only pretrained Shared seed7
and CF-JEPA-inspired seeds7/13/23 encoders, their saved latent predictors, TRAIN
scalers and provenance. It is not a completed held-out study or app release.
Original four-model CPU replay passed; independent review of this copied package
and execution after relocation remain NOT_RUN. No SOTA or representation-transfer
advantage is established. Historical reports included here retain their own scope.

Input: H96 native integrated acoustic history, four observed-channel masks and
native measurement metadata. Preserve native integration bounds (including
0-230m at inference); a product must never be relabelled 0-200m. Horizon queries
are source interval offsets1/3/6, not verified UTC or intervention queries.

The reusable APIs are marine_echo.inference.native_encoder.NativeAcousticEncoder
for context embeddings and marine_echo.inference.native_latent.NativeLatentPredictor
for saved latent outputs. Add src to the Python path in an environment matching
the repository scientific dependencies. Encoder weights are selected_encoder.pt;
latent predictor weights are inference.pt. All loaders use weights_only CPU loading.
Ancestral paths are provenance, not runtime imports or commands to execute.

Shared outputs predict future latent blocks; CF encode uses the selected EMA
branch while its forward latent heads use online features and ordinal zones.
CF full H96 inference is outside its sampled crop support. Latents are not dB,
depth profiles, species, biomass, catch, or causal biological effects. The short
selection readouts included in inference.pt are not the strong transfer endpoints.
The reserved AEON2 final-test site has not been numerically assessed.

Do not run copied training entrypoints without the repository scientific review,
source-binding, resource and budget admission. No cloud or publication is implied.
CF-JEPA source license is included; this snapshot grants no new repository license.
"""
    folder.mkdir()
    for name, raw in payloads.items():
        destination = folder / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(raw)
        if digest(destination) != hashlib.sha256(raw).hexdigest():
            raise ValueError("Snapshot byte copy failed")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as output:
        for name in sorted(payloads):
            output.write(folder / name, name)
    receipt = {"status": "FOUR_ACTUAL_PRETRAINED_MODELS_COPIED_RELOCATION_QA_PENDING",
               "models": models, "directory": str(folder), "archive": str(archive), "archive_sha256": digest(archive),
               "files": {name: digest(folder / name) for name in sorted(payloads)}, "source_bindings": bindings,
               "fitting": False, "raw_corpus_decoded": False, "held_out_results": "NOT_RUN", "app_release": False}
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "models": len(models), "files": len(payloads), "archive_sha256": receipt["archive_sha256"]}))


if __name__ == "__main__":
    main()
