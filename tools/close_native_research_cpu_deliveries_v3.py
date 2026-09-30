"""Close actual CPU work without scientific prefit or final numeric authority."""

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ancestry = ROOT / "evidence/ssl-native-ancestry-inventory-builder-v1"
    prefix = ROOT / "evidence/ssl-prefix-desktop-builder-v3"
    research = ROOT / "evidence/ssl-research-v1"
    if "61 passed in 352.37s" not in (ancestry / "root-checks-green-v3.log").read_text(encoding="utf-8"):
        raise ValueError("Actual closed61-check suite required")
    if "35 passed, 1 skipped" not in (prefix / "root-checks-v3.log").read_text(encoding="utf-8"):
        raise ValueError("Actual desktop prefix check log required")
    fixture = prefix / "SYNTHETIC_CORRECTNESS_ONLY-Á-Replay-ROOT-v3"
    physical = json.loads((fixture / "owned-process.json").read_bytes())
    json.loads((fixture / "synthetic-completion.json").read_bytes())
    qa = research / "native-pretrained-snapshot-isolated-cpu-v2"
    owned = json.loads((qa / "resources.json").read_bytes())["resources"]
    for resources in (physical, owned):
        if (resources.get("exit_code") != 0 or resources.get("stopped_for") is not None
                or resources.get("owned_tree_cleanup_verified") is not True
                or resources.get("peak_process_rss_bytes", 22 * 1024**3) >= 22 * 1024**3):
            raise ValueError("Actual closed bounded CPU tree required")
    audit = ROOT / "evidence/native-completed-ssl13-inventory-v2/inventory.json"
    inventory = json.loads(audit.read_bytes())
    if inventory.get("train_rows") != 18593 or inventory.get("numeric_corpus_decoded") is not False or len(inventory["models"]) != 1:
        raise ValueError("Actual SSL13 metadata audit required")
    snapshot_path = research / "native-pretrained-model-snapshot-v1.json"
    snapshot = json.loads(snapshot_path.read_bytes())
    if digest(snapshot["archive"]) != snapshot["archive_sha256"]:
        raise ValueError("Model snapshot changed")
    with zipfile.ZipFile(snapshot["archive"]) as archive:
        if set(archive.namelist()) != set(snapshot["files"]):
            raise ValueError("Archive member identities differ")
        for name, sha in snapshot["files"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != sha:
                raise ValueError("Copied archive bytes differ")
    completed = json.loads((qa / "completion.json").read_bytes())
    if completed.get("status") != "COPIED_MODEL_ONLY_PACKAGE_ISOLATED_CPU_CORRECTNESS_PASSED" or len(completed["records"]) != 4:
        raise ValueError("Four copied actual models must pass isolated CPU correctness")
    bindings = {str(path): digest(path) for path in (ancestry / "root-checks-green-v3.log", audit,
                prefix / "root-checks-v3.log", fixture / "owned-process.json", fixture / "synthetic-completion.json",
                snapshot_path, qa / "completion.json", qa / "resources.json")}
    receipt = {"status": "ANCESTRY61_REAL_SSL13_AUDIT_PREFIX35_PHYSICAL_CPU_AND_FOUR_MODEL_PACKAGE_CLOSED",
               "actual_exits": [{"session_id": 50349, "exit_code": 0, "chunk_id": "aa843d"},
                                {"session_id": 39265, "exit_code": 0, "chunk_id": "f31576"},
                                {"session_id": 16780, "exit_code": 0, "chunk_id": "4e6ebf"},
                                {"session_id": 60046, "exit_code": 0, "chunk_id": "6d18a5"},
                                {"session_id": 96705, "exit_code": 0, "chunk_id": "79c7f1"}],
               "bindings": bindings, "prefix_physical_resources": physical, "package_cpu_resources": owned,
               "archive_members_verified": len(snapshot["files"]), "preserved_failures": True,
               "actual_public_prefix_fit": "NOT_RUN", "final_numeric_access": "NOT_RUN",
               "independent_review": "NOT_RUN", "scientific_prefit": "NOT_GRANTED_BY_SOFTWARE_CHECKS"}
    with (research / "native-research-cpu-deliveries-closeout-v3.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "prefit": receipt["scientific_prefit"]}))


if __name__ == "__main__":
    main()
