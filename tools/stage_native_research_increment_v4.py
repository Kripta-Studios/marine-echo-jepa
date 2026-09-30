"""Stage only explicit closed research files; never stage outputs or fixture trees."""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["docs/adr/0024-owner-authorized-vlc-desktop-coexecution.md", "docs/NATIVE_SSL_RESEARCH_CHECKPOINT_V3.md", "orchestration/STATUS.md", "orchestration/native_vlc_desktop_owner_resolution_v1.json", "orchestration/native_band_desktop_admission_v4.json", "orchestration/native_band_desktop_references_v4.json", "orchestration/native_completed_ssl13_inventory_audit_v1.json", "orchestration/native_completed_ssl13_inventory_audit_v2.json", "orchestration/ssl_vnext_band_desktop_operational_prefit_v4.txt", "orchestration/ssl_vnext_native_transfer_corpus_software_review_v1.txt", "orchestration/ssl_vnext_native_prefix_desktop_builder_v3.txt", "src/marine_echo/training/native_desktop_resources_v2.py", "src/marine_echo/training/native_band_operational_ssl.py", "src/marine_echo/training/native_band_operational_downstream.py", "src/marine_echo/evaluation/native_ancestry_inventory.py", "src/marine_echo/training/native_desktop_runtime_v3.py", "src/marine_echo/training/native_prefix_desktop_transfer_v3.py", "tests/unit/test_native_vlc_desktop_ownership_v2.py", "tests/unit/test_native_band_operational_execution.py", "tests/unit/test_native_band_desktop_review_transport_v4.py", "tests/unit/test_native_ancestry_inventory.py", "tests/unit/test_native_inventory_reserved_split.py", "tests/unit/test_native_inventory_real_membership.py", "tests/integration/test_native_ancestry_inventory.py", "tests/unit/test_native_prefix_desktop_execution_v3.py", "tests/integration/test_native_prefix_desktop_transfer_v3.py", "tools/version_native_band_desktop_execution_v3.py", "tools/version_native_band_operational_control_admission_v4.py", "tools/prepare_native_band_operational_policy_checks_v3.py", "tools/execute_native_band_operational_job.py", "tools/execute_native_band_operational_job_v4.py", "tools/prepare_native_band_desktop_admission_v4.py", "tools/materialize_native_band_desktop_reviews_v4.py", "tools/run_reviewed_native_band_desktop_queue_v4.py", "tools/close_native_desktop_operational_checks_v4.py", "tools/close_native_desktop_prefit_rejection_v4.py", "tools/integrate_native_ancestry_inventory_v1.py", "tools/prepare_native_research_inventory.py", "tools/prepare_completed_ssl13_inventory_audit_v1.py", "tools/prepare_completed_ssl13_inventory_audit_v2.py", "tools/integrate_native_prefix_desktop_v3.py", "tools/execute_native_prefix_desktop_job_v3.py", "tools/execute_native_prefix_desktop_worker_v3.py", "tools/package_native_pretrained_snapshot_v1.py", "tools/check_native_pretrained_snapshot_worker_v1.py", "tools/check_native_pretrained_snapshot_owned_v1.py", "tools/check_native_pretrained_snapshot_worker_v2.py", "tools/check_native_pretrained_snapshot_owned_v2.py", "tools/version_native_pretrained_namespace_check_v2.py", "tools/close_native_research_cpu_deliveries_v3.py", "tools/close_native_transfer_corpus_checks_v1.py", "tools/update_native_research_status_v2.py", "tools/stage_native_research_increment_v4.py", "evidence/ssl-research-v1/band-desktop-operational-source-proof-v3.json", "evidence/ssl-research-v1/band-operational-control-admission-proof-v4.json", "evidence/ssl-research-v1/band-operational-policy-test-transport-v3.json", "evidence/ssl-research-v1/band-operational-policy-checks-v3.log", "evidence/ssl-research-v1/band-operational-policy-checks-v4.log", "evidence/ssl-research-v1/band-operational-policy-fixture-green-v3.log", "evidence/ssl-research-v1/vlc-desktop-ownership-red-v2.log", "evidence/ssl-research-v1/vlc-desktop-ownership-red-v3.log", "evidence/ssl-research-v1/vlc-desktop-ownership-green-v2.log", "evidence/ssl-research-v1/band-desktop-review-transport-checks-v4.log", "evidence/ssl-research-v1/band-desktop-operational-root-checks-closeout-v4.json", "evidence/ssl-research-v1/band-desktop-operational-prefit-rejection-closeout-v4.json", "evidence/ssl-research-v1/band-desktop-operational-prefit-events-v4.jsonl", "evidence/ssl-research-v1/band-desktop-operational-prefit-stderr-v4.log", "evidence/ssl-research-v1/native-transfer-corpus-software-review-rejection-closeout-v1.json", "evidence/ssl-research-v1/native-transfer-corpus-software-review-events-v1.jsonl", "evidence/ssl-research-v1/native-transfer-corpus-software-review-stderr-v1.log", "evidence/ssl-research-v1/native-pretrained-model-snapshot-v1.json", "evidence/ssl-research-v1/native-research-cpu-deliveries-closeout-v3.json", "evidence/ssl-native-transfer-corpus-builder-v1/root-checks-closeout-v1.json", "evidence/ssl-native-transfer-corpus-builder-v1/root-owned-lifecycle-v1.log", "evidence/ssl-native-transfer-corpus-builder-v1/SYNTHETIC_CORRECTNESS_ONLY-owned-e11346bb-593e-4ffe-9399-92a498d1dff7/stdout.log", "evidence/ssl-native-transfer-corpus-builder-v1/SYNTHETIC_CORRECTNESS_ONLY-owned-e11346bb-593e-4ffe-9399-92a498d1dff7/resources.json"]


def main():
    paths = set(FILES)
    for folder, receipt in (
        ("evidence/ssl-native-ancestry-inventory-builder-v1", "root-integration-v1.json"),
        ("evidence/ssl-prefix-desktop-builder-v3", "root-integration-v3.json"),
    ):
        document = json.loads((ROOT / folder / receipt).read_bytes())
        paths.update(document["copied_sha256"])
        paths.add(folder + "/" + receipt)
    for folder, names in (
        ("evidence/ssl-native-ancestry-inventory-builder-v1", ("root-checks-v1.log", "root-checks-green-v2.log", "root-checks-green-v3.log", "root-reserved-split-red-v1.log", "root-reserved-split-green-v1.log", "root-membership-red-v2.log", "root-membership-green-v2.log", "root-real-ssl13-audit-v1.log", "root-real-ssl13-audit-v2.log")),
        ("evidence/ssl-prefix-desktop-builder-v3", ("root-checks-v3.log", "root-owned-fixture-v3.log")),
        ("evidence/ssl-prefix-desktop-builder-v3/SYNTHETIC_CORRECTNESS_ONLY-Á-Replay-ROOT-v3", ("owned-process.json", "synthetic-completion.json", "console.log", "resources.json")),
        ("evidence/ssl-research-v1/native-pretrained-snapshot-isolated-cpu-v1", ("stdout.log", "resources.json")),
        ("evidence/ssl-research-v1/native-pretrained-snapshot-isolated-cpu-v2", ("stdout.log", "resources.json", "completion.json")),
    ):
        paths.update(folder + "/" + name for name in names if (ROOT / folder / name).is_file())
    actual = ROOT / "evidence/native-completed-ssl13-inventory-v2"
    paths.update(str(path.relative_to(ROOT)) for path in actual.glob("*.json"))
    for name in paths:
        path = ROOT / name
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise ValueError(f"Exact existing regular research file required: {name}")
        if path.suffix in (".pt", ".npz", ".zip") or path.is_relative_to(ROOT / "outputs"):
            raise ValueError("No model/data binary or output directory staging")
    subprocess.run(["git", "add", "--", *sorted(paths)], cwd=ROOT, check=True)
    print(json.dumps({"status": "EXPLICIT_CLOSED_RESEARCH_FILES_STAGED", "files": len(paths), "ledger_written": False}))


if __name__ == "__main__":
    main()
