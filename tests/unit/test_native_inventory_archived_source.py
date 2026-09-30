"""Resolve preserved historical source bytes without approving compatibility."""

import copy
import json
from pathlib import Path

import pytest

from marine_echo.evaluation import native_ancestry_inventory as inventory

ROOT = Path(__file__).resolve().parents[2]


def receipt():
    return json.loads((ROOT / "evidence/ssl-research-v1/ancestor-source/manifest.json").read_bytes())


def test_original_shared_source_resolves_to_existing_exact_archive():
    record = receipt()
    before = copy.deepcopy(record)
    resolved = inventory.resolve_original_source_binding(record["original_path"], record["sha256"], "shared_ssl", [record])
    assert resolved == Path(record["path"])
    assert record == before
    assert record["status"] == "ORIGINAL_SOURCE_PRESERVED_NOT_COMPATIBILITY_APPROVAL"


def test_current_source_binding_requires_no_archive():
    record = receipt()
    path = Path(record["original_path"])
    assert inventory.resolve_original_source_binding(path, inventory.sha256(path), "shared_ssl", []) == path


@pytest.mark.parametrize("defect", ["wrong_method", "wrong_digest", "approval_promotion", "wrong_original", "snapshot_substitution", "missing_archive"])
def test_archive_cannot_relabel_bytes_scope_or_authority(defect):
    record = receipt()
    original, digest, method = record["original_path"], record["sha256"], "shared_ssl"
    records = [record]
    if defect == "wrong_method":
        method = "cf_jepa"
    elif defect == "wrong_digest":
        record["sha256"] = "0" * 64
    elif defect == "approval_promotion":
        record["status"] = "APPROVED_COMPATIBILITY"
    elif defect == "wrong_original":
        record["original_path"] = str(ROOT / "src/marine_echo/training/native_ssl.py")
    elif defect == "snapshot_substitution":
        record["path"] = record["original_path"]
    else:
        records = []
    with pytest.raises(ValueError):
        inventory.resolve_original_source_binding(original, digest, method, records)
