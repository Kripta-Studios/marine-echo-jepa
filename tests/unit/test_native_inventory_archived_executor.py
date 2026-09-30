"""An original executor is source provenance, never an executable fallback."""

import json
from pathlib import Path

import pytest

from marine_echo.evaluation import native_ancestry_inventory as inventory

ROOT = Path(__file__).resolve().parents[2]


def test_exact_recovered_original_executor_bytes_can_be_bound_as_provenance():
    record = json.loads((ROOT / "evidence/ssl-research-v1/ancestor-source/execute_native_ssl_job_shared_screen_v1.manifest.json").read_bytes())
    assert inventory.resolve_original_source_binding(record["original_path"], record["sha256"], "shared_ssl", [record]) == Path(record["path"])
    assert record["status"] == "ORIGINAL_SOURCE_PRESERVED_NOT_COMPATIBILITY_APPROVAL"


def test_original_shared_executor_archive_cannot_authorize_cf_execution():
    record = json.loads((ROOT / "evidence/ssl-research-v1/ancestor-source/execute_native_ssl_job_shared_screen_v1.manifest.json").read_bytes())
    with pytest.raises(ValueError):
        inventory.resolve_original_source_binding(record["original_path"], record["sha256"], "cf_jepa", [record])
