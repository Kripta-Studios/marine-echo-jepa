"""Check real reservation metadata without decoding any acoustic values."""

import copy
import json
from pathlib import Path

import pytest

from marine_echo.evaluation import native_ancestry_inventory as inventory

ROOT = Path(__file__).resolve().parents[2]


def split():
    return json.loads((ROOT / "configs/native_ssl_split_v1.json").read_bytes())


def test_original_native_split_preserves_whole_site_final_test_and_four_train_archives():
    document = split()
    before = copy.deepcopy(document)
    groups = inventory.reserved_groups(document, evidence="REAL_TRAIN_DEVELOPMENT_FIT")
    assert document == before
    assert {source["file_id"] for source in groups["train"]} == {61937263, 61937272, 61937275, 61937281}
    assert {source["site"] for source in groups["final_test"]} == {"AEON2_ECS"}
    assert {source["site"] for source in groups["development"]} == {"AEON4_JOB"}


@pytest.mark.parametrize("defect", ["train_row_count_as_archive", "heldout_site_leak", "duplicate_archive", "missing_role", "unknown_role"])
def test_reserved_identity_corruption_rejected(defect):
    document = split()
    if defect == "train_row_count_as_archive":
        document["sources"][0]["file_id"] = 18593
    elif defect == "heldout_site_leak":
        heldout = next(source for source in document["sources"] if source["role"] == "final_test")
        document["sources"][0]["site"] = heldout["site"]
    elif defect == "duplicate_archive":
        document["sources"][1]["archive_sha256"] = document["sources"][0]["archive_sha256"]
    elif defect == "missing_role":
        document["sources"] = [source for source in document["sources"] if source["role"] != "development"]
    else:
        document["sources"][0]["role"] = "all_data_ssl"
    with pytest.raises(ValueError):
        inventory.reserved_groups(document, evidence="REAL_TRAIN_DEVELOPMENT_FIT")
