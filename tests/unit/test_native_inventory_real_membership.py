"""Use completed sampling metadata without loading weights or acoustic arrays."""

import copy
import json
from pathlib import Path

import pytest

from marine_echo.evaluation import native_ancestry_inventory as inventory

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def metadata():
    member = json.loads((ROOT / "outputs/native_acoustic_ssl_v1/band_shared_ssl_seed13_h96_replication_v2/membership.json").read_bytes())
    member["sequence"] = member["sequence"][:1]
    cohort = json.loads((ROOT / "orchestration/native_assessment_seed7_metadata_v1/train_cohort.json").read_bytes())
    report = json.loads((ROOT / "data/processed/native_ssl_v1/train.json").read_bytes())
    split = json.loads((ROOT / "configs/native_ssl_split_v1.json").read_bytes())
    train = [source for source in split["sources"] if source["role"] == "train"]
    rows = [tuple(row) for row in cohort["rows"]]
    return member, rows, train, report


def test_actual_context_and_target_membership_layout_is_valid(metadata):
    member, rows, train, report = metadata
    before = copy.deepcopy(member)
    assert inventory._check_membership(member, rows, train, report) == rows
    assert member == before


@pytest.mark.parametrize("defect", ["foreign_context", "foreign_target", "duplicate_context", "ambiguous_layout", "context_hash", "target_hash"])
def test_actual_sampling_metadata_corruption_is_rejected(metadata, defect):
    member, rows, train, report = metadata
    item = member["sequence"][0]
    if defect == "foreign_context":
        item["context_indices"][0] = len(rows)
    elif defect == "foreign_target":
        item["target_indices"][0] = len(rows)
    elif defect == "duplicate_context":
        item["context_indices"][0] = item["context_indices"][1]
    elif defect == "ambiguous_layout":
        item["indices"] = item["context_indices"].copy()
    elif defect == "context_hash":
        item["context_sha256"] = "0" * 64
    else:
        item["target_multiset_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        inventory._check_membership(member, rows, train, report)
