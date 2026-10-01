"""Original supervised ancestors retain their actually approved cadence."""

import copy
import json
from pathlib import Path

import pytest

from marine_echo.evaluation.native_ancestry_inventory import _config

ROOT = Path(__file__).resolve().parents[2]


def original(name):
    manifest = json.loads((ROOT / "orchestration/native_completed_inventory_v3.json").read_bytes())
    entry = manifest["endpoints"][name]
    report = json.loads((Path(entry["directory"]) / "run.json").read_bytes())
    return copy.deepcopy(report.get("core_config", report["config"])), entry


@pytest.mark.parametrize("name", ["direct_short_probe", "direct_frozen_readout"])
def test_actual_approved_supervised_ancestor_750_cadence_is_admitted(name):
    config, entry = original(name)
    assert config["pretrain_cadence"] == 750
    _config(config, entry)


@pytest.mark.parametrize("name", ["direct_short_probe", "direct_frozen_readout"])
def test_supervised_ancestor_cadence_cannot_be_relabelled_as_ssl(name):
    config, entry = original(name)
    config["pretrain_cadence"] = 1500
    with pytest.raises(ValueError, match="backbone/objective/schedule"):
        _config(config, entry)


def test_real_ssl_cadence_cannot_be_changed_to_supervised_cadence():
    manifest = json.loads((ROOT / "orchestration/native_completed_inventory_v3.json").read_bytes())
    name = next(name for name, entry in manifest["endpoints"].items()
                if entry["method"] == "shared_ssl" and entry["mode"] == "core_frozen_readout")
    config, entry = original(name)
    assert config["pretrain_cadence"] == 1500
    config["pretrain_cadence"] = 750
    with pytest.raises(ValueError, match="backbone/objective/schedule"):
        _config(config, entry)
