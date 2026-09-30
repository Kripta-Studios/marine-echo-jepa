"""Preserve historical embedded scalers by binding an existing TRAIN scaler file."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("inventory_bound_scalers_support", ROOT / "evidence/ssl-native-ancestry-inventory-builder-v1/synthetic_fixture.py")
support = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = support
spec.loader.exec_module(support)


def prepared():
    fixture = support.Fixture().add("parent").add("child", mode="frozen_readout", parent="parent").finish()
    path = fixture.root / "child/scalers.json"
    original = fixture.root / "parent/scalers.json"
    assert path.read_bytes() == original.read_bytes()
    path.unlink()  # Newly owned visibly synthetic fixture only; no user artifact.
    manifest = json.loads(fixture.manifest.read_bytes())
    manifest["bindings"].pop(str(path))
    manifest["endpoints"]["child"]["scalers_path"] = str(original)
    fixture.rewrite(fixture.manifest, manifest)
    return fixture


def test_missing_standalone_child_scalers_use_exact_bound_existing_train_scalers():
    fixture = prepared()
    result = fixture.candidate.derive_inventory(fixture.manifest, fixture.output)
    assert result["status"] == "DERIVED_METADATA_NOT_FINAL_SELECTION"
    assert len(result["models"]) == 2
    assert result["numeric_corpus_decoded"] is False


@pytest.mark.parametrize("defect", ["unbound", "relative", "wrong_scalers"])
def test_alternate_scalers_cannot_replace_train_normalization(defect, monkeypatch):
    fixture = prepared()
    manifest = json.loads(fixture.manifest.read_bytes())
    path = fixture.root / "parent/scalers.json"
    if defect == "unbound":
        manifest["bindings"].pop(str(path))
    elif defect == "relative":
        manifest["endpoints"]["child"]["scalers_path"] = "parent/scalers.json"
    else:
        replacement = fixture.root / "different-scalers.json"
        scalers = json.loads(path.read_bytes())
        scalers["channel_mean"][0] += 1
        fixture.rewrite(replacement, scalers, rebind=True)
        manifest = json.loads(fixture.manifest.read_bytes())
        manifest["endpoints"]["child"]["scalers_path"] = str(replacement)
    fixture.rewrite(fixture.manifest, manifest)
    if defect != "wrong_scalers":
        monkeypatch.setattr(fixture.candidate, "_safe_load", lambda path: pytest.fail("Unbound scaler metadata reached weights"))
    with pytest.raises((ValueError, FileNotFoundError)):
        fixture.candidate.derive_inventory(fixture.manifest, fixture.output)
