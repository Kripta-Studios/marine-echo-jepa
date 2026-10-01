"""Fixed recipe and source registration correctness; no acoustic-array decoding."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import execute_native_cf_control_job_v1 as executor
import prepare_native_cf_control_configs_v1 as registration


def test_catalog_matches_actual_training_factory():
    from marine_echo.training.native_cf_controls import Config

    for recipe in executor.catalog().values():
        config = Config(**recipe)
        config.validate()
        assert config.to_dict() == recipe


def test_executor_source_closure_covers_actual_trainer():
    from marine_echo.training.native_cf_controls import Config, source_paths

    config = Config(**next(iter(executor.catalog().values())))
    assert set(source_paths(config)) <= set(executor.source_paths(executor.ROOT))


def test_original_campaign_preserves_exact_47_and_43_guard():
    from marine_echo.evaluation.native_seed_summary_v1 import CAMPAIGN_METHODS, REFERENCES

    path = executor.ROOT / "orchestration/native_development_comparison_v3.json"
    value = json.loads(path.read_bytes())
    assert len(CAMPAIGN_METHODS) == 47
    assert len(CAMPAIGN_METHODS - REFERENCES) == 43
    assert len(value["methods"]) == 28
    assert set(value["methods"]) <= CAMPAIGN_METHODS
    assert not any(name.startswith("cf_random_frozen") for name in CAMPAIGN_METHODS)


def test_registration_rejects_foreign_checkout_before_any_source_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(registration, "source_paths", lambda *a: pytest.fail("source loading"))
    with pytest.raises(ValueError, match="integrated research checkout"):
        registration.prepare(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_registration_preserves_existing_manifest_before_imports(tmp_path, monkeypatch):
    monkeypatch.setattr(registration, "ROOT", tmp_path)
    folder = tmp_path / "orchestration"
    folder.mkdir()
    path = folder / "native_cf_matched_controls_v1.json"
    path.write_bytes(b"SYNTHETIC_CORRECTNESS_ONLY keep existing")
    monkeypatch.setattr(registration, "source_paths", lambda *a: pytest.fail("source loading"))
    with pytest.raises(FileExistsError, match="Preserve"):
        registration.prepare(tmp_path)
    assert path.read_bytes() == b"SYNTHETIC_CORRECTNESS_ONLY keep existing"
