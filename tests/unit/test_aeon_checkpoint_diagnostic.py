"""Bounded diagnostic contracts; all fixtures are explicitly synthetic."""

import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from marine_echo.models.aeon_ssl import AeonDirect, AeonTemporalSSL
from tools.aeon_checkpoint_diagnostic import (
    HistoricalInputs,
    centered_stats,
    fixed_batches,
    fixed_endpoint,
    gradient_probe,
    lineage_role,
    sampling_audit,
    temporal_support,
    three_seed_mean,
)


def test_temporal_mapping_is_interleaved_values_then_masks_per_position():
    gradient = torch.zeros(2, 192)
    gradient[0, 8 * 19 + 2] = 3
    gradient[1, 8 * 19 + 6] = 4
    support = temporal_support(gradient)
    assert len(support) == 24
    assert support[19]["value_l2"] == 3
    assert support[19]["mask_l2"] == 4
    assert support[19]["nonzero"] == 2
    assert all(row["nonzero"] == 0 for i, row in enumerate(support) if i != 19)


@pytest.mark.parametrize("mode", ["ema", "shared_sigreg", "direct"])
def test_gradient_probe_keeps_input_and_entire_original_model_unchanged(mode):
    torch.manual_seed(7)
    model = AeonDirect(width=16) if mode == "direct" else AeonTemporalSSL(mode=mode, width=16)
    state = copy.deepcopy(model.state_dict())
    values = torch.randn(64, 24, 4)
    mask = torch.ones_like(values, dtype=torch.bool)
    truth = torch.randn(64, 3)
    before = values.clone()
    result = gradient_probe(model, values, mask, truth, torch.ones_like(truth, dtype=torch.bool))
    assert torch.equal(values, before)
    assert all(torch.equal(v, state[k]) for k, v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())
    total = result["gradients"]["total"]
    suffix = total["first_layer_temporal_support"][18:]
    if mode == "ema":
        assert all(row["nonzero"] == 0 for row in suffix)
        assert total["teacher"]["parameters_with_gradient"] == 0
        assert total["head"]["parameters_with_gradient"] == 0
        assert result["input_gradient_temporal_l2"][18:] == [0.0] * 6
    else:
        assert all(row["nonzero"] > 0 for row in suffix)
    if mode == "shared_sigreg":
        assert result["gradients"]["prediction"]["first_layer_temporal_support"][23]["nonzero"] > 0


def test_centered_rank_does_not_conflate_constant_mean_with_variation():
    stats = centered_stats(torch.full((64, 8), 100.0))
    assert stats["rms"] == 100
    assert stats["centered_rms"] == 0
    assert stats["centered_effective_rank"] == 0
    varied = centered_stats(torch.tensor([[101.0, 100.0], [99.0, 100.0]]))
    assert varied["centered_effective_rank"] == pytest.approx(1)


def test_fixed_endpoint_reports_missing_without_substitution(tmp_path):
    other = tmp_path / "checkpoint-supervised-500.pt"
    other.write_bytes(b"synthetic checkpoint")
    slot = {
        "checkpoints": [
            {
                "phase": "supervised",
                "step": 500,
                "path": other.name,
                "sha256": hashlib.sha256(other.read_bytes()).hexdigest(),
            }
        ]
    }
    assert fixed_endpoint(tmp_path, slot, "supervised", 1500)["status"] == "MISSING_INDEX_ENTRY"
    slot["checkpoints"].append(
        {
            "phase": "supervised",
            "step": 1500,
            "path": "checkpoint-supervised-1500.pt",
            "sha256": "x",
        }
    )
    assert fixed_endpoint(tmp_path, slot, "supervised", 1500)["status"] == "MISSING_ARTIFACT"


def test_fixed_endpoint_rejects_corruption_duplicates_and_escape(tmp_path):
    path = tmp_path / "checkpoint-supervised-1500.pt"
    path.write_bytes(b"synthetic")
    entry = {"phase": "supervised", "step": 1500, "path": path.name, "sha256": "wrong"}
    with pytest.raises(ValueError, match="digest"):
        fixed_endpoint(tmp_path, {"checkpoints": [entry]}, "supervised", 1500)
    with pytest.raises(ValueError, match="ambiguous"):
        fixed_endpoint(tmp_path, {"checkpoints": [entry, entry]}, "supervised", 1500)
    entry["path"] = "../outside.pt"
    with pytest.raises(ValueError, match="path"):
        fixed_endpoint(tmp_path, {"checkpoints": [entry]}, "supervised", 1500)


def test_ensemble_requires_exact_three_seeds():
    predictions = {seed: np.full((3, 3, 5), seed, dtype=float) for seed in (7, 13, 23)}
    assert np.all(three_seed_mean(predictions) == (7 + 13 + 23) / 3)
    with pytest.raises(ValueError, match="three"):
        three_seed_mean({7: predictions[7], 13: predictions[13]})


def test_prior_year_lineage_is_train_for_expanded_descendants():
    assert lineage_role("expanded", "prior") == "TRAIN"
    assert lineage_role("expanded_descendant", "prior") == "TRAIN"
    assert lineage_role("original_frozen", "prior") == "HISTORICAL_DESCRIPTIVE_TRANSFER_ONLY"


def test_corrected_score_excludes_under_supported_source_dates():
    from marine_echo.evaluation.aeon import daily_pinball

    truth = np.zeros((19, 3))
    forecast = np.zeros((19, 3, 5))
    forecast[-1] = 1000
    times = np.full((19, 3), np.datetime64("2024-10-10T01", "us"))
    times[-1] = np.datetime64("2024-10-11T01", "us")
    result = daily_pinball(truth, forecast, np.ones_like(truth, dtype=bool), times)
    assert result["primary_daily_mean_pinball_db"] == 0
    assert len(json.dumps(result)) > 0


def test_fixed_batch_selection_rejects_validation_and_is_deterministic():
    rows = [SimpleNamespace(partition="train") for _ in range(4965)]
    a, b = fixed_batches(rows), fixed_batches(rows)
    assert len(a) == 2 and all(len(batch) == 64 for batch in a)
    assert len(np.unique(np.concatenate(a))) == 128
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    rows[0].partition = "validation"
    with pytest.raises(ValueError, match="TRAIN"):
        fixed_batches(rows)


def test_historical_input_hash_guard_detects_mutation(tmp_path):
    path = tmp_path / "synthetic-historical-input.bin"
    path.write_bytes(b"original")
    inputs = HistoricalInputs()
    inputs.bind(path)
    inputs.verify_unchanged()
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        inputs.verify_unchanged()


def test_source_bound_shuffled_sampling_changes_contexts_and_resets_supervised_rng(tmp_path):
    from marine_echo.training.aeon_campaign import _campaign_code_sha256

    # Explicit synthetic index fixture, never a scientific acoustic dataset.
    rows = [
        SimpleNamespace(cutoff_interval_id=i, target_mask=np.ones(3, dtype=bool))
        for i in range(1600)
    ]
    (tmp_path / "manifest.json").write_text(json.dumps({"code_sha256": _campaign_code_sha256()}))
    result = sampling_audit(rows, HistoricalInputs(), tmp_path)
    aligned, shuffled = result["modes"]["aligned"], result["modes"]["shuffled"]
    assert result["current_code_matches_manifest"]
    assert aligned["duplicate_draws"] == shuffled["duplicate_draws"] == 0
    assert shuffled["minimum_pair_separation_intervals"] >= 24
    assert (
        aligned["pretrain_context_sequence_sha256"] != shuffled["pretrain_context_sequence_sha256"]
    )
    assert (
        aligned["supervised_sequence_sha256_after_actual_reset"]
        == shuffled["supervised_sequence_sha256_after_actual_reset"]
    )
    assert not result["pairing_only_intervention"]


def test_30k_trajectory_reads_protocol_checks_without_endpoint_selection():
    from tools.aeon_checkpoint_diagnostic import validation_checks

    checks = [{"step": 2500}, {"step": 30000}]
    assert validation_checks({"protocol_validation_checks": checks}) == checks
    assert validation_checks({"validation_checks": checks}) == checks
    with pytest.raises(ValueError, match="missing"):
        validation_checks({})
