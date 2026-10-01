"""SYNTHETIC_CORRECTNESS_ONLY private disk gates; ROOT-only optimizer cases."""

import io
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "evidence/ssl-transfer-integration-builder-v1")
)
from test_support import BUILDER, physical_case, prefix_module, suffix_case, write_json

prefix = prefix_module()


def test_actual_private_metadata_admission_before_arrays_or_rng(monkeypatch):
    c = physical_case()
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("Admission decoded data"))
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("Admission decoded tensors"))
    before = torch.get_rng_state().clone()
    admission = prefix.admit(
        c.folder / "manifest.json", c.folder / "review.json", c.folder / "fresh"
    )
    assert len(admission.partition["fit_rows"]) == 18
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize(
    "damage", ["review", "hash", "source", "suffix_path", "public_smoke", "parent_kind"]
)
def test_stale_or_missing_authority_before_decode(damage, monkeypatch):
    c = physical_case()
    if damage in ("review", "source"):
        import json

        review = json.loads((c.folder / "review.json").read_text())
        if damage == "review":
            review["reviewer_session_id"] = prefix.IMPLEMENTER_SESSION_ID
        else:
            review["bindings"].pop(str(Path(prefix.core.__file__)))
        write_json(c.folder / "review.json", review)
    elif damage == "hash":
        (c.folder / "stats.json").write_text("{}")
    else:
        c.manifest[
            "suffix_numeric_path"
            if damage == "suffix_path"
            else "prefix_npz"
            if damage == "public_smoke"
            else "encoder"
        ] = str(BUILDER.parent / "public-forbidden.npz")
        write_json(c.folder / "manifest.json", c.manifest)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("Invalid fit decoded arrays"))
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("Invalid fit decoded weights"))
    with pytest.raises((ValueError, FileNotFoundError)):
        prefix.admit(c.folder / "manifest.json", c.folder / "review.json", c.folder / "fresh")


def test_actual_weight_codec_context_only_replay_and_original_train_scaling():
    c = suffix_case()
    before = torch.get_rng_state().clone()
    api = prefix.PrefixPredictor(c.folder / "synthetic-model.pt")
    p = api.forecast(
        c.suffix["x"], c.suffix["context_observed"], c.suffix["metadata"], c.suffix["query"]
    )
    assert p.shape == (36, 3, 5) and np.isfinite(p).all()
    assert torch.equal(before, torch.get_rng_state())
    x = c.suffix["x"].copy()
    x[~c.suffix["context_observed"]] = 1e20
    np.testing.assert_array_equal(
        p, api.forecast(x, c.suffix["context_observed"], c.suffix["metadata"], c.suffix["query"])
    )
    assert api.scalers.to_dict() == c.statistics
    bad_metadata, bad_query = c.suffix["metadata"].copy(), c.suffix["query"].copy()
    bad_metadata[:, 0, 4] = 200 / 250
    bad_query[..., 4] = 200 / 250
    with pytest.raises(ValueError, match="230"):
        api.forecast(x, c.suffix["context_observed"], bad_metadata, bad_query)
    nonfinite = x.copy()
    nonfinite[0, -1, 0] = np.inf
    with pytest.raises(ValueError):
        api.forecast(
            nonfinite, c.suffix["context_observed"], c.suffix["metadata"], c.suffix["query"]
        )
    with pytest.raises(TypeError):
        api.forecast(
            x,
            c.suffix["context_observed"],
            c.suffix["metadata"],
            c.suffix["query"],
            targets=c.suffix["targets"],
        )


def test_actual_prefix_decoder_ignores_future_and_never_reads_suffix_values():
    c = physical_case()
    admission = prefix.admit(
        c.folder / "manifest.json", c.folder / "review.json", c.folder / "fresh"
    )
    admission.manifest["_base"] = str(c.folder)
    first = prefix._decode(admission, "prefix_npz", "prefix")
    # An object array would be rejected by allow_pickle=False if accessed.
    transport = io.BytesIO()
    np.savez_compressed(
        transport, **{**c.prefix, "future": np.array(["FORBIDDEN_FUTURE"], dtype=object)}
    )
    admission.snapshots[c.folder / "prefix.npz"] = transport.getvalue()
    # The suffix corpus was never admitted for fitting and cannot supply labels.
    (c.folder / "suffix.npz").write_bytes(
        b"SYNTHETIC perturbed reserved values; intentionally undecodable"
    )
    second = prefix._decode(admission, "prefix_npz", "prefix")
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


def test_actual_conventional_features_fifteen_boosters_and_resume(monkeypatch):
    """Actual tiny LightGBM fit on synthetic contexts only; no Torch optimizer."""
    from dataclasses import replace

    from test_support import native_references

    c = physical_case()
    config = replace(c.config, method="lightgbm", family="reference", mode="conventional")
    policy = {**prefix.REFERENCE_POLICY, "seed": 7}
    admission = prefix.Admission(
        {"evidence_kind": prefix.EVIDENCE}, config, {}, {}, c.partition, {"SYNTHETIC": "a" * 64}
    )
    full = c.folder / "conventional-full"
    full.mkdir()
    reference = prefix._fit_reference(admission, c.prefix, c.dev, full, policy, c.statistics)
    assert len(reference["candidates"]) == 4
    assert reference["label_counts"] == [18, 18, 18]
    assert reference["original_train_scalers"] == c.statistics
    loaded = prefix.ReferencePredictor(full / "inference.json")
    actual = loaded.forecast(
        c.suffix["x"], c.suffix["context_observed"], c.suffix["metadata"], c.suffix["query"]
    )
    features = native_references.feature_matrix(
        {
            "x": c.suffix["x"],
            "observed": c.suffix["context_observed"],
            "metadata": c.suffix["metadata"],
        },
        96,
    )
    expected = np.sort(
        np.stack(
            [
                b.predict(features, num_iteration=loaded.iteration, num_threads=4)
                for b in loaded.boosters
            ],
            axis=1,
        ).reshape(36, 3, 5),
        axis=-1,
    )
    np.testing.assert_array_equal(actual, expected)
    partial = c.folder / "conventional-partial"
    partial.mkdir()
    original_save = prefix.core.atomic_checkpoint
    with monkeypatch.context() as patch:

        def stop_after_saved(path, value):
            original_save(path, value)
            if len(value["boosters"]) == 2:
                raise RuntimeError("SYNTHETIC explicit interruption after two saved boosters")

        patch.setattr(prefix.core, "atomic_checkpoint", stop_after_saved)
        with pytest.raises(RuntimeError):
            prefix._fit_reference(admission, c.prefix, c.dev, partial, policy, c.statistics)
    resume = partial / "latest.pt"
    admission.snapshots[resume] = resume.read_bytes()
    prefix._fit_reference(admission, c.prefix, c.dev, partial, policy, c.statistics, resume=resume)
    resumed = prefix.ReferencePredictor(partial / "inference.json")
    np.testing.assert_array_equal(
        actual,
        resumed.forecast(
            c.suffix["x"], c.suffix["context_observed"], c.suffix["metadata"], c.suffix["query"]
        ),
    )
    assert not torch.cuda.is_initialized()


ROOT = (
    os.environ.get("NATIVE_PREFIX_MATCHED_ROOT_CHECKS") == "1"
    and BUILDER.name == "marine-echo-jepa"
)


@pytest.mark.skipif(
    not ROOT, reason="NOT_RUN builder: retained optimizer/cache denial; ROOT-only fixture"
)
@pytest.mark.parametrize("mode", ["frozen_readout", "scratch_direct"])
def test_root_actual_cf_optimizer_exact_resume(mode):
    import importlib.util
    from dataclasses import replace

    spec = importlib.util.spec_from_file_location(
        "matched_cf_unit", BUILDER / "tests/unit/test_native_prefix_matched_transfer_v4.py"
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    c = physical_case()
    selected, _ = helper.selected()
    config = helper.cfg() if mode == "frozen_readout" else helper.cfg("direct", mode)
    backbone = (
        selected["config"]
        if mode == "frozen_readout"
        else prefix.cf_controls.core_config(
            prefix.cf_controls.Config(
                width=8, latent=8, blocks=1, updates=4, cadence=1, batch_size=4
            )
        ).to_dict()
    )
    args = (
        config,
        backbone,
        selected if mode == "frozen_readout" else None,
        c.statistics,
        c.prefix,
        c.dev,
        {"SYNTHETIC": "a" * 64},
    )
    full = prefix._Trajectory(*args)
    rng = prefix.core.rng_state()
    full.advance()
    partial = prefix._Trajectory(*args)
    prefix.core.restore_rng(rng)
    partial.advance(2)
    resumed = prefix._Trajectory(*args)
    resumed.restore(prefix.decode_checkpoint(prefix.encode_checkpoint(partial.checkpoint())))
    resumed.advance()
    assert full.samples == resumed.samples and len(full.candidates) == 4
    assert all(
        torch.equal(v, resumed.model.state_dict()[k]) for k, v in full.model.state_dict().items()
    )
    artifact = resumed.inference_artifact()
    loaded = prefix.PrefixPredictor(io.BytesIO(prefix.encode_checkpoint(artifact)))
    expected = prefix.predict(loaded.model, c.suffix, loaded.scalers, replace(config, device="cpu"))
    np.testing.assert_array_equal(
        expected,
        loaded.forecast(
            c.suffix["x"], c.suffix["context_observed"], c.suffix["metadata"], c.suffix["query"]
        ),
    )
