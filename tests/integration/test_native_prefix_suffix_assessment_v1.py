"""SYNTHETIC_CORRECTNESS_ONLY physical weights/score reconstruction, no fit."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "evidence/ssl-transfer-integration-builder-v1")
)
from test_support import rebind_suffix_fixture, suffix_case, write_json

from marine_echo.evaluation import native_prefix_suffix_assessment_v1 as suffix


def test_actual_saved_weights_suffix_scores_and_protected_output():
    c = suffix_case()
    report = suffix.run(
        c.folder / "suffix-manifest.json",
        c.folder / "suffix-review.json",
        c.folder / "suffix-output",
    )
    assert report["status"] == "COMPLETED" and report["zero_shot"] is False
    assert report["results"]["scratch"]["label_counts"] == [18, 18, 18]
    for name in ("scratch", "persistence"):
        with np.load(c.folder / "suffix-output" / (name + ".npz"), allow_pickle=False) as archive:
            predicted, targets = archive["predictions"], archive["targets"]
            mask, dates = archive["observed"], archive["target_dates"]
            residual = targets[..., None] - predicted
            q = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
            loss = np.maximum(q * residual, (q - 1) * residual).mean(-1)
            hs = []
            for h in range(3):
                ds = [
                    loss[(dates[:, h] == date) & mask[:, h], h].mean()
                    for date in np.unique(dates[:, h])
                    if ((dates[:, h] == date) & mask[:, h]).sum() >= 18
                ]
                hs.append(np.mean(ds))
            assert report["results"][name]["metrics"]["primary_pinball_db"] == pytest.approx(
                np.mean(hs)
            )
            np.testing.assert_array_equal(archive["observed"], c.suffix["target_observed"])
            assert np.all(archive["query_native_bounds_m"][..., 1] == 230)
    assert report["paired"]["persistence"]["interval"] == [0.0, 0.0]
    with pytest.raises(FileExistsError):
        suffix.run(
            c.folder / "suffix-manifest.json",
            c.folder / "suffix-review.json",
            c.folder / "suffix-output",
        )
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize(
    "damage",
    [
        "execution",
        "software",
        "numeric",
        "freeze",
        "parent",
        "prefix_days",
        "partition",
        "cross_deployment",
        "source",
    ],
)
def test_guarded_suffix_admission_rejects_before_any_decode(damage, monkeypatch):
    c = suffix_case()
    target = (
        "suffix-review.json"
        if damage == "execution"
        else "suffix-software.json"
        if damage == "software"
        else "suffix-access.json"
        if damage == "numeric"
        else "freeze.json"
        if damage == "freeze"
        else "synthetic-completion.json"
        if damage in ("parent", "partition")
        else "suffix-manifest.json"
    )
    value = json.loads((c.folder / target).read_text())
    if damage in ("execution", "software", "numeric"):
        value["status"] = "APPROVED_FINAL_ASSESSMENT_EXECUTION"
    elif damage == "freeze":
        value["suffix_selection"] = True
    elif damage == "parent":
        value["zero_shot"] = True
    elif damage == "partition":
        value["partition"]["fit_interval_ids"].append(value["partition"]["suffix_interval_ids"][0])
    elif damage == "prefix_days":
        value["cells"][0]["prefix_days"] = 7
    elif damage == "cross_deployment":
        value["cells"][0]["deployment"] = "foreign-source"
    else:
        value["input_npz"] = str(c.folder.parent / "public-forbidden.npz")
    write_json(c.folder / target, value)
    rebind_suffix_fixture(c)
    monkeypatch.setattr(
        np, "load", lambda *a, **k: pytest.fail("Wrong suffix approval decoded arrays")
    )
    monkeypatch.setattr(
        torch, "load", lambda *a, **k: pytest.fail("Wrong suffix approval decoded weights")
    )
    with pytest.raises((ValueError, FileNotFoundError)):
        suffix.admit(
            c.folder / "suffix-manifest.json",
            c.folder / "suffix-review.json",
            c.folder / "suffix-output",
        )
