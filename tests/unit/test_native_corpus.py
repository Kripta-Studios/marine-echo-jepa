"""Synthetic correctness tests for the separately versioned native contract."""

import json

import numpy as np
import pytest

from marine_echo.data.native_ssl_corpus import NativeSlot, issue_windows, sha256, train_statistics


def slots(count=120, bound=230.0, deployment="site:deployment"):
    return [
        NativeSlot(
            i,
            np.datetime64("2023-01-01T00:00") + np.timedelta64(i, "h"),
            np.array([-70.0, -71.0, -72.0, -73.0]),
            np.ones(4, dtype=bool),
            np.array([[0.0, bound]] * 4),
            ("p",) * 4,
            (150,) * 4,
            deployment,
            "a" * 64,
            ("OBSERVED_CENSORING_UNKNOWN",) * 4,
        )
        for i in range(count)
    ]


def test_native_bounds_and_nonoverlapping_future():
    corpus = issue_windows(slots(), history=96)
    assert corpus["x"].shape == (16, 96, 4)
    assert np.all(corpus["query"][:, :, 4] == 230 / 250)
    assert np.all(corpus["context_ids"][:, -1] < corpus["future_ids"][:, 0, 0])
    assert np.all(corpus["future_ids"][:, :, 0] - corpus["cutoff"][:, None] == [1, 3, 6])


def test_assessment_future_never_changes_inputs_or_issuance():
    original = slots()
    changed = slots()
    changed[101].values[:] = 999
    changed[101].observed[:] = False
    first = issue_windows(original, history=96)
    second = issue_windows(changed, history=96)
    # Cutoff95 precedes modified future; target changes but issuance and inputs do not.
    assert first["row_id"][0] == second["row_id"][0]
    np.testing.assert_array_equal(first["x"][0], second["x"][0])
    np.testing.assert_array_equal(first["observed"][0], second["observed"][0])
    assert first["y_observed"][0, 2] and not second["y_observed"][0, 2]


def test_gap_and_geometry_configuration_are_not_bridged():
    source = slots()
    source[50].bounds[:] = [0, 220]
    assert issue_windows(source, history=96)["x"].shape[0] == 0
    with pytest.raises(ValueError, match="deployment"):
        issue_windows(slots(70) + slots(70, deployment="other"), history=24)


def test_train_statistics_ignore_masked_and_do_not_accept_dev(tmp_path):
    corpus = issue_windows(slots(), history=96)
    split = tmp_path / "split.json"
    split.write_text(
        json.dumps(
            {
                "sources": [
                    {"deployment": "site:deployment", "role": "train", "archive_sha256": "a" * 64}
                ]
            }
        )
    )
    corpus["corpus_role"] = np.asarray("train")
    corpus["split_sha256"] = np.asarray(sha256(split))
    corpus["observed"][0, 0, 0] = False
    corpus["x"][0, 0, 0] = 1e20
    stat = train_statistics(corpus, role="train", split_path=split)
    assert np.all(stat["channel_mean"] == [-70, -71, -72, -73])
    with pytest.raises(ValueError, match="TRAIN"):
        train_statistics(corpus, role="development", split_path=split)
    corpus["corpus_role"] = np.asarray("development")
    with pytest.raises(ValueError, match="TRAIN"):
        train_statistics(corpus, role="train", split_path=split)
    corpus["corpus_role"] = np.asarray("train")
    corpus["deployment"][:] = "wrong-deployment"
    with pytest.raises(ValueError, match="TRAIN"):
        train_statistics(corpus, role="train", split_path=split)


def test_unknown_orientation_and_processing_provenance():
    corpus = issue_windows(slots())
    assert (corpus["metadata"][:, :, 5] == 0).all()  # numeric code 0 explicitly UNKNOWN
    assert (corpus["orientation"] == "UNKNOWN").all()
    assert (corpus["processing_id_or_unknown"] == "p").all()
    assert (corpus["future_processing_id_or_unknown"] == "p").all()
    assert (corpus["context_metadata"][:, :, :, 4] == 230 / 250).all()
