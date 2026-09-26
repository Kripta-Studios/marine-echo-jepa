import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from marine_echo.evaluation.metrics import daily_metrics, paired_blocks
from marine_echo.evaluation.protocol import digest_file, verify_seal
from marine_echo.serving.api import create_app


def test_empty_forged_review_cannot_open_test(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}")
    seal = tmp_path / "seal.json"
    seal.write_text(
        json.dumps(
            {
                "review_verdict": "APPROVED",
                "review_id": "R2",
                "protocol_sha256": digest_file(protocol),
                "files": [],
            }
        )
    )
    with pytest.raises(ValueError):
        verify_seal(protocol, seal)


def test_abstaining_on_worst_row_cannot_generate_a_winning_score() -> None:
    truth = np.array([[0.0], [100.0]])
    pred = np.zeros((2, 1, 5))
    pred[1] = np.nan
    result = daily_metrics(
        truth, pred, np.array(["2020-01-01T00:00", "2020-01-02T00:00"], dtype="datetime64[m]")
    )
    assert result["daily_mean_pinball_db"] is None
    assert result["prediction_coverage"] == 0.5


def test_duplicate_bootstrap_anchors_rejected() -> None:
    with pytest.raises(ValueError):
        paired_blocks(
            np.array(["2020-01-01", "2020-01-01"], dtype="datetime64[D]"), np.ones((2, 2))
        )


def test_host_spoofing_rejected(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path))
    result = client.get(
        "/health", headers={"Host": "evil.example", "Origin": "http://evil.example"}
    )
    assert result.status_code == 400
