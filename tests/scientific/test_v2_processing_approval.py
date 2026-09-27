import pytest

from tools.v2_process_native import validate_processing_review


def test_review_cannot_impersonate_an_unassigned_session():
    bound = {"processor": {"sha256": "p"}, "aggregation": {"sha256": "a"}}
    review = {
        "disposition": "APPROVE_NATIVE_PROCESSING_IMPLEMENTATION",
        "processor_sha256": "p",
        "aggregation_sha256": "a",
        "reviewer_session": "/root/not_the_reviewer",
        "approved_start": "2020-02-17",
        "approved_end": "2020-02-18",
    }
    with pytest.raises(ValueError, match="distinct-session"):
        validate_processing_review(review, bound, "2020-02-17", "2020-02-18")


def test_one_day_review_cannot_authorize_longer_processing():
    bound = {"processor": {"sha256": "p"}, "aggregation": {"sha256": "a"}}
    review = {
        "disposition": "APPROVE_NATIVE_PROCESSING_IMPLEMENTATION",
        "processor_sha256": "p",
        "aggregation_sha256": "a",
        "reviewer_session": "/root/v2_reviewer",
        "approved_start": "2020-02-17",
        "approved_end": "2020-02-18",
    }
    validate_processing_review(review, bound, "2020-02-17", "2020-02-18")
    with pytest.raises(ValueError, match="scope"):
        validate_processing_review(review, bound, "2020-02-17", "2020-04-15")
