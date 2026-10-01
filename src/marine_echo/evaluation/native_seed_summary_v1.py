"""Summarize reconstructed fixed-campaign seed scores without selecting a seed.

This pure reporting function reads no predictions, models or corpora and grants
no scientific approval. It requires the complete original campaign identities.
Matched CF extension fits are reported separately, never inserted into its 47.
"""

import math
import statistics


def triplet(base):
    return (base, base + "_seed13", base + "_seed23")


GROUPS = {
    "cf_short_probe": triplet("cf_short_probe"),
    "cf_strong_frozen": triplet("cf_jepa_frozen_readout"),
    "cf_full_finetune": triplet("cf_jepa_full_finetune"),
    "shared_scratch": triplet("direct_end_to_end"),
    "band_short_probe": triplet("band_shared_ssl_short_probe"),
    "band_strong_frozen": triplet("band_shared_ssl_frozen_readout"),
    "band_full_finetune": triplet("band_shared_ssl_full_finetune"),
    "band_scratch": triplet("band_direct_end_to_end"),
}
REFERENCES = frozenset(("persistence", "seasonal24", "lightgbm", "chronos2_zero_shot"))
ORIGINAL_UNREPLICATED = (
    "direct_frozen_readout",
    "direct_short_probe",
    "masked_ssl_frozen_readout",
    "masked_ssl_full_finetune",
    "masked_ssl_short_probe",
    "permuted_ssl_frozen_readout",
    "permuted_ssl_short_probe",
    "random_frozen_frozen_readout",
    "random_frozen_short_probe",
    "shared_frozen_readout",
    "shared_full_finetune",
    "shared_short_probe",
)
BAND_UNREPLICATED = (
    "band_masked_ssl_short_probe",
    "band_permuted_ssl_short_probe",
    "band_random_frozen_short_probe",
    "band_masked_ssl_frozen_readout",
    "band_masked_ssl_full_finetune",
    "band_permuted_ssl_frozen_readout",
    "band_random_frozen_frozen_readout",
)
CAMPAIGN_METHODS = frozenset(
    [name for group in GROUPS.values() for name in group]
    + list(REFERENCES)
    + list(ORIGINAL_UNREPLICATED)
    + list(BAND_UNREPLICATED)
)


def summarize(result):
    """Return means of three seed scores, not scores of averaged predictions."""
    if result.get("role") != "development":
        raise ValueError("Only the original development campaign is accepted")
    methods = result.get("methods")
    if not isinstance(methods, dict) or len(methods) != 47:
        raise ValueError("Exactly 47 completed method records are required")
    if set(methods) != CAMPAIGN_METHODS:
        raise ValueError("Fixed original 43-neural/47-method identities differ")
    scores = {}
    for name, record in methods.items():
        value = record["metrics"]["primary_pinball_db"]
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("A numerical seed score or explicit None is required")
            if not math.isfinite(value) or value < 0:
                raise ValueError("Pinball score must be finite and nonnegative")
        scores[name] = value
    groups = {}
    for name, members in GROUPS.items():
        values = [scores[member] for member in members]
        complete = all(value is not None for value in values)
        groups[name] = {
            "status": "ASSESSABLE" if complete else "NOT_ASSESSABLE",
            "seeds": [7, 13, 23],
            "methods": list(members),
            "seed_scores_db": values,
            "mean_seed_score_db": statistics.mean(values) if complete else None,
            "sample_sd_db": statistics.stdev(values) if complete else None,
            "estimand": "mean_of_fixed_seed_scores_not_prediction_ensemble",
        }
    return {
        "role": "development",
        "neural_endpoints": 43,
        "references": 4,
        "seed_score_groups": groups,
        "ensemble_scores": "NOT_COMPUTED",
        "matched_cf_extension": "SEPARATE_STUDY_NOT_INCLUDED",
        "independent_numeric_review": "REQUIRED_SEPARATELY",
        "sota": "NOT_ESTABLISHED_BY_THIS_SUMMARY",
    }
