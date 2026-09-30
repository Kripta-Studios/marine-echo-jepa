"""Bind the explicit owner reply; never infer permission from elapsed time."""

import datetime
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT / "orchestration/native_band_budget_owner_resolution_v1.json"
    original = ["shared_ssl", "cf_jepa", "masked_ssl", "permuted_ssl", "direct", "random_frozen"]
    revision = ["band_shared_ssl", "band_masked_ssl", "band_permuted_ssl", "band_random_frozen", "band_direct_end_to_end"]
    value = {"status": "ROOT_RESOLVED", "kind": "native_band_owner_budget_resolution_v1",
             "recorded_utc": datetime.datetime.now(datetime.UTC).isoformat(),
             "owner_authorization": {"source": "explicit user question reply in this session",
                                     "question_item_id": '["request_user_input_async","call_Rv7gJ3z47TmhGL550vOhbcqZ",0]',
                                     "verbatim_answer": "Allow up to 11 recipes, including controls",
                                     "verbatim_question": "For the single band-conditioning revision in ADR0020, should the programme allow up to 11 total seed-7 screening recipes, including matched controls, with at most 12 additional GPU-hours inside the existing 96-hour budget?"},
             "seed7_recipe_limit_including_controls": 11,
             "original_recipes": original, "single_revision_recipes": revision,
             "revision_gpu_hours_limit_full_owned": 12,
             "aggregate_gpu_hours_limit_full_owned": 96,
             "revision_cost_scope": "All band-family scientific GPU operations, including screening, controls, later allowed replication/readouts, inference/assessment, failures and resumes; no free startup/evaluation/save time",
             "additional_cf_control_status": "PROPOSED_UNFITTED_DEFERRED_NOT_IN_AUTHORIZED_ELEVEN",
             "new_architecture_or_lr_search": "NOT_AUTHORIZED",
             "prefit_source_review": "REQUIRED_NOT_GRANTED_BY_OWNER_REPLY",
             "whole_site_final_numeric_access": "NOT_AUTHORIZED_BY_THIS_REPLY",
             "local_resource_limits_unchanged": True, "cloud_or_paid_resource_approval": False,
             "app_release_work": "FROZEN", "historical_evidence_rewritten": False}
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"status": "OWNER_RESOLUTION_RECORDED", "total_recipes": len(original) + len(revision),
                      "revision_hours": 12, "aggregate_hours": 96, "fitting": False}))


if __name__ == "__main__":
    main()
