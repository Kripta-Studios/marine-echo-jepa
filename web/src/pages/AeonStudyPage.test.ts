import { renderToStaticMarkup } from "react-dom/server";
import { createElement } from "react";
import { describe, expect, it } from "vitest";
import { AeonStudyPage } from "./AeonStudyPage";
import type { AeonStudyEvidence } from "../api/client";

const study: AeonStudyEvidence = {
  study_id: "aeon3_geb_2024_hourly_sv_v1",
  title: "AEON3 Georges Basin hourly acoustic development",
  classification: "REVIEWED_TRAIN_VALIDATION_DEVELOPMENT_NOT_FINAL_EVALUATION",
  source_time_basis: "SOURCE_REPORTED_UNSPECIFIED_NOT_UTC",
  source: { publisher: "Figshare AEON", site: "Georges Basin", archive_sha256: "a".repeat(64) },
  target: {
    source_variable: "Sv_mean", frequency_hz: 38000, product: "60minFullDepth",
    nominal_layer_m: [0, 200], unit: "dB re 1 m^-1",
    calibration_claim: "SOURCE_REPORTED_CONDITIONED_NOT_INDEPENDENTLY_FIELD_VERIFIED",
    horizon_source_interval_steps: [1, 3, 6],
  },
  assessment_partition: "validation", final_evaluation: false,
  selection: "NOT_PERFORMED_IN_THIS_REPORT", cached_forecasts: 0,
  calibration_outcomes: "NOT_OPENED_FOR_THIS_REPORT",
  retrospective_test_outcomes: "NOT_OPENED_FOR_THIS_REPORT",
  development: {
    core: { status: "INDEPENDENTLY_REVIEWED", issued_rows: 1219, eligible_days_per_horizon: [50, 50, 50], slot_primary_pinball_db: { ridge: 0.6676 } },
    hybrid: { status: "INDEPENDENTLY_REVIEWED", primary_pinball_db: { core_direct_equal_three_seed_ensemble: 0.6391 } },
    post_hoc_supervised: { family: "LightGBM", status: "INDEPENDENTLY_REVIEWED_POST_HOC_DEVELOPMENT", primary_pinball_db: 0.6385 },
    forward_ema: { status: "INDEPENDENTLY_REVIEWED_POST_HOC_DEVELOPMENT", primary_pinball_db: 0.6533, individual_seed_primary_pinball_db: {} },
    chronos2: { status: "INDEPENDENTLY_REVIEWED_POST_HOC_ZERO_SHOT_DEVELOPMENT", primary_pinball_db: 0.6935, model_revision: "fixed-review-revision" },
  },
  limitations: ["No AEON forecasts are cached or served by this release."],
};

describe("AEON study page", () => {
  it("labels validation metrics and never offers an AEON forecast action", () => {
    const html = renderToStaticMarkup(createElement(AeonStudyPage, { study, state: "success" }));
    expect(html).toContain("source-reported conditioned");
    expect(html).toContain("not UTC");
    expect(html).toContain("TRAIN/validation development");
    expect(html).toContain("0.6385");
    expect(html).not.toContain("Forecast here");
    expect(html).toContain("0.6533");
    expect(html).toContain("0.6935");
    expect(html).toContain("post-hoc development evidence");
  });
});
