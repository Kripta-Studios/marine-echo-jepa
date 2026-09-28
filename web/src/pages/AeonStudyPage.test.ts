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

  it("separates reviewed CAL widening from validation and final TEST", () => {
    const calibrated: AeonStudyEvidence = {
      ...study,
      classification: "REVIEWED_DEVELOPMENT_AND_CALIBRATION_NOT_FINAL_EVALUATION",
      calibration_outcomes: "INDEPENDENTLY_REVIEWED_CALIBRATION_ONLY",
      calibration: {
        status: "INDEPENDENTLY_REVIEWED_CALIBRATION_ONLY",
        classification: "CALIBRATION_ONLY_NOT_MODEL_SELECTION_OR_FINAL_EVALUATION",
        issued_rows: 810, eligible_days_per_horizon: [34, 34, 34],
        artifact_sha256: "b".repeat(64), outcome_review_sha256: "c".repeat(64),
        models: Object.fromEntries([
          "core_direct_equal_three_seed_ensemble", "core_ema_equal_three_seed_ensemble",
          "post_hoc_lightgbm",
        ].map((key) => [key, {
          selection_classification: "FROZEN_BEFORE_CAL", raw_primary_pinball_db: 0.6531,
          raw_per_horizon_pinball_db: [0.57, 0.68, 0.70],
          raw_coverage90_per_horizon: [0.82, 0.77, 0.79],
          interval_widening_db_by_horizon: [0.47, 0.77, 0.78],
          widened_in_sample_coverage90_per_horizon: [0.901, 0.901, 0.900],
        }])),
      },
    };
    const html = renderToStaticMarkup(createElement(AeonStudyPage, { study: calibrated, state: "success" }));
    expect(html).toContain("CALIBRATION ONLY");
    expect(html).toContain("810 issued CAL rows");
    expect(html).toContain("0.6531");
    expect(html).toContain("0.470 / 0.770 / 0.780");
    expect(html).toContain("do not select or rank models");
    expect(html).toContain("retrospective TEST remains unopened");
    expect(html).not.toContain("CAL and retrospective TEST outcomes are not opened");
  });

  it("labels external primary ineligibility and secondary descriptive negative result", () => {
    const transferred: AeonStudyEvidence = {
      ...study,
      external_transfer: {
        study_id: "aeon_external_transfer_20260928_v1",
        status: "INDEPENDENTLY_REVIEWED_EXTERNAL_TRANSFER",
        classification: "POST_HOC_INITIATED_EXTERNAL_TRANSFER_NOT_SEALED",
        source: {
          publisher_article: "https://figshare.com/articles/dataset/AZFP/29247113",
          publisher_article_version: 2, file_ids: [61937269, 61937275],
          license: "CC BY 4.0", original_archives_bundled: false,
          acoustic_quantity: "source-reported conditioned Sv_mean",
        },
        primary: {
          status: "METADATA_INELIGIBLE_NO_CANDIDATES",
          site: "AEON2 Eastern Coastal Shelf",
          candidate_count: 0,
          fixed_target_layer_geometry_m: "0:200",
          observed_layer_geometry_m: "0:230",
          observed_38khz_rows: 8682,
          numeric_sv_access: "NOT_RUN_METADATA_INELIGIBLE",
          jepa_value_gate: "NOT_EVALUATED_METADATA_INELIGIBLE",
        },
        secondary: {
          status: "COMPLETED_ZERO_SHOT_EXTERNAL_TRANSFER",
          site: "AEON3 Georges Basin",
          deployment: "Feb2023-Feb2024",
          candidate_count: 8507,
          issued_rows: 8507,
          eligible_dates_per_horizon: [353, 353, 353],
          direct_daily_pinball_db: 0.670178309914835,
          ema_daily_pinball_db: 0.678921677909798,
          ema_relative_loss_reduction: -0.0130463309027024,
          paired_95pct_ema_minus_direct_db: [0.00225278251735256, 0.0155161330474045],
          jepa_value_gate: "DESCRIPTIVE_ONLY_NOT_PRIMARY_GATE",
        },
        outcome_review_sha256: "d".repeat(64),
        manifest_sha256: "e".repeat(64),
      },
    };
    const html = renderToStaticMarkup(createElement(AeonStudyPage, { study: transferred, state: "success" }));
    expect(html).toContain("POST-HOC EXTERNAL TRANSFER");
    expect(html).toContain("0–230 m");
    expect(html).toContain("8,682");
    expect(html).toContain("0.6702");
    expect(html).toContain("0.6789");
    expect(html).toContain("-1.30%");
    expect(html).toContain("prior-year same-site");
    expect(html).toContain("No cross-site model result");
    expect(html).toContain("61937269");
    expect(html).toContain("CC BY 4.0");
  });
});
