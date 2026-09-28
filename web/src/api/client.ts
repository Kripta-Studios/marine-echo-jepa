export interface Selection {
  datasetId: string;
  modelId: string;
  cutoff: string;
  age: number;
}

export function forecastRequest(selection: Selection) {
  return {
    dataset_id: selection.datasetId,
    model_id: selection.modelId,
    cutoff: selection.cutoff,
    observation_age_hours: selection.age,
    mode: "cached_replay" as const,
  };
}

export interface Dataset {
  id: string;
  name: string;
  domain: string;
  license: string;
  calibration_status: string;
  start: string;
  end: string;
  source_sha256: string;
  observations_id: string;
}
export interface ObservationRow {
  event_time_utc: string;
  counts: (number | null)[][];
  ping_count: number;
}
export interface Observations {
  rows: ObservationRow[];
  frequency_hz: number[];
  units: string;
  calibrated: boolean;
  data_kind: string;
  range_convention: string;
}
export interface Model {
  id: string;
  family: string;
  status: string;
  reason: string;
}
export interface Evidence {
  title: string;
  release_class?: string;
  forecast_unavailability_reason?: string;
  limitations: string[];
  source_files: { name: string; bytes: number; sha256: string }[];
  gates: Record<string, string>;
  protocol: unknown;
  attribution: string;
}

export interface RawDevelopmentHorizonMetrics {
  horizon_hours: 1 | 3 | 6;
  eligible_rows: number;
  target_days: number;
  ridge: {
    daily_mean_pinball_code: number;
    mae_code_median: number;
  };
  direct_neural: {
    daily_mean_pinball_code: number;
    mae_code_median: number;
  };
}

export interface RawDevelopmentPrediction {
  horizon_hours: 1 | 3 | 6;
  target_start_utc: string;
  eligible: boolean;
  truth_code: number | null;
  ridge_quantiles_code: number[];
  direct_quantiles_code: number[];
}

export interface RawDevelopmentRow {
  cutoff_utc: string;
  horizons: RawDevelopmentPrediction[];
}

export interface RawDevelopmentEvidence {
  study_id: string;
  run_id: string;
  status: "REAL_TRAIN_DEVELOPMENT_ONLY";
  quantity: "complete_positive_azfp_backscatter_r_code_mean";
  unit: string;
  calibrated: false;
  final_evaluation: false;
  comparison_label: string;
  direct_checkpoint_128_sha256: string;
  horizons: RawDevelopmentHorizonMetrics[];
  rows: RawDevelopmentRow[];
  limitations: string[];
}

export interface AeonStudyEvidence {
  study_id: "aeon3_geb_2024_hourly_sv_v1";
  title: string;
  classification: "REVIEWED_TRAIN_VALIDATION_DEVELOPMENT_NOT_FINAL_EVALUATION" | "REVIEWED_DEVELOPMENT_AND_CALIBRATION_NOT_FINAL_EVALUATION" | "REVIEWED_RETROSPECTIVE_TEST_NOT_SEALED";
  source_time_basis: "SOURCE_REPORTED_UNSPECIFIED_NOT_UTC";
  source: { publisher: string; site: string; archive_sha256: string };
  target: {
    source_variable: "Sv_mean";
    frequency_hz: 38000;
    product: "60minFullDepth";
    nominal_layer_m: number[];
    unit: string;
    calibration_claim: "SOURCE_REPORTED_CONDITIONED_NOT_INDEPENDENTLY_FIELD_VERIFIED";
    horizon_source_interval_steps: number[];
  };
  assessment_partition: "validation" | "retrospective_test";
  final_evaluation: boolean;
  selection: "NOT_PERFORMED_IN_THIS_REPORT" | "FROZEN_BEFORE_TEST_NO_RELEASE_RERANKING";
  cached_forecasts: number;
  calibration_outcomes: "NOT_OPENED_FOR_THIS_REPORT" | "INDEPENDENTLY_REVIEWED_CALIBRATION_ONLY";
  retrospective_test_outcomes: "NOT_OPENED_FOR_THIS_REPORT" | "INDEPENDENTLY_REVIEWED_RETROSPECTIVE_TEST";
  retrospective_test?: {
    status: "INDEPENDENTLY_REVIEWED_RETROSPECTIVE_TEST";
    classification: "RETROSPECTIVE_EVALUATION_NOT_SEALED";
    issued_rows: number;
    eligible_days_per_horizon: number[];
    jepa_value_gate: "PASSED_FROZEN_WITHIN_STUDY_RETROSPECTIVE_GATE";
    jepa_representation_attribution: "NOT_ESTABLISHED";
    test_score_sha256: string;
    test_outcome_review_sha256: string;
    models: Record<string, {
      selection_classification: string;
      raw_primary_daily_mean_pinball_db: number;
      raw_per_horizon_pinball_db: number[];
      raw_coverage90_per_horizon: number[];
      widened_coverage90_per_horizon: number[];
    }>;
    comparisons: Record<string, {
      primary_relative_loss_change: number | null;
      paired_95_percent_interval_db: number[];
      passes_full_unnarrowed_promotion_rule: boolean;
    }>;
  };
  external_transfer?: {
    study_id: "aeon_external_transfer_20260928_v1";
    status: "INDEPENDENTLY_REVIEWED_EXTERNAL_TRANSFER";
    classification: "POST_HOC_INITIATED_EXTERNAL_TRANSFER_NOT_SEALED";
    source: {
      publisher_article: string;
      publisher_article_version: 2;
      file_ids: number[];
      license: "CC BY 4.0";
      original_archives_bundled: false;
      acoustic_quantity: "source-reported conditioned Sv_mean";
    };
    primary: {
      status: "METADATA_INELIGIBLE_NO_CANDIDATES";
      site: string;
      candidate_count: 0;
      fixed_target_layer_geometry_m: "0:200";
      observed_layer_geometry_m: string;
      observed_38khz_rows: number;
      numeric_sv_access: "NOT_RUN_METADATA_INELIGIBLE";
      jepa_value_gate: "NOT_EVALUATED_METADATA_INELIGIBLE";
    };
    secondary: {
      status: "COMPLETED_ZERO_SHOT_EXTERNAL_TRANSFER";
      site: string;
      deployment: string;
      candidate_count: number;
      issued_rows: number;
      eligible_dates_per_horizon: number[];
      direct_daily_pinball_db: number;
      ema_daily_pinball_db: number;
      ema_relative_loss_reduction: number;
      paired_95pct_ema_minus_direct_db: number[];
      jepa_value_gate: "DESCRIPTIVE_ONLY_NOT_PRIMARY_GATE";
    };
    outcome_review_sha256: string;
    manifest_sha256: string;
  };
  scaling_development?: {
    study_id: "aeon3_geb_2024_hourly_sv_scale_30k_development_v1";
    status: "INDEPENDENTLY_REVIEWED_30K_STAGE1_NEGATIVE";
    classification: "POST_HOC_TRAIN_VALIDATION_DEVELOPMENT_NOT_FINAL_EVALUATION";
    assessment_partition: "validation";
    final_evaluation: false;
    source_time_basis: "SOURCE_REPORTED_UNSPECIFIED_NOT_UTC";
    validation_rows: number;
    eligible_rows_per_horizon: number[];
    eligible_dates_per_horizon: number[];
    eligible_date_sha256_by_horizon: string[];
    slots: Record<"direct_seed7" | "ema_jepa_seed7", {
      family: string;
      seed: 7;
      final_pinball_db: number;
      original_3k_pinball_db: number;
      relative_loss_change: number;
      supervised_updates: number;
      ssl_pretrain_updates: number;
      slot_sha256: string;
      prediction_sha256: string;
      endpoint_model_sha256: string;
    }>;
    stage_2: "NOT_RUN_BY_PREDECLARED_ONE_PERCENT_GATE";
    fifty_thousand_updates: "NOT_AUTHORIZED_BY_THIS_REVIEW";
    selection: "FIXED_FINAL_ENDPOINT_ONLY";
    outcome_review_sha256: string;
    manifest_sha256: string;
    config_sha256: string;
    protocol_sha256: string;
    cohort_sha256: string;
    provenance_sha256: Record<string, string>;
  };
  expanded_train_development?: {
    study_id: "aeon3_geb_expanded_train_3k_development_v1";
    status: "INDEPENDENTLY_REVIEWED_EXPANDED_TRAIN_3K_DEVELOPMENT";
    classification: "POST_HOC_TRAIN_VALIDATION_DEVELOPMENT_NOT_EXTERNAL_EVALUATION";
    assessment_partition: "original_validation";
    final_evaluation: false;
    source_time_basis: "SOURCE_REPORTED_UNSPECIFIED_NOT_UTC";
    joint_train_windows: number;
    prior_train_windows: number;
    current_train_windows: number;
    validation_rows: number;
    eligible_rows_per_horizon: number[];
    eligible_dates_per_horizon: number[];
    eligible_date_sha256_by_horizon: string[];
    sampling: "NATURAL_POOLED_SOURCE_PROPORTIONS";
    slots: Record<"direct_seed7" | "ema_jepa_seed7", {
      family: string;
      seed: 7;
      final_pinball_db: number;
      original_only_3k_pinball_db: number;
      relative_loss_reduction: number;
      supervised_updates: number;
      ssl_pretrain_updates: number;
      sampled_supervised_rows_by_source: { prior: number; current: number };
      slot_sha256: string;
      prediction_sha256: string;
      endpoint_model_sha256: string;
    }>;
    outcome_review_sha256: string;
    manifest_sha256: string;
    cohort_report_sha256: string;
    cohort_sha256: string;
    config_sha256: string;
    protocol_sha256: string;
    source_archive_sha256: { prior: string; current: string };
    provenance_sha256: Record<string, string>;
  };
  calibration?: {
    status: "INDEPENDENTLY_REVIEWED_CALIBRATION_ONLY";
    classification: "CALIBRATION_ONLY_NOT_MODEL_SELECTION_OR_FINAL_EVALUATION";
    issued_rows: number;
    eligible_days_per_horizon: number[];
    artifact_sha256: string;
    outcome_review_sha256: string;
    models: Record<string, {
      selection_classification: string;
      raw_primary_pinball_db: number;
      raw_per_horizon_pinball_db: number[];
      raw_coverage90_per_horizon: number[];
      interval_widening_db_by_horizon: number[];
      widened_in_sample_coverage90_per_horizon: number[];
    }>;
  };
  development: {
    core: {
      status: "INDEPENDENTLY_REVIEWED";
      issued_rows: number;
      eligible_days_per_horizon: number[];
      slot_primary_pinball_db: Record<string, number>;
    };
    hybrid: { status: "INDEPENDENTLY_REVIEWED"; primary_pinball_db: Record<string, number> };
    post_hoc_supervised: {
      family: "LightGBM";
      status: "INDEPENDENTLY_REVIEWED_POST_HOC_DEVELOPMENT";
      primary_pinball_db: number;
    };
    forward_ema: {
      status: "INDEPENDENTLY_REVIEWED_POST_HOC_DEVELOPMENT";
      primary_pinball_db: number;
      individual_seed_primary_pinball_db: Record<string, number>;
    };
    chronos2: {
      status: "INDEPENDENTLY_REVIEWED_POST_HOC_ZERO_SHOT_DEVELOPMENT";
      primary_pinball_db: number;
      model_revision: string;
    };
  };
  limitations: string[];
}

export async function getJson<T>(
  path: string,
  signal?: AbortSignal,
): Promise<T> {
  const response = await fetch(path, { signal, cache: "no-store" });
  if (!response.ok) {
    const error = await response
      .json()
      .catch(() => ({ detail: "Local service unavailable." }));
    throw new Error(error.detail || `Request failed (${response.status}).`);
  }
  return response.json() as Promise<T>;
}
