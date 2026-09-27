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
