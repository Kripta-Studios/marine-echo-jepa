import { useEffect, useState } from "react";
import { PageHeading, ResearchStateNotice, SectionHeading } from "../components/ResearchPrimitives";
import { getJson, type AeonStudyEvidence } from "../api/client";

const testModels = [
  ["Direct ensemble", "core_direct_equal_three_seed_ensemble"],
  ["EMA-JEPA ensemble", "core_ema_equal_three_seed_ensemble"],
  ["LightGBM · post-hoc", "post_hoc_lightgbm"],
] as const;

type ReplayPage = {
  classification: string;
  source_time_basis: string;
  target: string;
  horizon_source_interval_steps: number[];
  quantile_levels: number[];
  total: number;
  offset: number;
  rows: Array<{
    row_id: string;
    cutoff_interval_id: number;
    cutoff_source_timestamp: string;
    quantiles_db: number[][];
  }>;
};

function AeonReplay() {
  const [model, setModel] = useState<string>(testModels[0][1]);
  const [offset, setOffset] = useState(0);
  const [page, setPage] = useState<ReplayPage | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    setPage(null);
    setError(null);
    getJson<ReplayPage>(
      `/api/v1/studies/aeon/replay?model_id=${encodeURIComponent(model)}&offset=${offset}&limit=12`,
      controller.signal,
    ).then(setPage).catch((reason: unknown) => {
      if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : "Replay unavailable.");
    });
    return () => controller.abort();
  }, [model, offset]);
  return (
    <section className="panel" aria-labelledby="aeon-replay-title">
      <SectionHeading id="aeon-replay-title" eyebrow="HISTORICAL REPLAY" title="Saved TEST predictions"
        description="Every issued cutoff is available in chronological source order. These are saved, truth-free predictions from the reviewed retrospective run, not live forecasts. The source clock is unspecified, not UTC." />
      <label htmlFor="aeon-replay-model">Frozen family</label>{" "}
      <select id="aeon-replay-model" value={model} onChange={(event) => { setModel(event.target.value); setOffset(0); }}>
        {testModels.map(([name, id]) => <option key={id} value={id}>{name}</option>)}
      </select>
      {error && <p role="alert">{error}</p>}
      {!page && !error && <p>Loading saved predictions…</p>}
      {page && <>
        <p>Rows {page.offset + 1}–{Math.min(page.offset + page.rows.length, page.total)} of {page.total}.</p>
        <div className="table-scroll"><table>
          <thead><tr><th scope="col">Source cutoff</th><th scope="col">Interval ID</th><th scope="col">+1 source interval · q05 / q25 / q50 / q75 / q95 (dB)</th><th scope="col">+3</th><th scope="col">+6</th></tr></thead>
          <tbody>{page.rows.map((row) => <tr key={row.row_id}>
            <th scope="row">{row.cutoff_source_timestamp}</th><td>{row.cutoff_interval_id}</td>
            {row.quantiles_db.map((values, horizon) => <td key={horizon}>{values.map((value) => value.toFixed(2)).join(" / ")}</td>)}
          </tr>)}</tbody>
        </table></div>
        <div className="button-row">
          <button type="button" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - 12))}>Previous</button>{" "}
          <button type="button" disabled={offset + 12 >= page.total} onClick={() => setOffset(offset + 12)}>Next</button>
        </div>
      </>}
    </section>
  );
}

export function AeonStudyPage({
  study,
  state,
}: {
  study: AeonStudyEvidence | null;
  state: "loading" | "success" | "unavailable";
}) {
  if (!study) {
    return (
      <ResearchStateNotice
        state={state === "loading" ? "loading" : "not-executed"}
        title="AEON study artifact unavailable"
        message="This offline package has no reviewed AEON study artifact. Historical MOSAiC evidence remains separate."
      />
    );
  }
  const core = study.development.core.slot_primary_pinball_db;
  const hybrid = study.development.hybrid.primary_pinball_db;
  const rows = [
    ["Raw-only histogram gradient boosting", core.hist_gradient_boosting, "Core conventional"],
    ["Direct neural · equal three-seed ensemble", hybrid.core_direct_equal_three_seed_ensemble, "Core conventional"],
    ["EMA-JEPA · equal three-seed ensemble", hybrid.core_ema_equal_three_seed_ensemble, "Core JEPA"],
    ["Shared-SIGReg · equal three-seed ensemble", hybrid.core_shared_equal_three_seed_ensemble, "Core JEPA"],
    ["EMA raw + latent hybrid ensemble", hybrid.ema_hybrid_equal_three_seed_ensemble, "Hybrid development"],
    ["Shared raw + latent hybrid ensemble", hybrid.shared_hybrid_equal_three_seed_ensemble, "Hybrid development"],
    ["LightGBM", study.development.post_hoc_supervised.primary_pinball_db, "Post-hoc development"],
    ["Forward EMA-JEPA · equal three-seed ensemble", study.development.forward_ema.primary_pinball_db, "Post-hoc development"],
    ["Chronos-2 · frozen zero-shot", study.development.chronos2.primary_pinball_db, "Post-hoc development"],
  ] as const;
  return (
    <div className="screen-stack">
      <PageHeading
        eyebrow="06 / SEPARATE AEON STUDY"
        title={study.title}
        description="Reviewed TRAIN/validation development of source-reported conditioned 38-kHz hourly Sv_mean. This is a separate Georges Basin study, not the MOSAiC raw-count replay."
      />
      <section className="panel" aria-labelledby="aeon-source-title">
        <SectionHeading
          id="aeon-source-title"
          eyebrow="SOURCE AND TARGET"
          title="Published hourly acoustic product"
          description={`${study.source.publisher}. The source clock is unspecified, not UTC; forecast horizons are scheduled source interval steps.`}
        />
        <p>38 kHz · {study.target.product} · nominal 0–200 m · {study.target.unit}</p>
        <p>Calibration: source-reported conditioned Sv; exact per-unit settings are not independently field verified.</p>
        <p>Source archive SHA-256: <code>{study.source.archive_sha256}</code></p>
      </section>
      <section className="panel" aria-labelledby="aeon-metrics-title">
        <SectionHeading
          id="aeon-metrics-title"
          eyebrow="VALIDATION ONLY"
          title="Reviewed development comparison"
          description={`${study.development.core.issued_rows} issued validation rows; ${study.development.core.eligible_days_per_horizon.join(" / ")} eligible source dates across the 1 / 3 / 6-step horizons. Lower daily mean pinball loss is better. No model is selected in this report.`}
        />
        <div className="table-scroll">
          <table>
            <thead><tr><th scope="col">Model</th><th scope="col">Daily pinball (dB)</th><th scope="col">Evidence class</th></tr></thead>
            <tbody>
              {rows.map(([label, value, classification]) => (
                <tr key={label}><th scope="row">{label}</th><td>{typeof value === "number" && Number.isFinite(value) ? value.toFixed(4) : "Not supplied"}</td><td>{classification}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
        <p>Forward EMA and Chronos-2 have independent outcome reviews. Their scores are post-hoc development evidence; neither establishes state of the art or JEPA incremental value.</p>
        <p>{study.retrospective_test ? "Reviewed retrospective TEST evidence is shown separately below." : study.calibration ? "CAL is shown separately below; retrospective TEST remains unopened." : "CAL and retrospective TEST outcomes are not opened for this report."} {study.retrospective_test ? "Saved source-clock TEST predictions are available as historical replay." : "No AEON cached forecasts are served."}</p>
      </section>
      {study.calibration && (
        <section className="panel" aria-labelledby="aeon-calibration-title">
          <SectionHeading
            id="aeon-calibration-title"
            eyebrow="CALIBRATION ONLY"
            title="Reviewed interval widening"
            description={`${study.calibration.issued_rows} issued CAL rows; ${study.calibration.eligible_days_per_horizon.join(" / ")} eligible source dates by horizon. These in-sample CAL diagnostics do not select or rank models and are not final evaluation.`}
          />
          <div className="table-scroll">
            <table>
              <thead><tr><th scope="col">Frozen family</th><th scope="col">Raw CAL pinball (dB)</th><th scope="col">90% interval widening, +1 / +3 / +6 (dB)</th><th scope="col">Raw 90% coverage, +1 / +3 / +6</th><th scope="col">Widened in-sample 90% coverage, +1 / +3 / +6</th></tr></thead>
              <tbody>
                {([
                  ["Direct ensemble", "core_direct_equal_three_seed_ensemble"],
                  ["EMA-JEPA ensemble", "core_ema_equal_three_seed_ensemble"],
                  ["LightGBM · post-hoc", "post_hoc_lightgbm"],
                ] as const).map(([label, key]) => {
                  const model = study.calibration!.models[key];
                  return <tr key={key}>
                    <th scope="row">{label}</th>
                    <td>{model.raw_primary_pinball_db.toFixed(4)}</td>
                    <td>{model.interval_widening_db_by_horizon.map((value) => value.toFixed(3)).join(" / ")}</td>
                    <td>{model.raw_coverage90_per_horizon.map((value) => `${(value * 100).toFixed(1)}%`).join(" / ")}</td>
                    <td>{model.widened_in_sample_coverage90_per_horizon.map((value) => `${(value * 100).toFixed(1)}%`).join(" / ")}</td>
                  </tr>;
                })}
              </tbody>
            </table>
          </div>
          <p>Widening changed only the outer interval endpoints on CAL. Approximately 90% in-sample CAL coverage does not guarantee retrospective TEST coverage.</p>
          <p>CAL artifact SHA-256: <code>{study.calibration.artifact_sha256}</code></p>
        </section>
      )}
      {study.retrospective_test && <>
        <section className="panel" aria-labelledby="aeon-test-title">
          <SectionHeading id="aeon-test-title" eyebrow="REVIEWED RETROSPECTIVE TEST" title="Frozen-family evaluation"
            description={`${study.retrospective_test.issued_rows} issued TEST rows; ${study.retrospective_test.eligible_days_per_horizon.join(" / ")} eligible source dates at +1 / +3 / +6 source intervals. The split was retrospective and is not sealed or an external replication.`} />
          <div className="table-scroll"><table>
            <thead><tr><th scope="col">Frozen family</th><th scope="col">Raw daily pinball (dB)</th><th scope="col">+1 / +3 / +6 pinball (dB)</th><th scope="col">Raw 90% coverage</th><th scope="col">CAL-widened 90% coverage on TEST</th></tr></thead>
            <tbody>{testModels.map(([name, id]) => {
              const result = study.retrospective_test!.models[id];
              return <tr key={id}><th scope="row">{name}</th><td>{result.raw_primary_daily_mean_pinball_db.toFixed(4)}</td>
                <td>{result.raw_per_horizon_pinball_db.map((value) => value.toFixed(4)).join(" / ")}</td>
                <td>{result.raw_coverage90_per_horizon.map((value) => `${(value * 100).toFixed(1)}%`).join(" / ")}</td>
                <td>{result.widened_coverage90_per_horizon.map((value) => `${(value * 100).toFixed(1)}%`).join(" / ")}</td></tr>;
            })}</tbody>
          </table></div>
          <p>The frozen EMA-JEPA ensemble passed the prespecified within-study retrospective gate against direct. This does not isolate the gain to learned JEPA representations, establish a sealed holdout, or show external generalization. Post-hoc LightGBM failed the same five-percent point gate.</p>
          <p>EMA relative raw loss change versus direct: {(study.retrospective_test.comparisons.core_ema_equal_three_seed_ensemble.primary_relative_loss_change! * 100).toFixed(2)}%; paired 95% interval {study.retrospective_test.comparisons.core_ema_equal_three_seed_ensemble.paired_95_percent_interval_db.map((value) => value.toFixed(4)).join(" to ")} dB. CAL widened intervals without selecting a model. No species, biomass or operational claim follows from these acoustic scores.</p>
          <p>TEST score SHA-256: <code>{study.retrospective_test.test_score_sha256}</code>. Independent outcome review SHA-256: <code>{study.retrospective_test.test_outcome_review_sha256}</code>.</p>
        </section>
        <AeonReplay />
      </>}
      <section className="panel" aria-label="Study limitations">
        <h2>Limits</h2>
        <ul>{study.limitations.map((limit) => <li key={limit}>{limit}</li>)}</ul>
      </section>
    </div>
  );
}
