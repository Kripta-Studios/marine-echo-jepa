import { PageHeading, ResearchStateNotice, SectionHeading } from "../components/ResearchPrimitives";
import type { AeonStudyEvidence } from "../api/client";

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
        <p>CAL and retrospective TEST outcomes are not opened for this report. No AEON cached forecasts are served.</p>
      </section>
      <section className="panel" aria-label="Study limitations">
        <h2>Limits</h2>
        <ul>{study.limitations.map((limit) => <li key={limit}>{limit}</li>)}</ul>
      </section>
    </div>
  );
}
