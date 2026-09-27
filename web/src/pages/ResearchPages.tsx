import type { ReactNode } from "react";
import {
  PageHeading,
  ReadoutGrid,
  ResearchStateNotice,
  SectionHeading,
  type ReadoutField,
  type ResearchState,
} from "../components/ResearchPrimitives";
import type { RawDevelopmentEvidence } from "../api/client";

export interface DeploymentReplayPageProps {
  state: ResearchState;
  stateMessage: string;
  plot?: ReactNode;
  metadata?: readonly ReadoutField[];
  cutoffLabel?: string;
  channelLabel?: string;
  rangeLabel?: string;
  orientationLabel?: string;
  modelLabel?: string;
  modeLabel?: string;
  uncertaintyLabel?: string;
  calibrationLabel?: string;
  canForecast?: boolean;
  onForecastHere?: () => void;
  canRevealObservedOutcome?: boolean;
  onRevealObservedOutcome?: () => void;
  outcomeRevealed?: boolean;
  revealedOutcome?: ReactNode;
}

export function DeploymentReplayPage({
  state,
  stateMessage,
  plot,
  metadata,
  cutoffLabel,
  channelLabel,
  rangeLabel,
  orientationLabel,
  modelLabel,
  modeLabel,
  uncertaintyLabel,
  calibrationLabel,
  canForecast = false,
  onForecastHere,
  canRevealObservedOutcome = false,
  onRevealObservedOutcome,
  outcomeRevealed = false,
  revealedOutcome,
}: DeploymentReplayPageProps) {
  const plotAvailable =
    (state === "success" || state === "partial") && plot != null;

  return (
    <div className="screen-stack">
      <PageHeading
        eyebrow="01 / PAST-ONLY INPUT"
        title="Replay a real acoustic window"
        description="Move the prediction cutoff through available observations. Playback changes the replay cutoff, never a buoy clock."
        trailing={
          <span className="mode-chip">
            <i aria-hidden="true" />
            {modeLabel || "Mode not supplied"}
          </span>
        }
      />

      <section className="panel replay-panel" aria-labelledby="replay-title">
        <SectionHeading
          id="replay-title"
          eyebrow="ACOUSTIC WINDOW"
          title="Deployment replay"
          description="Use the timestamp and instrument context supplied with the selected source."
          action={
            <span className="timestamp-chip">
              <span>Prediction cutoff</span>
              <strong>{cutoffLabel || "Not supplied"}</strong>
            </span>
          }
        />

        <div className="replay-settings" aria-label="Replay settings">
          <ReadoutGrid
            className="readout-grid--compact"
            fields={[
              { label: "Channel", value: channelLabel },
              { label: "Range", value: rangeLabel },
              { label: "Orientation", value: orientationLabel },
            ]}
          />
        </div>

        <div
          className="visualization-frame"
          role="region"
          aria-label="Acoustic replay"
        >
          {plotAvailable ? (
            <div className="visualization-frame__content">{plot}</div>
          ) : (
            <div className="visualization-placeholder">
              <div className="placeholder-ruler" aria-hidden="true">
                <i />
                <i />
                <i />
                <i />
              </div>
              <div className="placeholder-center">
                <span className="signal-glyph" aria-hidden="true">
                  <i />
                  <i />
                  <i />
                </span>
                <ResearchStateNotice
                  state={state}
                  title={
                    state === "success" ? "Replay view not supplied" : undefined
                  }
                  message={
                    state === "success"
                      ? "No visualization payload was provided."
                      : stateMessage
                  }
                />
              </div>
              <span className="placeholder-unit">
                Intensity unit follows verified source metadata
              </span>
            </div>
          )}
        </div>

        <div className="legend-row" aria-label="Replay legend">
          <span>
            <i
              className="legend-swatch legend-swatch--observed"
              aria-hidden="true"
            />
            Observed
          </span>
          <span>
            <i
              className="legend-swatch legend-swatch--forecast"
              aria-hidden="true"
            />
            Forecast
          </span>
          <span>
            <i
              className="legend-swatch legend-swatch--missing"
              aria-hidden="true"
            />
            Missing
          </span>
          <span className="legend-unit">
            Raw counts remain counts; dB requires verified calibration.
          </span>
        </div>

        <div className="replay-actions">
          <div>
            <button
              className="button button--primary"
              type="button"
              disabled={!canForecast || !onForecastHere}
              onClick={onForecastHere}
            >
              Forecast here <span aria-hidden="true">→</span>
            </button>
            <button
              className="button button--secondary"
              type="button"
              disabled={!canRevealObservedOutcome || !onRevealObservedOutcome}
              onClick={onRevealObservedOutcome}
            >
              Reveal observed outcome
            </button>
          </div>
          <p>
            Observed outcomes stay separate until the reveal action is enabled.
          </p>
        </div>

        {outcomeRevealed ? (
          <section
            className="revealed-panel"
            aria-label="Revealed observed outcome"
          >
            <SectionHeading
              eyebrow="REVEALED AFTER FORECAST"
              title="Observed outcome"
              description="Predictions and observations must share the same time and value scales."
            />
            {revealedOutcome ?? (
              <ResearchStateNotice
                state="empty"
                message="The reveal was requested, but no observed-outcome view was supplied."
              />
            )}
          </section>
        ) : null}
      </section>

      <section className="readout-section" aria-labelledby="replay-readouts">
        <SectionHeading
          id="replay-readouts"
          eyebrow="CONTEXT AT THE CUTOFF"
          title="Replay readouts"
          description="Values appear only when supplied by a verified replay or model artifact."
        />
        <ReadoutGrid
          fields={[
            {
              label: "Last available observation",
              value: metadata?.find(
                (item) => item.label === "Last available observation",
              )?.value,
            },
            {
              label: "Valid support",
              value: metadata?.find((item) => item.label === "Valid support")
                ?.value,
            },
            { label: "Calibration status", value: calibrationLabel },
            { label: "Selected model", value: modelLabel },
            { label: "Uncertainty", value: uncertaintyLabel },
            ...(metadata ?? []).filter(
              (item) =>
                !["Last available observation", "Valid support"].includes(
                  item.label,
                ),
            ),
          ]}
        />
      </section>
    </div>
  );
}

export interface ForecastComparisonPageProps {
  state: ResearchState;
  stateMessage: string;
  chart?: ReactNode;
  accessibleTable?: ReactNode;
  horizonRows?: readonly ForecastHorizonRow[];
  aggregationFormula?: string;
  strongestReferenceLabel?: string;
  candidateLabel?: string;
  profilePointForecast?: ReactNode;
  onOpenProtocol?: () => void;
}

export interface ForecastHorizonRow {
  horizon: "+1 h" | "+3 h" | "+6 h";
  referenceMedian?: string;
  referenceInterval90?: string;
  candidateMedian?: string;
  candidateInterval90?: string;
  unit?: string;
}

export function ForecastComparisonPage({
  state,
  stateMessage,
  chart,
  accessibleTable,
  horizonRows,
  aggregationFormula,
  strongestReferenceLabel,
  candidateLabel,
  profilePointForecast,
  onOpenProtocol,
}: ForecastComparisonPageProps) {
  const displayRows: readonly ForecastHorizonRow[] = horizonRows ?? [
    { horizon: "+1 h" },
    { horizon: "+3 h" },
    { horizon: "+6 h" },
  ];

  return (
    <div className="screen-stack">
      <PageHeading
        eyebrow="02 / PREDICTION EVIDENCE"
        title="Compare forecasts"
        description="Compare the candidate with the strongest supported reference at each declared horizon."
        trailing={
          <button
            className="text-button"
            type="button"
            onClick={onOpenProtocol}
            disabled={!onOpenProtocol}
          >
            Experiment protocol <span aria-hidden="true">↗</span>
          </button>
        }
      />

      <section className="panel" aria-labelledby="forecast-panel-title">
        <SectionHeading
          id="forecast-panel-title"
          eyebrow="ACOUSTIC-INDEX FORECAST"
          title="Median and empirical 90% interval"
          description="The interval and unit labels come from the frozen evaluation report."
          action={
            <span className="unit-chip">Acoustic-index units required</span>
          }
        />
        <div className="model-compare-strip" aria-label="Model comparison">
          <div>
            <span
              className="compare-key compare-key--reference"
              aria-hidden="true"
            />
            <span>Strongest reference</span>
            <strong>{strongestReferenceLabel || "Not selected"}</strong>
          </div>
          <div>
            <span
              className="compare-key compare-key--candidate"
              aria-hidden="true"
            />
            <span>JEPA candidate</span>
            <strong>{candidateLabel || "Not selected"}</strong>
          </div>
        </div>

        <div
          className="comparison-plot"
          role="region"
          aria-label="Forecast comparison chart"
        >
          {chart ?? (
            <div className="empty-chart">
              <div className="empty-chart__axis" aria-hidden="true">
                <i />
                <i />
                <i />
              </div>
              <ResearchStateNotice state={state} message={stateMessage} />
            </div>
          )}
        </div>
        {accessibleTable ? (
          <details className="accessible-data" open>
            <summary>Forecast values as a table</summary>
            <div className="table-scroll">{accessibleTable}</div>
          </details>
        ) : (
          <details className="accessible-data" open>
            <summary>Forecast values as a table</summary>
            <div className="table-scroll">
              <table className="forecast-table">
                <thead>
                  <tr>
                    <th scope="col">Horizon</th>
                    <th scope="col">Reference median</th>
                    <th scope="col">Reference empirical 90% interval</th>
                    <th scope="col">Candidate median</th>
                    <th scope="col">Candidate empirical 90% interval</th>
                    <th scope="col">Unit</th>
                  </tr>
                </thead>
                <tbody>
                  {displayRows.map((row) => (
                    <tr key={row.horizon}>
                      <th scope="row">{row.horizon}</th>
                      <td>{row.referenceMedian || "Not reported"}</td>
                      <td>{row.referenceInterval90 || "Not reported"}</td>
                      <td>{row.candidateMedian || "Not reported"}</td>
                      <td>{row.candidateInterval90 || "Not reported"}</td>
                      <td>{row.unit || "Not supplied"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        )}
        <p className="aggregation-note">
          Window aggregation formula:{" "}
          {aggregationFormula || "Not supplied with the evaluation artifact."}
        </p>
      </section>

      <section className="panel panel--subtle">
        <SectionHeading
          eyebrow="SEPARATE OUTPUT"
          title="Depth-profile point forecast"
          description="Point-only output; no distribution is implied."
        />
        {profilePointForecast ?? (
          <ResearchStateNotice
            state="not-executed"
            message="No valid depth-profile distribution or point forecast was supplied."
          />
        )}
      </section>
    </div>
  );
}

export interface FreshnessOption {
  value: string;
  label: string;
}

export interface ObservationFreshnessPageProps {
  state: ResearchState;
  stateMessage: string;
  options?: readonly FreshnessOption[];
  selectedAge?: string;
  onAgeChange?: (value: string) => void;
  scenarioLabel?: string;
  scenarioOutput?: ReactNode;
  qualityReasons?: readonly string[];
  suggestedRefreshRule?: string;
}

export function ObservationFreshnessPage({
  state,
  stateMessage,
  options,
  selectedAge,
  onAgeChange,
  scenarioLabel,
  scenarioOutput,
  qualityReasons,
  suggestedRefreshRule,
}: ObservationFreshnessPageProps) {
  return (
    <div className="screen-stack">
      <PageHeading
        eyebrow="03 / INPUT AVAILABILITY"
        title="Test observation freshness"
        description="Select an observation age or documented dropout scenario and inspect the replay response."
        trailing={
          <span className="simulation-badge">
            <span aria-hidden="true">↻</span>Replay simulation
          </span>
        }
      />

      <section className="panel freshness-panel">
        <div className="freshness-controls">
          <label className="field-label" htmlFor="freshness-scenario">
            Observation age / dropout scenario
          </label>
          <select
            id="freshness-scenario"
            value={selectedAge ?? ""}
            onChange={(event) => onAgeChange?.(event.currentTarget.value)}
            disabled={!options?.length || !onAgeChange}
          >
            <option value="">Select a documented scenario</option>
            {(options ?? []).map((option) => (
              <option value={option.value} key={option.value}>
                {option.label}
              </option>
            ))}
          </select>
          <p className="field-help">
            Scenarios must come from the experiment manifest; no new mask is
            generated here.
          </p>
        </div>

        <div className="freshness-summary">
          <div>
            <p className="eyebrow">SELECTED REPLAY</p>
            <h3>{scenarioLabel || "No scenario selected"}</h3>
            <p>
              Input availability, model response, uncertainty and quality
              reasons belong to the same replay record.
            </p>
          </div>
          <span className="freshness-stamp">
            SIMULATED
            <br />
            AVAILABILITY
          </span>
        </div>

        <div className="freshness-output">
          {scenarioOutput ?? (
            <ResearchStateNotice state={state} message={stateMessage} />
          )}
        </div>

        <section className="quality-panel" aria-labelledby="quality-title">
          <SectionHeading
            id="quality-title"
            eyebrow="WHY THIS OUTPUT CHANGED"
            title="Quality reasons"
            description="Reasons should be emitted by the replay and validation rule."
          />
          {qualityReasons?.length ? (
            <ul className="quality-list">
              {qualityReasons.map((reason, index) => (
                <li key={index}>{reason}</li>
              ))}
            </ul>
          ) : (
            <p className="muted-copy">No quality reasons supplied.</p>
          )}
        </section>

        <div className="refresh-rule">
          <div>
            <p className="eyebrow">SUGGESTED REFRESH</p>
            <p>
              {suggestedRefreshRule ||
                "No validation-fitted refresh rule supplied."}
            </p>
          </div>
          <span aria-hidden="true">⌁</span>
        </div>
      </section>

      <p className="boundary-note">
        This replay is not an optimized satellite schedule, a biological
        intervention or evidence of fuel savings.
      </p>
    </div>
  );
}

export interface ExperimentRow {
  id: string;
  family: string;
  seedCount?: string;
  primaryLoss?: string;
  mae?: string;
  coverage90?: string;
  intervalWidth?: string;
  sampleAndDayCounts?: string;
  trainingCost?: string;
  outcome?: string;
}

export type ExperimentReportScope = "validation" | "calibration" | "final-test";

export interface ExperimentLabPageProps {
  state: ResearchState;
  stateMessage: string;
  rows?: readonly ExperimentRow[];
  rawDevelopment?: RawDevelopmentEvidence | null;
  rawDevelopmentState?: "loading" | "success" | "not-executed";
  selectedScope?: ExperimentReportScope;
  onScopeChange?: (scope: ExperimentReportScope) => void;
  pairedDifferenceChart?: ReactNode;
  confidenceIntervalMethod?: string;
  protocolHref?: string;
}

export function ExperimentLabPage({
  state,
  stateMessage,
  rows,
  rawDevelopment,
  rawDevelopmentState = "not-executed",
  selectedScope,
  onScopeChange,
  pairedDifferenceChart,
  confidenceIntervalMethod,
  protocolHref,
}: ExperimentLabPageProps) {
  return (
    <div className="screen-stack">
      <PageHeading
        eyebrow="04 / RUN REGISTRY"
        title="Inspect the experiment lab"
        description="A missing run stays not executed. A scientific rejection is a result, not a software error."
        trailing={
          protocolHref ? (
            <a className="text-button" href={protocolHref}>
              Frozen protocol <span aria-hidden="true">↗</span>
            </a>
          ) : (
            <span className="unit-chip">Protocol link not supplied</span>
          )
        }
      />

      <section className="panel experiment-panel">
        <div className="experiment-toolbar">
          <div>
            <p className="eyebrow">REPORT VIEW</p>
            <h3>Run outcomes</h3>
          </div>
          <label className="scope-filter">
            <span>Report split</span>
            <select
              value={selectedScope ?? "validation"}
              onChange={(event) =>
                onScopeChange?.(
                  event.currentTarget.value as ExperimentReportScope,
                )
              }
              disabled={!onScopeChange}
            >
              <option value="validation">Validation</option>
              <option value="calibration">Calibration</option>
              <option value="final-test">Final test</option>
            </select>
          </label>
        </div>

        <ResearchStateNotice state={state} message={stateMessage} />
        <div
          className="experiment-table-scroll"
          tabIndex={0}
          aria-label="Scrollable experiment results table"
        >
          <table className="experiment-table">
            <thead>
              <tr>
                <th scope="col">Family</th>
                <th scope="col">Seeds</th>
                <th scope="col">Primary loss</th>
                <th scope="col">MAE</th>
                <th scope="col">Coverage 90%</th>
                <th scope="col">Interval width</th>
                <th scope="col">Samples / days</th>
                <th scope="col">Training cost</th>
                <th scope="col">Outcome</th>
              </tr>
            </thead>
            <tbody>
              {rows?.length ? (
                rows.map((row) => (
                  <tr key={row.id}>
                    <th scope="row">{row.family}</th>
                    <td>{row.seedCount || "Not reported"}</td>
                    <td>{row.primaryLoss || "Not reported"}</td>
                    <td>{row.mae || "Not reported"}</td>
                    <td>{row.coverage90 || "Not reported"}</td>
                    <td>{row.intervalWidth || "Not reported"}</td>
                    <td>{row.sampleAndDayCounts || "Not reported"}</td>
                    <td>{row.trainingCost || "Not reported"}</td>
                    <td>{row.outcome || "Not reported"}</td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td className="empty-table-cell" colSpan={9}>
                    No run rows supplied for this report split.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>

      <section
        className="panel raw-development-panel"
        aria-labelledby="raw-development-title"
      >
        <SectionHeading
          eyebrow="REAL TRAIN DEVELOPMENT · ENGINEERING ONLY"
          title="Raw response-code TRAIN development"
          description="A complete-positive transformed AZFP backscatter_r code mean study. These development metrics use raw response codes, not calibrated Sv. This is not a JEPA result, and no final evaluation was run."
          id="raw-development-title"
        />
        {rawDevelopment ? (
          <RawDevelopmentResult evidence={rawDevelopment} />
        ) : (
          <ResearchStateNotice
            state={rawDevelopmentState}
            title={
              rawDevelopmentState === "loading"
                ? "Loading raw development evidence"
                : "Raw development evidence unavailable"
            }
            message={
              rawDevelopmentState === "loading"
                ? "Reading the local TRAIN-development evidence artifact."
                : "The local service did not provide the raw TRAIN-development artifact. No prediction or performance result is displayed."
            }
          />
        )}
      </section>

      <section className="panel panel--subtle">
        <SectionHeading
          eyebrow="PAIRED COMPARISON"
          title="Difference with uncertainty"
          description={
            "Confidence interval method: " +
            (confidenceIntervalMethod || "Not supplied")
          }
        />
        {pairedDifferenceChart ?? (
          <ResearchStateNotice
            state="not-executed"
            message="No paired-difference report was supplied."
          />
        )}
      </section>
    </div>
  );
}

function RawDevelopmentResult({
  evidence,
}: {
  evidence: RawDevelopmentEvidence;
}) {
  const predictionRows = evidence.rows.flatMap((row) =>
    row.horizons.map((horizon) => ({
      cutoff_utc: row.cutoff_utc,
      ...horizon,
    })),
  );
  const directWorseAtEveryHorizon =
    evidence.horizons.length === 3 &&
    evidence.horizons.every(
      (horizon) =>
        horizon.direct_neural.daily_mean_pinball_code >
          horizon.ridge.daily_mean_pinball_code &&
        horizon.direct_neural.mae_code_median > horizon.ridge.mae_code_median,
    );

  return (
    <div className="raw-development-content">
      <ResearchStateNotice
        state="partial"
        title="TRAIN-development artifact · engineering only"
        message={`${evidence.status} · ${evidence.study_id} · ${evidence.run_id}. The reported comparison is ${evidence.comparison_label}.`}
      />
      <dl className="raw-development-facts">
        <div>
          <dt>Quantity</dt>
          <dd>{evidence.quantity}</dd>
        </div>
        <div>
          <dt>Unit</dt>
          <dd>{evidence.unit}</dd>
        </div>
        <div>
          <dt>Calibration</dt>
          <dd>
            {evidence.calibrated ? "Calibrated" : "Uncalibrated · not Sv"}
          </dd>
        </div>
        <div>
          <dt>Final evaluation</dt>
          <dd>{evidence.final_evaluation ? "Performed" : "Not performed"}</dd>
        </div>
      </dl>
      <p className="raw-development-conclusion">
        {directWorseAtEveryHorizon
          ? "Direct neural is worse than ridge on both reported metrics at all three horizons."
          : evidence.comparison_label}
      </p>

      <div
        className="raw-development-metrics-scroll"
        role="region"
        tabIndex={0}
        aria-label="Scrollable three-horizon development metrics"
      >
        <table className="raw-development-metrics">
          <caption>
            Recorded TRAIN-development metrics by forecast horizon
          </caption>
          <thead>
            <tr>
              <th scope="col" rowSpan={2}>
                Horizon
              </th>
              <th scope="col" rowSpan={2}>
                Eligible rows
              </th>
              <th scope="col" rowSpan={2}>
                Target days
              </th>
              <th scope="colgroup" colSpan={2}>
                Ridge
              </th>
              <th scope="colgroup" colSpan={2}>
                Direct neural
              </th>
            </tr>
            <tr>
              <th scope="col">Daily mean pinball (code)</th>
              <th scope="col">Median MAE (code)</th>
              <th scope="col">Daily mean pinball (code)</th>
              <th scope="col">Median MAE (code)</th>
            </tr>
          </thead>
          <tbody>
            {evidence.horizons.map((horizon) => (
              <tr key={horizon.horizon_hours}>
                <th scope="row">{horizon.horizon_hours} h</th>
                <td>{horizon.eligible_rows}</td>
                <td>{horizon.target_days}</td>
                <td>{String(horizon.ridge.daily_mean_pinball_code)}</td>
                <td>{String(horizon.ridge.mae_code_median)}</td>
                <td>{String(horizon.direct_neural.daily_mean_pinball_code)}</td>
                <td>{String(horizon.direct_neural.mae_code_median)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div
        className="raw-development-predictions-scroll"
        role="region"
        tabIndex={0}
        aria-label="All raw TRAIN-development cutoff and horizon predictions"
      >
        <table className="raw-development-predictions">
          <caption>
            All TRAIN-development predictions and truths ·{" "}
            {evidence.rows.length} cutoffs · {predictionRows.length}{" "}
            cutoff-by-horizon rows · no rows omitted
          </caption>
          <thead>
            <tr>
              <th scope="col">Cutoff (UTC)</th>
              <th scope="col">Horizon</th>
              <th scope="col">Target start (UTC)</th>
              <th scope="col">Eligible</th>
              <th scope="col">Truth code</th>
              <th scope="col">Ridge quantile codes (source order)</th>
              <th scope="col">Direct neural quantile codes (source order)</th>
            </tr>
          </thead>
          <tbody>
            {predictionRows.map((row) => (
              <tr key={`${row.cutoff_utc}-${row.horizon_hours}`}>
                <th scope="row">{row.cutoff_utc}</th>
                <td>{row.horizon_hours} h</td>
                <td>{row.target_start_utc}</td>
                <td>{row.eligible ? "Eligible" : "Ineligible"}</td>
                <td>
                  {row.truth_code === null
                    ? "Not scored"
                    : String(row.truth_code)}
                </td>
                <td>
                  <code>{row.ridge_quantiles_code.map(String).join(", ")}</code>
                </td>
                <td>
                  <code>
                    {row.direct_quantiles_code.map(String).join(", ")}
                  </code>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="raw-development-limitations">
        <h4>Study limitations</h4>
        <ul>
          {evidence.limitations.map((limitation, index) => (
            <li key={`${index}-${limitation}`}>{limitation}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}

export interface EvidenceItem {
  label: string;
  value?: string;
  detail?: string;
}

export interface TransferField {
  field: string;
  purpose?: string;
  status?: string;
}

export interface EvidenceTransferPageProps {
  evidence?: readonly EvidenceItem[];
  transferFields?: readonly TransferField[];
  reportView?: ReactNode;
  testSealLabel?: string;
  onExportCsv?: () => void;
  onExportJson?: () => void;
  onOpenReport?: () => void;
  csvAvailable?: boolean;
  jsonAvailable?: boolean;
  reportAvailable?: boolean;
}

export function EvidenceTransferPage({
  evidence,
  transferFields,
  reportView,
  testSealLabel,
  onExportCsv,
  onExportJson,
  onOpenReport,
  csvAvailable = false,
  jsonAvailable = false,
  reportAvailable = false,
}: EvidenceTransferPageProps) {
  const evidenceItems: readonly EvidenceItem[] = evidence ?? [
    { label: "Data provenance" },
    { label: "SHA-256 checksums" },
    { label: "License" },
    { label: "Model card" },
    { label: "Calibration assumptions" },
    { label: "Protected-test seal", value: testSealLabel },
  ];

  return (
    <div className="screen-stack">
      <PageHeading
        eyebrow="05 / SOURCE AND SCOPE"
        title="Review evidence and transfer"
        description="Trace every displayed result to its source, calibration assumptions, immutable artifact and evaluation split."
      />

      <div className="evidence-grid">
        <section className="panel evidence-panel">
          <SectionHeading
            eyebrow="PROVENANCE"
            title="Evidence record"
            description="Evidence fields are populated only from recorded source and run artifacts."
          />
          <dl className="evidence-list">
            {evidenceItems.map((item) => (
              <div className="evidence-item" key={item.label}>
                <dt>{item.label}</dt>
                <dd>{item.value || "Not supplied"}</dd>
                {item.detail ? <p>{item.detail}</p> : null}
              </div>
            ))}
          </dl>
          <div className="export-actions">
            <button
              className="button button--secondary"
              type="button"
              onClick={onExportCsv}
              disabled={!csvAvailable || !onExportCsv}
            >
              Export CSV
            </button>
            <button
              className="button button--secondary"
              type="button"
              onClick={onExportJson}
              disabled={!jsonAvailable || !onExportJson}
            >
              Export JSON
            </button>
            <button
              className="button button--quiet"
              type="button"
              onClick={onOpenReport}
              disabled={!reportAvailable || !onOpenReport}
            >
              Local report <span aria-hidden="true">↗</span>
            </button>
          </div>
        </section>

        <section className="panel transfer-panel">
          <SectionHeading
            eyebrow="FUTURE TRANSFER"
            title="Proposed export fields"
            description="A public acoustic prototype and a future customer pilot have different evidence requirements."
          />
          {transferFields?.length ? (
            <dl className="transfer-list">
              {transferFields.map((item) => (
                <div className="transfer-item" key={item.field}>
                  <dt>{item.field}</dt>
                  <dd>{item.purpose || "Purpose not supplied"}</dd>
                  <span>{item.status || "Proposed"}</span>
                </div>
              ))}
            </dl>
          ) : (
            <ResearchStateNotice
              state="not-executed"
              message="No customer schema or approved export mapping is supplied. Marine production integration is not established."
            />
          )}
          <p className="transfer-boundary">
            Public-data results do not establish tuna biomass, species, catch,
            fuel savings or Marine production integration.
          </p>
        </section>
      </div>

      {reportView ? (
        <section className="panel panel--subtle">
          <SectionHeading eyebrow="LOCAL ARTIFACT" title="Report preview" />
          {reportView}
        </section>
      ) : null}
    </div>
  );
}
