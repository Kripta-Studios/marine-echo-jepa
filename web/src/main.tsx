import { StrictMode, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { AppShell, type AppRoute, type AppScreen } from "./components/AppShell";
import { ResearchStateNotice } from "./components/ResearchPrimitives";
import {
  DeploymentReplayPage,
  ForecastComparisonPage,
  ObservationFreshnessPage,
  ExperimentLabPage,
  EvidenceTransferPage,
  type ExperimentReportScope,
} from "./pages/ResearchPages";
import {
  forecastRequest,
  getJson,
  type Dataset,
  type Observations,
  type Model,
  type Evidence,
} from "./api/client";
import { RawEchogram } from "./charts/RawEchogram";
import { safeCutoff } from "./api/cutoff";
import "./styles/integration.css";

const blocked =
  "Physical-unit forecasts are unavailable. Environmental profiles have been found, but calibration applicability and the manual/archive instrument serial discrepancy remain unresolved. Required model experiments have not run.";

function App() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [models, setModels] = useState<Model[]>([]);
  const [evidence, setEvidence] = useState<Evidence | null>(null);
  const [error, setError] = useState("");
  const [past, setPast] = useState<Observations | null>(null);
  const [future, setFuture] = useState<Observations | null>(null);
  const [channel, setChannel] = useState(0);
  const [cutoff, setCutoff] = useState(
    safeCutoff(new URLSearchParams(location.search).get("cutoff")),
  );
  const [age, setAge] = useState("0");
  const [forecastStatus, setForecastStatus] = useState("");
  const [scope, setScope] = useState<ExperimentReportScope>("validation");
  const [experiments, setExperiments] = useState<unknown[]>([]);
  const showError = (e: Error) => {
    if (e.name !== "AbortError") setError(e.message);
  };
  const dataset = datasets[0];
  const pending = useRef(new AbortController());
  useEffect(() => {
    const controller = new AbortController();
    Promise.all([
      getJson<Dataset[]>("/api/v1/datasets", controller.signal),
      getJson<Model[]>("/api/v1/models", controller.signal),
      getJson<Evidence>("/api/v1/evidence/research", controller.signal),
      getJson<unknown[]>("/api/v1/experiments", controller.signal),
    ])
      .then(([d, m, e, runs]) => {
        setDatasets(d);
        setModels(m);
        setEvidence(e);
        setExperiments(runs);
        setError("");
      })
      .catch((e) => {
        if (e.name !== "AbortError") setError(e.message);
      });
    return () => controller.abort();
  }, []);
  useEffect(() => {
    const sync = () =>
      setCutoff(safeCutoff(new URLSearchParams(location.search).get("cutoff")));
    addEventListener("popstate", sync);
    return () => removeEventListener("popstate", sync);
  }, []);
  useEffect(() => {
    if (!dataset) return;
    const controller = new AbortController();
    setPast(null);
    setError("");
    pending.current.abort();
    pending.current = new AbortController();
    setFuture(null);
    setForecastStatus("");
    const available = new Date(
      Date.parse(cutoff) - Number(age) * 3600000,
    ).toISOString();
    const query = new URLSearchParams({
      start: dataset.start,
      end: dataset.end,
      cutoff: available,
    });
    getJson<Observations>(
      `/api/v1/datasets/${dataset.id}/observations?${query}`,
      controller.signal,
    )
      .then((result) => {
        setPast(result);
        setError("");
      })
      .catch((e) => {
        if (e.name !== "AbortError") setError(e.message);
      });
    return () => controller.abort();
  }, [dataset, cutoff, age]);
  async function forecast() {
    if (!dataset) return;
    const signal = pending.current.signal;
    const response = await fetch("/api/v1/forecast", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal,
      body: JSON.stringify(
        forecastRequest({
          datasetId: dataset.id,
          modelId: "direct",
          cutoff,
          age: Number(age),
        }),
      ),
    });
    const body = await response.json();
    if (!signal.aborted)
      setForecastStatus(
        response.ok ? "Verified cached forecast received." : body.detail,
      );
  }
  async function reveal() {
    if (!dataset) return;
    const signal = pending.current.signal;
    const query = new URLSearchParams({
      start: new Date(Date.parse(cutoff) + 1).toISOString(),
      end: dataset.end,
    });
    const result = await getJson<Observations>(
      `/api/v1/datasets/${dataset.id}/observations?${query}`,
      signal,
    );
    if (!signal.aborted) setFuture(result);
  }
  const navigateEvidence = () => {
    location.hash = "evidence-transfer";
  };
  const exportJson = () => {
    location.href = "/api/v1/exports/research-export";
  };
  const controls = (
    <div className="replay-controls">
      <label>
        Prediction cutoff (UTC)
        <select
          value={cutoff}
          onChange={(event) => {
            setCutoff(event.target.value);
            history.pushState(
              null,
              "",
              `?cutoff=${encodeURIComponent(event.target.value)}${location.hash}`,
            );
          }}
        >
          {[6, 12, 18].map((hour) => {
            const value = `2020-02-17T${String(hour).padStart(2, "0")}:00:00Z`;
            return (
              <option key={hour} value={value}>
                {value}
              </option>
            );
          })}
        </select>
      </label>
      <label>
        Observed frequency
        <select
          value={channel}
          onChange={(event) => setChannel(Number(event.target.value))}
        >
          {[38000, 125000, 200000, 455000].map((hz, index) => (
            <option key={hz} value={index}>
              {hz / 1000} kHz
            </option>
          ))}
        </select>
      </label>
    </div>
  );
  const plot = past ? (
    <>
      {controls}
      <RawEchogram
        rows={past.rows}
        channel={channel}
        label="Observed before cutoff"
      />
      <div className="cutoff-mark">Prediction cutoff → {cutoff}</div>
    </>
  ) : undefined;
  const screens: Record<AppRoute, AppScreen> = {
    "deployment-replay": {
      title: "Deployment replay",
      content: (
        <>
          {error && <ResearchStateNotice state="error" message={error} />}
          <DeploymentReplayPage
            state={
              past
                ? past.rows.length
                  ? "partial"
                  : "empty"
                : error
                  ? "error"
                  : "loading"
            }
            stateMessage={
              error || "Reading verified local acoustic observations."
            }
            plot={plot}
            cutoffLabel={cutoff}
            channelLabel={`${[38, 125, 200, 455][channel]} kHz`}
            rangeLabel="Sample-index groups · not metres"
            orientationLabel="Downward · Arctic ice-tethered"
            modeLabel="Raw-count diagnostic replay"
            calibrationLabel="Blocked · raw counts only"
            modelLabel="None promoted"
            uncertaintyLabel="Not evaluated"
            metadata={[
              {
                label: "Last available observation",
                value: past?.rows.at(-1)?.event_time_utc,
              },
              {
                label: "Valid support",
                value: "Physical support not established",
              },
              { label: "Source", value: "PANGAEA 949811 · CC BY 4.0" },
            ]}
            canForecast={!!dataset}
            onForecastHere={() => {
              void forecast().catch(showError);
            }}
            canRevealObservedOutcome={!!past}
            onRevealObservedOutcome={() => {
              void reveal().catch(showError);
            }}
            outcomeRevealed={future !== null}
            revealedOutcome={
              future && (
                <RawEchogram
                  rows={future.rows}
                  channel={channel}
                  label="Revealed later observations · not model input"
                />
              )
            }
          />
          {forecastStatus && (
            <ResearchStateNotice
              state="unsupported"
              title="Forecast unavailable"
              message={forecastStatus}
            />
          )}
        </>
      ),
    },
    "forecast-comparison": {
      title: "Forecast comparison",
      content: (
        <ForecastComparisonPage
          state="blocked"
          stateMessage={blocked}
          onOpenProtocol={navigateEvidence}
          profilePointForecast={
            <ResearchStateNotice
              state="not-executed"
              message="No calibrated profile predictor has been trained. Raw observations are available in Deployment replay."
            />
          }
        />
      ),
    },
    "observation-freshness": {
      title: "Observation freshness",
      content: (
        <ObservationFreshnessPage
          state="partial"
          stateMessage="This replay removes recent observations. Forecast-error and interval-width effects are not evaluated."
          options={[0, 1, 3, 6].map((h) => ({
            value: String(h),
            label: `${h} hours old`,
          }))}
          selectedAge={age}
          onAgeChange={setAge}
          scenarioLabel="Predefined age mask · replay simulation"
          scenarioOutput={plot}
          qualityReasons={[blocked]}
          suggestedRefreshRule="Demonstration heuristic: request a refresh after three hours. No measured operational benefit."
        />
      ),
    },
    "experiment-lab": {
      title: "Experiment lab",
      content: (
        <>
          <ExperimentLabPage
            state="not-executed"
            stateMessage={blocked}
            selectedScope={scope}
            onScopeChange={setScope}
            rows={models.map((m) => ({
              id: m.id,
              family: m.family,
              seedCount: "0 completed",
              outcome: m.status,
              trainingCost: "No benchmark training",
            }))}
            confidenceIntervalMethod="Planned: paired 48-hour calendar blocks, 2,000 draws. No inferential result exists."
            protocolHref="/api/v1/evidence/research"
          />
          <details>
            <summary>Required run registry ({experiments.length} runs)</summary>
            <pre>{JSON.stringify(experiments, null, 2)}</pre>
          </details>
        </>
      ),
    },
    "evidence-transfer": {
      title: "Evidence & transfer",
      content: (
        <EvidenceTransferPage
          evidence={[
            {
              label: "Release class",
              value: "ENGINEERING_DEMO_ONLY",
              detail: "The full P0 forecasting MVP is incomplete.",
            },
            {
              label: "Source",
              value: "PANGAEA 949811 · CC BY 4.0",
              detail: evidence?.attribution,
            },
            { label: "Archive SHA-256", value: dataset?.source_sha256 },
            {
              label: "Physical calibration",
              value: "BLOCKED",
              detail: blocked,
            },
            { label: "Commercial Marine validation", value: "NOT_EVALUATED" },
          ]}
          testSealLabel="Not opened · R2 freeze unavailable"
          reportView={
            <div className="evidence-report">
              <h3>Actual evidence and limitations</h3>
              <ul>
                {evidence?.limitations.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
              <details>
                <summary>Local source hashes and protocol</summary>
                <pre>{JSON.stringify(evidence, null, 2)}</pre>
              </details>
            </div>
          }
          jsonAvailable={!!evidence}
          onExportJson={exportJson}
          reportAvailable={!!evidence}
          onOpenReport={() =>
            document
              .querySelector(".evidence-report")
              ?.scrollIntoView({ behavior: "smooth" })
          }
          transferFields={[
            {
              field: "Acquisition and receipt timestamps",
              purpose: "Reconstruct data actually available at each decision.",
            },
            {
              field: "Instrument IDs, calibration and acoustic arrays",
              purpose: "Verify units and deployment-specific geometry.",
            },
            {
              field: "Existing model outputs and full low-signal histories",
              purpose:
                "Evaluate against the operational comparator without selection bias.",
            },
          ]}
        />
      ),
    },
  };
  return (
    <AppShell
      screens={screens}
      provenance={{
        sourceLabel: "PANGAEA 949811 / CC BY 4.0",
        datasetLabel: "MOSAiC · AZFP 55170",
        verificationLabel: "Local SHA-256 + ZIP CRC",
        origin: dataset ? "public-data" : "unavailable",
      }}
      context={{
        cutoffLabel: cutoff,
        modeLabel: "Diagnostic replay · uncalibrated",
        uncertaintyLabel: "No forecast intervals",
      }}
    />
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
