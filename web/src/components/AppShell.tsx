import { useEffect, useState, type ReactNode } from "react";
import "../styles/research-demo.css";

export type AppRoute =
  | "deployment-replay"
  | "forecast-comparison"
  | "observation-freshness"
  | "experiment-lab"
  | "evidence-transfer";

export interface AppScreen {
  title: string;
  eyebrow?: string;
  content: ReactNode;
}

export type DataOrigin = "public-data" | "cached-replay" | "synthetic-fixture" | "unavailable";

export interface ShellProvenance {
  sourceLabel?: string;
  datasetLabel?: string;
  verificationLabel?: string;
  origin?: DataOrigin;
}

export interface ShellContext {
  cutoffLabel?: string;
  modeLabel?: string;
  uncertaintyLabel?: string;
}

export interface AppShellProps {
  screens: Readonly<Record<AppRoute, AppScreen>>;
  provenance?: ShellProvenance;
  context?: ShellContext;
}

const routeItems: ReadonlyArray<{ id: AppRoute; label: string; number: string }> = [
  { id: "deployment-replay", label: "Deployment replay", number: "01" },
  { id: "forecast-comparison", label: "Forecast comparison", number: "02" },
  { id: "observation-freshness", label: "Observation freshness", number: "03" },
  { id: "experiment-lab", label: "Experiment lab", number: "04" },
  { id: "evidence-transfer", label: "Evidence & transfer", number: "05" },
];

function routeFromLocation(): AppRoute {
  if (typeof window === "undefined") return "deployment-replay";
  const rawRoute = window.location.hash.replace(/^#/, "");
  return routeItems.some((route) => route.id === rawRoute)
    ? (rawRoute as AppRoute)
    : "deployment-replay";
}

function originLabel(origin: DataOrigin | undefined): string {
  switch (origin) {
    case "public-data": return "Public data";
    case "cached-replay": return "Cached replay";
    case "synthetic-fixture": return "Synthetic fixture";
    case "unavailable":
    default: return "Source pending";
  }
}

export function AppShell({ screens, provenance, context }: AppShellProps) {
  const [activeRoute, setActiveRoute] = useState<AppRoute>(routeFromLocation);
  const activeScreen = screens[activeRoute];

  useEffect(() => {
    const syncRoute = () => setActiveRoute(routeFromLocation());
    window.addEventListener("hashchange", syncRoute);
    return () => window.removeEventListener("hashchange", syncRoute);
  }, []);

  const links = routeItems.map((route) => (
    <a
      className="primary-nav__link"
      href={"#" + route.id}
      key={route.id}
      aria-current={activeRoute === route.id ? "page" : undefined}
      onClick={(event) => {
        const compactMenu = event.currentTarget.closest("details");
        if (compactMenu) compactMenu.open = false;
      }}
    >
      <span className="primary-nav__number" aria-hidden="true">{route.number}</span>
      <span>{route.label}</span>
      <span className="primary-nav__arrow" aria-hidden="true">↗</span>
    </a>
  ));

  return (
    <div className="marine-app">
      <a className="skip-link" href="#main">Skip to main content</a>
      <aside className="desktop-rail" aria-label="Application navigation">
        <a className="brand-lockup" href="#deployment-replay" aria-label="Marine Echo JEPA home">
          <span className="brand-mark" aria-hidden="true"><i /><i /><i /></span>
          <span><strong>Marine Echo</strong><small>JEPA / FIELD NOTE 01</small></span>
        </a>
        <div className="rail-label">Research workspace</div>
        <nav className="primary-nav" aria-label="Research screens">{links}</nav>
        <div className="rail-bottom">
          <span className="rail-coordinate" aria-hidden="true">ARCTIC / PUBLIC STUDY</span>
          <p>Acoustic forecasting<br />and evidence review</p>
          <span className="rail-dot" aria-hidden="true" />
        </div>
      </aside>

      <div className="workspace">
        <header className="topbar">
          <div className="topbar__brand">
            <span className="brand-mark brand-mark--small" aria-hidden="true"><i /><i /><i /></span>
            <span>Marine Echo JEPA</span>
          </div>
          <div className="topbar__identity">
            <p>Public-data research demonstrator</p>
            <span className="origin-badge" data-origin={provenance?.origin ?? "unavailable"}>
              <span className="origin-badge__dot" aria-hidden="true" />
              {originLabel(provenance?.origin)}
            </span>
          </div>
          <div className="provenance-bar" aria-label="Dataset provenance">
            <div><span className="topbar-label">Source</span><strong>{provenance?.sourceLabel || "Not verified"}</strong></div>
            <div><span className="topbar-label">Dataset</span><strong>{provenance?.datasetLabel || "Not selected"}</strong></div>
            <div><span className="topbar-label">Integrity</span><strong>{provenance?.verificationLabel || "Not checked"}</strong></div>
          </div>
          <div className="compact-provenance" aria-label="Dataset provenance">
            <div><span>Source</span><strong>{provenance?.sourceLabel || "Not verified"}</strong></div>
            <div><span>Dataset</span><strong>{provenance?.datasetLabel || "Not selected"}</strong></div>
            <div><span>Integrity</span><strong>{provenance?.verificationLabel || "Not checked"}</strong></div>
          </div>
          <details className="mobile-navigation">
            <summary aria-label="Open research screens">
              <span className="mobile-navigation__bars" aria-hidden="true"><i /><i /><i /></span>
              <span>Sections</span>
            </summary>
            <nav className="primary-nav" aria-label="Research screens">{links}</nav>
          </details>
        </header>

        <div className="mobile-context" aria-label="Replay context">
          <ContextItem label="Cutoff" value={context?.cutoffLabel} />
          <ContextItem label="Mode" value={context?.modeLabel} />
          <ContextItem label="Uncertainty" value={context?.uncertaintyLabel} />
        </div>

        <main className="main-content" id="main" tabIndex={-1}>
          <div className="page-meta">
            <div>
              <p className="eyebrow">{activeScreen.eyebrow || "MARINE ECHO / RESEARCH"}</p>
              <h1>{activeScreen.title}</h1>
            </div>
            <div className="desktop-context" aria-label="Replay context">
              <ContextItem label="Cutoff" value={context?.cutoffLabel} />
              <ContextItem label="Mode" value={context?.modeLabel} />
              <ContextItem label="Uncertainty" value={context?.uncertaintyLabel} />
            </div>
          </div>
          <div className="persistent-notice" role="note">
            <span className="persistent-notice__mark" aria-hidden="true">!</span>
            <p>Arctic research deployment. Not tuna, catch or Marine Instruments validation.</p>
          </div>
          <section className="active-screen" id={activeRoute} aria-labelledby="screen-title">
            <span id="screen-title" className="visually-hidden">{activeScreen.title}</span>
            {activeScreen.content}
          </section>
          <footer className="app-footer">
            <span>LOCAL RESEARCH INTERFACE</span>
            <span>Public data · provenance required for every claim</span>
          </footer>
        </main>
      </div>
    </div>
  );
}

function ContextItem({ label, value }: { label: string; value?: string }) {
  return (
    <div className="context-item">
      <span>{label}</span>
      <strong>{value || "Not supplied"}</strong>
    </div>
  );
}
