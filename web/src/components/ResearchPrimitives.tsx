import type { ReactNode } from "react";

export type ResearchState =
  | "loading"
  | "empty"
  | "partial"
  | "unsupported"
  | "error"
  | "success"
  | "blocked"
  | "not-executed"
  | "rejected";

const stateLabels: Record<ResearchState, string> = {
  loading: "Loading",
  empty: "No data",
  partial: "Partial data",
  unsupported: "Unsupported",
  error: "Error",
  success: "Available",
  blocked: "Blocked",
  "not-executed": "Not executed",
  rejected: "Rejected",
};

export interface ResearchStateNoticeProps {
  state: ResearchState;
  message: string;
  title?: string;
}

export function ResearchStateNotice({
  state,
  message,
  title,
}: ResearchStateNoticeProps) {
  const isError = state === "error";

  return (
    <section
      className={"state-notice state-notice--" + state}
      role={isError ? "alert" : "status"}
      aria-live={isError ? "assertive" : "polite"}
      aria-busy={state === "loading"}
      data-state={state}
    >
      <div className="state-notice__mark" aria-hidden="true">
        {state === "loading" ? <span className="state-spinner" /> : <span />}
      </div>
      <div className="state-notice__copy">
        <p className="eyebrow">{stateLabels[state]}</p>
        <h3>{title ?? stateLabels[state]}</h3>
        <p>{message}</p>
      </div>
    </section>
  );
}

export interface ReadoutField {
  label: string;
  value?: ReactNode;
  note?: string;
}

export interface ReadoutGridProps {
  fields: readonly ReadoutField[];
  className?: string;
}

export function ReadoutGrid({ fields, className }: ReadoutGridProps) {
  const classes = ["readout-grid", className].filter(Boolean).join(" ");

  return (
    <dl className={classes}>
      {fields.map((field) => {
        const missing =
          field.value === undefined ||
          field.value === null ||
          field.value === "";
        return (
          <div className="readout" key={field.label}>
            <dt>{field.label}</dt>
            <dd>
              {missing ? (
                <span className="muted-value">Not supplied</span>
              ) : (
                field.value
              )}
            </dd>
            {field.note ? <p className="readout__note">{field.note}</p> : null}
          </div>
        );
      })}
    </dl>
  );
}

export interface PageHeadingProps {
  eyebrow: string;
  title: string;
  description: string;
  trailing?: ReactNode;
}

export function PageHeading({
  eyebrow,
  title,
  description,
  trailing,
}: PageHeadingProps) {
  return (
    <header className="page-heading">
      <div className="page-heading__copy">
        <p className="eyebrow">{eyebrow}</p>
        <h2>{title}</h2>
        <p className="page-heading__description">{description}</p>
      </div>
      {trailing ? (
        <div className="page-heading__trailing">{trailing}</div>
      ) : null}
    </header>
  );
}

export interface SectionHeadingProps {
  id?: string;
  eyebrow?: string;
  title: string;
  description?: string;
  action?: ReactNode;
}

export function SectionHeading({
  id,
  eyebrow,
  title,
  description,
  action,
}: SectionHeadingProps) {
  return (
    <div className="section-heading">
      <div>
        {eyebrow ? <p className="eyebrow">{eyebrow}</p> : null}
        <h3 id={id}>{title}</h3>
        {description ? <p>{description}</p> : null}
      </div>
      {action ? <div className="section-heading__action">{action}</div> : null}
    </div>
  );
}
