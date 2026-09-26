# Web app: Acoustic Forecast & Freshness Lab

## Product tone

Use English throughout. A clean technical workspace, not an imitation of Marine Instruments'
brand or a fake deployed customer portal. Brand `Marine Echo JEPA`; subtitle `Public-data
research demonstrator`. Persistent notice: `Arctic research deployment. Not tuna, catch or
Marine Instruments validation.` Never put a fabricated customer logo or a fleet of invented
buoys on a map.

Design for 1440x900 meeting display and 390x844 mobile. Desktop sidebar plus top provenance
bar; on small screens collapse navigation without losing cutoff, mode or uncertainty. Neutral
navy/slate base with accessible accents is a suggested UI design, not a requirement to copy an
existing site. Use system/local fonts. Visible focus, keyboard navigation, descriptive button
labels, accessible table fallback to charts and no colour-only state encoding.

## Five useful screens

### 1. Deployment replay

Real track/location context, source/license/date, selected channel and instrument orientation.
Main time-range echogram with a vertical prediction-cutoff marker; acoustic intensity legend
in dB, explicitly range below transducer unless true depth established. Distinguish observed,
forecast and missing cells with labels/textures. Range/channel settings cannot silently change
the trained model's input contract. Playback changes the cutoff, not the clock of a real buoy.

Display bottom cards: last available observation, valid support, calibration status, selected
model, uncertainty and mode. Include `Forecast here` and separate `Reveal observed outcome`.
Before reveal, no future observations in chart payloads. After reveal, show predictions and
observations with equal axis scales; keep the cutoff visible.

### 2. Forecast comparison

At +1/+3/+6h show acoustic-index median and empirical90% interval. Compare the strongest
reference and the candidate. A depth-profile point forecast is a separate chart and labelled
as point-only unless a valid profile distribution has been implemented. No fish-count icons,
tonnes or risk colours implying catch. Show units and window aggregation formula in tooltip.

### 3. Observation freshness

Select an observation age or predefined dropout scenario. Mark this a replay simulation.
Show changed input availability, resulting forecasts, empirical uncertainty and quality reasons.
A suggested refresh can be based on a transparent validation-fitted rule. It is not an
optimized satellite schedule, a biological intervention or a demonstrated fuel-saving tool.

### 4. Experiment lab

Table with family, seed count, primary loss, MAE, coverage90, interval width, sample/day counts,
training cost and outcome. Show conventional winner even when JEPA loses. Filter validation,
calibration and final-test reports explicitly. A scientific rejection is not a software crash;
a missing run is `Not executed`, not `Failed hypothesis`. Include paired-difference charts,
confidence interval method and a link to the exact protocol. No artificially smooth metrics.

### 5. Evidence and transfer

Data provenance, checksums, licenses, model card, calibration assumptions and test seal. A
second panel shows proposed Marine export fields and the difference between the public
prototype and a future customer pilot. Provide CSV/JSON export and local report view. No
chatbot, login/signup workflow, billing, admin CMS, cloud upload or operational control in P0.

## UI acceptance

Loading, empty, partial/missing, unsupported window, error and success states are implemented.
Hover/touch tooltips cannot hide essential numbers. Charts share the same timestamp semantics.
No horizontal page overflow at mobile width; tables may scroll within their own container.
All navigation buttons work, browser back preserves selected dataset/cutoff, and refresh does
not swap the selected model. Requests are cancellable; old responses cannot overwrite a new
cutoff. Cache keys include all provenance. An offline Playwright test denies external network
and completes the full replay -> forecast -> reveal -> evidence -> export journey.
