# Proposed Marine Instruments transfer — not implemented integration

## Public-to-commercial gap

The public source uses an Arctic research AZFP at 38/125/200/455kHz, with its own orientation,
calibration, sampling and environment. Marine's instruments, habitat, target organisms,
processing and communications differ. Frequency proximity is not equivalence; 38 is not50,
125 is not120, and an algorithmic biomass output is not raw backscatter. Do not rename axes
to match a customer product. New input adapters, calibration and evaluation are required.

Our transferable asset is the engineering/evaluation pipeline and model recipe, not a proven
set of universal marine weights. Reuse learned weights only as a separate measured initialization
arm against scratch and their existing software. The first private phase is read-only replay.

## Request for the first export

A proposed feasibility sample: 10–20 complete deployments with roughly30 continuous days where
possible, including several low/null-signal periods and changes. This is not a statistically
sufficient business-validation sample by definition. A smaller preliminary sample may establish
schemas but not predictive value.

Request:
- per-observation acquisition timestamp, server receipt timestamp and availability to the user;
- buoy/FAD/deployment IDs, instrument model, firmware, calibration and configuration changes;
- numerical acoustic arrays by channel and range/depth, units, noise/QC flags and masks;
- both pre-filter and post-filter output when legally/exportably available;
- GPS and motion/quality signals actually recorded, plus telemetry status and battery;
- requests, scheduled/actual transmissions and the complete observation history, not only maxima;
- oceanographic covariates with provider, license, forecast issue time and valid time;
- visits/arrivals/operations and any independently measured outcomes, including missingness;
- existing model/version outputs as a fair comparator and definition of the business decision.

Do not require all optional channels as a prerequisite to start. Record exactly which sensors
exist on each product and which data Marine may use across clients. Internal device temperature
must remain distinct from seawater temperature. No unobserved catch is coded as zero.

## Integration contract

Define a versioned ingestion adapter into the canonical observation schema and map only
supported fields. Side-by-side comparison runs with the historical data that was actually
available at each decision time. Keep the current operational model authoritative; forecast
service is read-only and makes no firmware or fleet-control change. Default outputs: future
signal, uncertainty, input quality and reasons to abstain. Any catch or route optimization
requires a separately defined target, evaluation and authorization.

Acceptance on Marine data must include unseen deployments and future time, sensor/firmware
strata, their current model, decision cost and relevant operators. A good Arctic test does not
waive these requirements. A rejected model may still yield useful data-quality findings, but
those are not equivalent to a successful predictive pilot.
