# Meeting playbook

## Deliver something testable, not a claim deck

Bring the laptop with a cold-start-tested local application, a second copy of the release,
the source commit, a short English technical report and a one-page data request. No internet,
cloud GPU, API subscription or production Marine account should be required to demonstrate it.
A screen recording is a backup, clearly labelled as such, not a substitute for running software.

## Suggested 10-minute walkthrough

1. (1 minute) `We built a public-data acoustic forecasting demonstrator. The question is whether
   a history of observations helps anticipate the signal at a later decision time. This is an
   Arctic research deployment, not your buoy data and not a tuna model.`
2. (2 minutes) Show one real deployment, coordinates, frequency/range axes, acquisition history,
   the cutoff and missing observations. Explain why the input is a numerical acoustic record,
   not a screenshot or an LLM reading a chart.
3. (2 minutes) Generate or load the explicitly labelled replay forecast. Show +1/+3/+6-hour
   numerical outcomes, intervals and the conventional reference. Reveal actual observations
   afterward. Show one difficult case as well as the representative case.
4. (2 minutes) Show the complete comparison table, three seeds, temporal holdout, uncertainty and
   the actual promotion decision. If JEPA loses, say so: `The platform and protocol work; this
   architecture has not earned deployment. The strongest baseline is the served model.`
5. (1 minute) Degrade observation freshness. Explain the information problem and abstention,
   without claiming a causal fish response or commercial transmission savings.
6. (2 minutes) Show the transfer contract and ask for a small time-complete export plus the
   acoustic/data owner. Suggest a subsequent private pilot with their existing model as the
   comparator, not replacing their software based on this demonstration.

## What should convince them

A real instrument file becomes a calibrated numerical representation; the pipeline preserves
units and timestamps; future predictions cannot inspect the future; the uncertainty is measured;
the application is usable offline; every number links to an artifact; the integration question
is specific and modest. A new neural acronym alone is not the sales argument.

## Expected questions

`Is this already trained on tuna?` No. Public Arctic acoustics demonstrate technical feasibility
and an evaluation workflow, not biological or commercial transfer.

`Why not our existing AI?` Their actual current predictor must be the main comparator in a
private pilot. Do not assume our baseline represents it.

`Why JEPA?` It is a testable way to learn predictive temporal representations; compare it with
simpler methods and keep it only where it improves the selected task.

`Does it predict tonnes?` No. The public outcome is an acoustic backscatter index/profile.
Biomass and species require additional calibration and independent ground truth.

`Can it reduce fuel or calls to the buoy?` Not demonstrated here. Observation-age replay
motivates a later controlled study; it does not measure real commercial benefit.

`Can it plug in tomorrow?` An adapter contract and read-only integration plan exist. Actual
integration needs their schemas, credentials, calibration, ownership and test environment.

`How much data?` Start with several complete deployment histories and all low/null observations,
not just selected good detections. Scale after auditing their cadence and outcome definitions.

## Written takeaway

Provide the actual benchmark result, three biggest limitations, the exact requested export
fields and a proposed next experiment with an owner. Do not present numerical targets from this
handoff as achieved performance. No invented endorsement or Marine Instruments branding.
