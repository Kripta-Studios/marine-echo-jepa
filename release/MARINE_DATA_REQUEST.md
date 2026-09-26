# Internal data-request checklist for a future Marine pilot

**Status:** Meeting preparation only. This checklist has not been sent, and no company outreach
is authorized by this document.

The public Arctic experiment cannot validate a Marine operational use case. Before a customer
pilot is designed, obtain permission and the data below from an authorized owner. Keep the
scientific study separate from commercial validation.

## 1. Resolve AZFP calibration provenance

The local archive XML/DPL identify downward AZFP serial 55170; the platform manual lists 55169.
Request documentation that resolves which instrument produced the observations and matches the
serial, deployment and dates:

- Instrument serial numbers, transducer/channel configuration, calibration certificates and
  maintenance/calibration history for the deployed AZFP.
- Authoritative seawater temperature, salinity and pressure measurements or approved values for
  each relevant time/location, including sensor type, location, timestamp, units and uncertainty.
- Absorption inputs or coefficients by acoustic frequency, with the calculation method,
  software/version and provenance. Confirm which parameters may be derived and which must be
  supplied directly.
- Transducer depth, attitude/orientation, range convention, sample timing and geometry, with
  any changes during deployment. State clearly whether a temperature sensor measures seawater
  or internal sonar temperature.
- An authoritative correction or explanation for the manual/archive serial discrepancy.

Do not substitute buoy-computer temperature, undocumented defaults or a configured sound-speed
value for verified environmental calibration evidence.

## 2. Define an authorized operational evaluation, if desired

For any future evaluation of a Marine use case, first agree in writing on the exact prediction
target and decision it would support. Then request an authorized, time-aligned data package with:

- Instrument/platform identifiers, acquisition and receipt timestamps, channel/configuration
  history, calibration records, missingness, transmission delays and quality-control flags.
- Existing model outputs and the current operational comparator, including the versions and
  timestamps that were available at each decision.
- Ground-truth observations or other independently defined outcomes, their collection method,
  timestamps, uncertainty, and when they became available. Do not infer biological labels from
  the public Arctic acoustic replay.
- Enough contiguous deployment history for a preregistered chronological evaluation, with
  distinct training, validation, calibration and test periods and an agreed protected-test
  procedure.
- Data-use rights, retention limits, approved storage location, de-identification requirements,
  attribution language and permission for derived artifacts or meeting demonstrations.

No claim about species, biomass, catch, fuel savings or production integration should be made
unless a separately approved study measures the corresponding outcome against an agreed
operational comparator.

## 3. Resource decision for the OOI fallback

The one-file OOI feasibility probe does not make OOI a benchmark corpus. A rough 90-day estimate
is about 63.6 GiB, above the configured 40 GiB raw-data ceiling. If OOI is considered again,
request an owner-approved acquisition, storage, retention and processing plan before any broader
download. Do not change the primary dataset or experiment protocol silently.

## Meeting outcome to record

Record whether authoritative serial-matched AZFP calibration material is available, who owns any
future customer data, what target and comparator are authorized, and whether an approved resource
plan exists. A “not available” answer remains a blocker; it does not authorize guessed calibration
values, a silent dataset switch, or a commercial claim.


## Later evidence update

The internet search found and downloaded MOSAiC environmental profiles (DOI 10.18739/A21J9790B). Their spatial/temporal applicability and calibration procedure remain unapproved; see ADR0003. Earlier missing-input statements describe the initial supplied files, not the absence of public environmental data. Model software tests and a 100-update CUDA resource profile passed using synthetic fixtures, not benchmark data. All three configured native agents subsequently returned usage-limit errors. Final independent scientific/release review and Sol review of these Luna-authored drafts remain incomplete.
