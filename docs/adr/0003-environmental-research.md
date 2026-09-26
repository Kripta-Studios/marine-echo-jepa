# ADR0003 — Environmental sources found; calibration remains unapproved

Date: 2026-09-26. Status: PROPOSED_REQUIRES_INDEPENDENT_SCIENTIFIC_REVIEW.
This adds evidence to ADR0002 without rewriting its earlier access findings.

The owner has no local environmental record and authorized internet research. Downloaded
the 79,914-byte scalar and 46,548,009-byte profile files from the Arctic Data Center
[MOSAiC hydrographic dataset](https://doi.org/10.18739/A21J9790B). SHA-256 hashes and
download URLs are in `evidence/calibration/environment_inventory.json`; original metadata
is preserved. Both sizes match the publisher metadata. Files remain separate from immutable
PANGAEA acoustic sources. No additional OOI download was made for this research.

The dataset contains daily Conservative Temperature and Absolute Salinity, not the in-situ
temperature and Practical Salinity expected by common absorption formulas. A reviewed
TEOS-10 conversion and pressure/depth convention are needed. Metadata says profiles combine
MSS, CTD, ITP and a CTD chain; it does not prove co-location with this AZFP. It also says
daily averages may combine observations later than a prediction cutoff. These may be
retrospective measurement corrections, but must not become future predictor covariates.

The [ADEON hardware specification, page 37](https://adeon.unh.edu/sites/default/files/user-uploads/ADEON%20Hardware%20Specification%20VERSION%202.3%20_final%20submission.pdf)
describes the AZFP thermistor as useful for sound speed near the recorder. This supports
investigating its water-temperature role, distinct from the buoy computer's SBPC temperature.
It does not certify this serial's thermistor accuracy or establish salinity or transducer depth.

Executed `tools/inspect_environment.py`. After selecting `Device == MOSAICdown`, 448
GPS records are available. The CSV also contains other devices; the initial unfiltered audit
was invalid for instrument co-location and its log is preserved as rejected exploratory work.
There are 165 environmental profiles in the deployment interval; 164 contain complete finite
temperature/salinity levels from 10–100m. This is environmental coverage, not acoustic eligibility.
Median nearest-record separation is 2.24 km; maximum 207.39 km includes long GPS gaps. Only 111
days meet an illustrative 5 km/24h screen. That screen is descriptive, not a frozen acceptance
threshold. It must not be tuned to maximize downstream acoustic scores. The first acoustic day
has no nearby-in-time `MOSAICdown` GPS record in this parser; deployment metadata and Iridium
positions are potential additional provenance, not yet integrated or validated.

Decision: retain D1 and all scientific gates. Do not claim environmental data are absent;
their applicability and a reproducible calibration procedure remain unresolved. Do not
apply generic Arctic constants, label raw counts Sv, train on all dates, silently alter the
chronological split, or use a fallback dataset without its documented decision process.

Remaining: validate position/time provenance, instrument depth and XML calibration constants,
define prospective temporal/spatial matching and sensitivity bounds, perform TEOS-10 conversion,
calibrate and QC a real hour, then the full corpus, verify eligible-day counts, obtain R0/R1
review and complete implementation before any required benchmark or R2-sealed test. The
independent reviewer session was interrupted by a service usage limit; no approval was granted.
