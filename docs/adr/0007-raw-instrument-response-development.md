# ADR 0007: fixed raw instrument-response development target

Date: 2026-09-27. Status: proposed before raw-response support processing or model fitting.
Study identity: `raw_response_development_v1`, separate from calibrated Candidate 2 and v1.

## Decision and purpose

The independently verified native Candidate 2 has zero April assessment target days
at its frozen 10% detected-range floor. ADR 0006 therefore stops further D1
calibrated-target/support search. The handoff expressly permits a separate
`RAW_RESPONSE_DEVELOPMENT` route for proving real corpus, baseline and neural
execution while physical calibration is unresolved. This ADR defines exactly one
such engineering measurement before its support is calculated. A failed support
audit ends this raw route; it does not trigger a changed bin band, date segment,
minimum or second raw candidate.

The target is the arithmetic mean of Echopype-derived AZFP 38 kHz
`backscatter_r` response codes over zero-based averaged range-bin indices 20
through 199 inclusive and over valid observed pings in a fixed UTC hour. This
is a transformed instrument response, not received digitizer amplitude, an
integer count, calibrated Sv, seawater backscatter, or a metre-depth profile.
The bound `data_type=1` parser reads a 32-bit linear sum and an overflow byte,
divides by 14 range samples per averaged bin, and applies a log/DS transform.
It replaces negative-infinite log of a zero linear sum with code zero. A code
zero is therefore ambiguous and is masked as `zero_or_undefined_code`, never
interpreted as a physical zero. Other finite nonzero response codes, including
negative values, remain valid. Nonfinite codes are masked separately. Because
the zero code can be caused by a true zero pre-log linear sum, requiring every
code in a ping to be nonzero conditions this engineering target on a complete
positive response and may select toward stronger returns. This conditioning
is part of the estimand, not an instrument-failure diagnosis or an unbiased
acoustic population claim. Report zero-code samples, affected pings, affected
hours, issued rows and scored coverage for each horizon without filling them.
No 16-bit
saturation threshold is applied to these transformed codes, and no overflow
status is invented from them. Parser/transform hashes and the exact acquisition
configuration must be bound before processing; a decoder change requires review.
There is no second logarithm or acoustic unit conversion in the target.

The fixed segment omits the earliest 20 averaged bins by preregistered
convention; it makes no claim that ringdown or near-field interference has
ended. It is the native-bin counterpart of the already declared 10–100 m
geometric band, without asserting identical physical edges or using sound
speed to define the target. The segment is supported by the preserved 999-bin
38 kHz acquisition configuration, not chosen from the failed Candidate 2
signal or April outcome.

A ping is target-valid only if all 180 codes are finite and nonzero. Record
zero/undefined-code and other nonfinite failures as disjoint ping categories,
with separate sample counts. Retain invalid pings in observed acquisition
counts. The hourly target is the arithmetic mean of all 180 codes on all
target-valid observed pings. Require at least 120 target-valid pings per hour
and at least 95% target-valid pings among observed pings. The 120 floor is half
the nominal 240 scheduled 15-second pings per hour; the 95% floor explicitly
limits how many observed pings with zero/undefined or nonfinite codes can be
excluded and reinforces the complete-response conditioning. Expected is
240/hour, missing scheduled is max(240-observed,0), and excess observed is
max(observed-240,0). Report all counts and acquisition coverage separately.
Do not reject low finite nonzero codes as below acoustic detection.

Keep the original fixed TRAIN-development calendar: fit 17 February through
31 March 2020 inclusive, assessment 1 through 14 April inclusive. Use 24
preceding one-hour slots, causal masks and actual past acquisition/configuration
only. Forecast the fixed future UTC hours at 1, 3 and 6 hours after issuance;
never shift to the next favorable observation. Issuance requires the last past
hour to have at least 120 target-valid pings, at least 18 of 24 past hours to
have at least 120 target-valid pings, and a single exact acquisition
configuration across observed past hours matching the frozen reference
configuration. Future acquisition or configuration cannot affect issuance,
inputs, or the forecast. At scoring, reject a target hour whose actual
configuration differs from that reference and record `target_configuration`
separately from a missing/invalid target. This is a scheduled-issue engineering
task; a missing future target is an abstention for scoring, not filled truth.
Perturbing future codes, masks, timestamps or configuration must leave the
issuance mask and model input unchanged. Each horizon's
common cohort is frozen before model comparison. Windows cannot cross the
fit/assessment boundary in raw input or target support; any required purge is
computed from the 24-hour context and 6-hour maximum target horizon.

Before fitting, require at least 20 distinct fit target days and five distinct
assessment target days at each reported horizon. The seven nonoverlapping
48-hour assessment blocks are anchored at 2020-04-01T00:00Z. Assign eligible
rows to blocks by target timestamp. A populated block has at least one
eligible target row for that horizon; require at least five populated blocks
separately at each reported horizon. If the block requirement is too ambitious
for 14 assessment days, record ineligibility; do not lower it afterward. Count target days
separately from issuance days, pings and overlapping rows. These are engineering
development adequacy limits, not a claim of statistical independence or final
evaluation power. Seven possible assessment blocks cannot meet the calibrated
v2 protocol's ten-block inference gate, so no calibrated interval, superiority
or JEPA-value inference may be made. This route never consumes the calibrated v2 core campaign
or its 25 slots.

The source is the existing downward AZFP serial 55170 TRAIN raw archive with
the exact historical source-file SHA256, XML SHA256, parser and transform code
SHA256, averaged-bin configuration and timestamps verified per file. Require
nominal channel order 38/125/200/455 kHz, digitization rate 20000, lockout
index 0, bins per channel 999/999/391/235, range samples per averaged bin 14,
data type 1, no ping averaging, transmit duration 0.001 second and 15-second
scheduled period. Assert XML detector-slope DS values in channel order
0.02300000004470/0.02290000021458/0.02319999970496/0.02309999987483.
DS is an instrument-specific coefficient in Echopype's code transform, not
evidence that this response is calibrated Sv. A mismatch blocks a window
rather than changing the target.
This route does not inherit a calibration approval, regional sound-speed
profile or numeric Sv from Candidate 2. Any model normalizer is fit on raw
fit records only. Input channels and profile aggregation for the neural executor
will be specified and reviewed before fitting; no fake zero channel may be
presented as an observation. If four-frequency context is used, group each
frequency's same averaged-bin indices 20–199 into 64 predetermined contiguous
groups using group index floor(64*(bin-20)/180). Store conditional means of
finite nonzero codes with explicit group masks and valid counts. The 38 kHz
target remains the full 180-bin/ping arithmetic mean, never a mean of 64
pre-aggregated profile cells.

All artifacts use `raw_response_development_v1` paths and label the quantity
`complete_positive_azfp_backscatter_r_code_mean`. The only intended conclusion is whether the REAL
software path can fit, checkpoint, resume and predict this recorded instrument
response on later TRAIN-development observations. Successful prediction here
does not establish a calibrated acoustic MVP, JEPA value, biology, biomass,
operational benefit, or a sealed holdout. The calibrated Candidate 2 failure
and blocked historical v1 registry remain intact.
