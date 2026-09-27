# Candidate 2: native sampled range intervals

Date: 2026-09-27. Status: proposed, before native candidate support or model scores.

Candidate 1 in ADR 0005 has zero eligible hourly index targets at its prospectively
fixed 10% detected-grid support floor. Preserve `evidence/v2/support/` unchanged.
No training ran. This is a failed measurement-support candidate, not a JEPA result.

The original conservative regrid requires every overlapping native range sample to
pass the 3 dB rule before a two-metre grid cell is valid. Approximately four to five
native samples overlap each cell. That population consists of entirely detected
two-metre cells, rather than all detected native sampled intervals. The first-hour
native audit retained valid detections that need not form such contiguous groups.

Candidate 2 retains 38 kHz, 10–100 m slant range, the unchanged calibration and
per-ping 3 dB background-subtraction method. It changes only the sampled population
to native acquisition range intervals, with exact geometric overlap lengths in the
fixed band. No missing or below-detection sample is inferred, interpolated or zero.

For ping p and native bin j, let w_j be the length in metres of the intersection of
that native sampling interval with [10,100). Let d_pj indicate a valid detected
noise-corrected native return. Define S=sum(w_j*d_pj*sv_pj), W=sum(w_j*d_pj),
O=sum(w_j) across geometrically available native intervals and observed pings.
The current geometry covers the full band, so O=90*number_of_observed_pings;
verify this equality, rather than assuming geometry coverage on new sources.
The conditional index is 10log10((S/W)/(1 m^-1));
the detection fraction is W/O. W is detected range-length summed across pings,
not a count of independent observations. Store actual native sample and ping counts
separately. Boundary intersections account for measured native sample intervals and
do not imply finer-resolution reconstruction.
This is range-length weighting, not weighting by insonified volume or beam area.
No additional beam-pattern, r-squared volume or biological-depth claim is made.

Keep the >=120 observed pings/hour and >=10% detection fraction floors. Keep the
calendar, horizons, past-only issuance, context, matched cohorts, joint loss,
engineering-smoke budget and claim limitations of ADR 0005. Do not use the failed
candidate to choose a new frequency, range, date segment or lower support floor.
This is candidate 2 of a maximum three defined measurement contracts; no third
candidate is currently defined and no alternative frequency is under selection.
Stopping rule: if Candidate 2 fails, no further D1 calibrated-target variant or
support search will be performed in this continuation. A third D1 candidate must
not be invented from its support results. The failed measurement route remains
failed; only the separately permitted alternative-source tracks and unfinished
software work may continue.

Context profiles may be represented on the same fixed two-metre display/model grid
as conditional means of detected native overlaps. They must carry detected overlap
weights, acquisition counts and masks. They are explicitly detection-conditioned
sampled profiles, never complete valid two-metre cells under v1. Aggregate the
scalar target from S and W, not from an unweighted average of conditional cell means.

Source processing must write a new namespace with source hashes, native timestamps,
configuration and calibration bindings. Reuse the preserved source bytes. This is
new native sufficient-statistic processing for the amended study, not a rerun of
the closed v1 grid census or any-band bound. Begin with the fixed development
calendar only after distinct review of this measurement and implementation. Do not
open non-TRAIN acoustic payloads or promote full-period calibration implicitly.
No fitting is permitted until each reported horizon has at least 20 eligible fit
target days and five eligible assessment target days. Report target days separately
from cutoff days, all issued rows, fixed 48-hour blocks, and every exclusion reason.
