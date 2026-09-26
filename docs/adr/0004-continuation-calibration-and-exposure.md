# ADR0004 — Retain the calibration block after numerical and trajectory audits

Date: 2026-09-26. Status: BLOCKED_PHYSICAL_PROMOTION; independent R0 review recorded in
`orchestration/reviews/R0-R1-continuation-20260926.json`. No protocol threshold or calendar
boundary is changed. Earlier ADRs and the r2 package retain their historical meaning.

The continuation distinguishes a successful numerical code-path assay from validated
physical calibration. Pinned Echopype 0.11.1 parses training file `20030300.01A`; its
four channels map to the supplied XML coefficients. Scalar recomputation agrees with
upstream Sv to about 2.84e-14 dB. This is implementation consistency, not an independent
field-calibration certificate. The XML configured sound speed is a sensitivity scenario,
not a water measurement.

Environmental depth is positive down in metres. GSW 3.6.23 converts negative height to
sea pressure in dbar, Absolute Salinity to Practical Salinity and Conservative Temperature
to in-situ temperature. Sources: [pressure](https://www.teos-10.org/pubs/gsw/html/gsw_p_from_z.html),
[salinity](https://teos-10.github.io/GSW-Python/gsw_flat.html#gsw.SP_from_SA),
[temperature](https://www.teos-10.org/pubs/gsw/html/gsw_t_from_CT.html),
[sound speed](https://www.teos-10.org/pubs/gsw/html/gsw_sound_speed.html).
The legacy website salinity example (v3.05) differs by up to 7.42e-5 from the pinned
package's atlas. That failed reference check is preserved. The final precise salinity
regression uses a supplied upstream MATLAB-reference fixture, without loosening tolerance
or replacing the expected value with this project's computed result.

Only Iridium fix code 1 is retained for a descriptive trajectory audit. The local source
readme documents signed degrees plus decimal minutes and identifies SBPC temperature as
internal telemetry. No internal buoy temperature is used as water temperature. Coordinate
cross-checks against XEOS support the parser, but duplicate timestamps, a 582.8-hour gap,
and unverified fix semantics prohibit treating the entire trajectory as quality approved.

The initial **Iridium-only** illustrative 5 km/24 h screen matched 97/100 training profile days, 25/25
validation days, 8/16 interval-calibration days and 0/24 profiles in the 26-day test
partition. Even 10 km matches no test profiles. This is environmental coverage, not
acoustic eligibility. That limited tracker audit must not be treated as complete location
coverage; the correction below includes XEOS. It does not justify enlarging the spatial
threshold or moving the test.
All profile availability timestamps remain unknown; retrospective daily means never
become issue-time inputs. Any future reference-target correction must state its distinct
retrospective role and receive independent review.

The original profile metadata links a CTD-chain source DOI 940320. A bounded attempt to
read that specific source's metadata returned HTTP 400; no binary data was downloaded and
no new source applicability was established. The supplied sensor handle resolves to the
public O2A registry item 5748, whose fetched page is only an application shell. Neither
attempt supplied the missing calibration provenance at that point. Later successful public
access and the independently verified certificate are recorded below.

## Subsequent source recovery and coverage correction

The owner renewed internet research authorization during this continuation. Four original
recovered SIT buoy records from [PANGAEA940320](https://doi.pangaea.de/10.1594/PANGAEA.940320)
were downloaded, preserving publisher TSV bytes, source URLs, licenses and SHA-256 in
`evidence/continuation/sit_inventory.json`. The records total488127156 bytes and provide
in-situ temperature, Practical Salinity, pressure, negative-up depth and per-variable QC.
They are ancillary measurements, not a replacement acoustic deployment. They do not supply
close matches to this buoy under the unchanged illustrative screen. Their five discrete
depths, retrospective QC and absent shore-receipt timestamps also need explicit treatment.

The source's independent XEOS tracker contains448 parsed MOSAICdown positions,133 during
the candidate test interval. It fills some long Iridium gaps. Selecting the nearest timestamp
across both trackers, never the nearest spatial distance, yields99/100 train,25/25 validation,
16/16 interval-calibration and12/24 candidate-test composite-profile day matches within5km/24h.
The earlier0/24 is retained only as an Iridium-only result. All three tracker policies and
all four SIT sources are reported in `evidence/continuation/trajectory_matches.json`.
Twelve descriptive matched profile days still do not establish20 eligible acoustic test days.
No matching threshold or chronological boundary was changed.

The independent reviewer resolved the deployed unit as55170 using raw metadata, both XML/DPL
files, PANGAEA and the O2A parent record. It then found a swapped pair of O2A attachments:
the [resource980 certificate](https://registry.o2a-data.de/rest/v2/items/4308/resources/980/payload)
under the item named55171 actually certifies55170. Its four-channel coefficients agree with
the deployment XML to certificate precision. The reviewer accepted that factory coefficient
mapping and provenance. The downloaded PDF's SHA-256 is
`6541fbe21f6af64e0731f53784a2763b87d8527ff79ddcba178bd0dd5eb3f464`;
`evidence/continuation/factory_certificate.json` preserves retrieval evidence.
This resolves factory provenance, not full field calibration, sound-path assumptions or QC.

The already documented OOI fallback remains resource-plan blocked. Existing listings imply
approximately 724 MiB raw/day. An optimistic 134 complete-day calendar needed for the
frozen split/support-day requirements would require about 94.7 GiB raw before ancillary
files, exceeding the 40 GiB cap. Its one-file calibration is a diagnostic only. No new
OOI download, dataset switch, paid compute or threshold relaxation is authorized here.

All audited raw-count app replays cover training day 17 February. The new acoustic assay
opens only 3 March, also training. Environmental applicability is examined across the
deployment and is explicitly recorded separately from acoustic outcome exposure. There
is no known test acoustic exposure in audited artifacts, but human viewing/discarded
history is incomplete. No R2 seal exists. Retain both replay serialization hashes:
tracked `5bbf9ce4...` and packaged `68240f4e...`; their parsed contents are equal.

Decision: finish unaffected software and prepare exact review evidence. Keep G0/R0 closed
until adequate late-drift environmental/geometry provenance or a justified independently
reviewed calibration alternative exists. R1 requires code fixes separately. The 25 required
benchmark entries remain BLOCKED; there is no JEPA-value result, no final evaluation, and
no basis for a new full release. No additional engineering-only ZIP is produced.

## Subsequent reviewed calibration alternative and actual TRAIN QC

The independent reviewer accepted the fixed factory/regional-condition method for bounded
TRAIN candidate preprocessing after prospective depth and slant sensitivity assays. The global
homogeneous path assay's 1.080721 dB failure remains historical; thresholds were never relaxed.
Using fixed2m depth strata, complete regional min/max envelopes, source-coordinate conversions
and conservative path/local-volume bounds gives0.230836 dB/0.825418m over starting depths0–30m.
An additional1886 sampled slant scenarios (start0–10m, angle0–45degrees) gives0.162029dB/0.608893m.
This is a sensitivity study with substantial primary margins, not a continuous uncertainty proof
or field calibration. The assumed0–10m beam-face bracket is an engineering assumption, not a
measurement. Preserve the larger455kHz bounds and the fixed regional/local-colocation distinction.

Actual XML-polynomial TRAIN tilt telemetry exists (~15.4–15.95degrees magnitude on3March).
It must not be confused with missing pitch/roll. ASL's generic nominal beam angles support
the conservative slant envelope; they are not instrument-specific beam certification.
Every date uses the same TRAIN-derived nominal environment; future environmental profiles
never enter forecast inputs. This alternative makes daily environmental matching nonblocking
for TRAIN candidate development, while R0/R1 remain closed pending actual QC/eligibility review.

The first full TRAIN-day pass failed on a quarter containing61 unique pings at a nominal15s
cadence. Timestamp audit shows14/16s rounding jitter, not duplicate timestamps. The reviewer
approved using max(nominal expected, observed) as the support denominator; preserve both and
the excess count. This only makes support more conservative. Failedv1 artifacts remain intact.
The separately versionedv2 ran24 sources,5342 observed of5760 nominal pings and a105-minute gap.
No quarter met80% primary support under per-ping upstream noise removal (10sample blocks,3dB
SNR) plus fixed10m proximal/geometry/fullscale/range masks. This is an actual candidate QC
finding, not a full-deployment eligibility result or a negative JEPA result. Do not lower the
support/SNR requirements to turn it into a benchmark.

The complete3March TRAIN day has now been visualized for QC, and this additional exposure is
recorded. No validation, interval-calibration or test acoustic outcome was visualized. Automated
protected-test QC may only follow exact QC-code/threshold review and must record machine
payload processing separately from human outcome access. R2 remains a separate future gate.
