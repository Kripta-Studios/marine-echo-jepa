# Acoustic data contract

## Identities and lineage

Every record carries `dataset_id`, `dataset_version`, `deployment_id`, `instrument_id`,
`source_file_sha256`, `config_sha256`, `processing_version` and `calibration_status`.
Use physical deployment identities, not filenames, as the scientific unit. Raw files are not
independent experimental replicates. Preserve the instrument's real frequency values in Hz.

## Canonical arrays

Persist chunked xarray-compatible arrays, using a pinned tested Zarr version or day-sharded
NetCDF/NPZ alternative. Shape: `[time, frequency, range_bin]`.

| Field | Contract |
|---|---|
| `event_time_utc` | timezone-aware acquisition/bin-end time; UTC |
| `available_time_utc` | when available to a user if known; otherwise null |
| `replay_available_time_utc` | explicit simulation value, separate from real availability |
| `frequency_hz` | actual numeric channel metadata; no nearest-frequency relabelling |
| `range_edges_m` | distance from transducer; use `depth_m` only with verified geometry |
| `sv_linear_m_inv` | calibrated linear volume backscattering coefficient, nonnegative |
| `sv_db` | 10 log10(sv), with a documented numerical floor and validity mask |
| `valid_mask` | true only for valid calibrated samples |
| `qc_reason` | flags for absent data, noise, geometry, saturation, configuration and calibration |
| `latitude`, `longitude` | observed positions and their observation timestamps, or null |
| `sensor_internal_temperature_c` | internal telemetry only when actually parsed |
| `orientation` | down/up/unknown; never silently flipped |

If calibration coefficients/environmental assumptions cannot be established, preserve counts
under an explicitly different field and stop calibrated-physics claims. Do not put raw counts in
`sv_db`. “Factory-calibrated using recorded/default conditions” is different from an independently
field-calibrated instrument. All assumptions belong in the calibration report.

## Working representation

Default proposal: 15-minute, trailing/non-centred bins; 64 two-metre range bins from 0 to 128 m;
four D1 frequencies and an explicit mask. The exact grid and usable analysis band are frozen
following training-only metadata/QC inspection, before validation model comparisons. Never
extrapolate a channel into unsupported range. Different-frequency masks may differ.

The primary index uses the 38 kHz channel and a proposed 10–100 m analysis band. If unavailable
or invalid, revise the protocol prospectively before any model comparison. Do not silently swap
frequency at test time. A profile can have valid values while its aggregate index is ineligible.

## Primary target

For a forecast issued at cutoff t, predict the mean acoustic state during the **one-hour period
ending at t+h**, where h is 1, 3 or 6 hours. Thus target intervals are [t,t+1h), [t+2h,t+3h),
and [t+5h,t+6h), with a documented boundary convention used everywhere.

Define a range/time-weighted mean of linear sv over the fixed band, then take 10 log10.
Require at least 80% valid weighted support. Store the support fraction and the unlogged value.
The resulting `backscatter_index_db` is an acoustic proxy, not fish count, tonnes, biomass or energy.

The profile target averages sv linearly over the same future hour for each frequency/range bin,
then logs it. Return validity and support, not invented complete profiles.

## Serving contract

Every forecast has model/artifact hashes, dataset identity, cutoff, target interval, quantiles,
status, input age, coverage, and provenance. `UNAVAILABLE`, `ABSTAIN` and `EXPERIMENTAL` are
normal states. Quantiles must be ordered and finite when a forecast is present. Use null for
missing JSON values, never NaN or zero as a substitute. Observed future truth is a separate
replay channel; the predictor must never receive it.

## Exact bin boundary rule

Store both bin_start_utc and bin_end_utc. Each numerical bin aggregates raw acquisitions in
[bin_start, bin_end). Context includes bins with bin_end <= cutoff. A physical future target
interval [start,end) therefore selects complete bins with bin_end > start and bin_end <= end.
Do not filter bin-end timestamps using [start,end): that would reuse the context's final bin
and drop the target's last bin. Keep raw-ping, bin-interval and UI timestamp conventions separate.
L06 must perturb a ping exactly at cutoff and verify it belongs only to the future target,
while a ping immediately before cutoff belongs only to the past input.
