# Preprocessing and acoustic integrity

Use upstream Echopype [S05, S06] for supported AZFP/EK60 parsing rather than inventing a binary
parser or rebuilding calibration equations. Pin the actual package/source version and verify
its API against a small real file. Public documentation lists v0.11.1 as a current release at
preparation; this version introduced dependency changes including Zarr 3. Do not combine it
with an old Zarr 2 recipe copied from another project.

## Pipeline

1. Immutable raw inventory and per-file SHA-256; UTC configuration/event mapping.
2. Match each raw segment to the applicable XML/configuration and calibration coefficients.
3. Parse one file with bounded CPU/RAM, calibrate to sv/Sv, and compare selected numerical
   outputs with a documented upstream example/reference to detect unit/order mistakes.
4. Preserve channel-specific range geometry and valid support before regridding.
5. Apply fixed near-field/surface/bottom/noise rules appropriate to instrument orientation.
   Do not call all missing/saturated returns absence of animals. Do not extrapolate the seabed
   from the catalog's water depth or assume a 5 m cable is exact transducer depth.
6. Regrid and average in linear sv, then log for model inputs/plots; never arithmetic-average dB
   when the quantity is intended to be mean backscatter. Keep original and QC-derived masks.
7. Aggregate using strictly trailing time intervals. Interpolation must not pull from future
   samples; long gaps remain gaps. Create sampling-mode boundaries and forbid windows across them.
8. Construct target windows and a split manifest before training. A window never crosses a
   split, missing-data discontinuity above the frozen threshold, or a deployment/config boundary.
9. Fit scaling, clipping thresholds, feature selection and context encoders on training only.
10. Persist day-sized partitions and an atomic completed-partition manifest.

## No hidden look-ahead

Calibration constants from instrument metadata may be used according to their physical validity.
Data-dependent denoising statistics, normalization and masks must not use held-out future values.
Time of day at a future horizon is known at issue time; future actual GPS, water conditions or
full-record smoothers are not. Distinguish acquisition timestamps from data availability.
When historical availability is absent, state that the benchmark assumes zero-latency availability.

## Windows resource rules

Use a separate `envs/data` environment if acoustic dependencies conflict with training packages.
Persist the canonical arrays as the interface; the two processes need not share an environment.
Start with two CPU workers and one file per worker; bound memory explicitly. Do not load the
complete raw record or create a Dask task for every ping. Windows entrypoints need main guards;
training DataLoader starts at zero workers and can move to two only after a measured test.

## Mandatory QC artefacts

`raw_inventory.parquet`, `calibration_report.json`, `coverage_by_day.csv`, `frequency_range_support.json`,
`configuration_boundaries.json`, `processing_manifest.json`, and a training-only review page with
representative/noisy/missing segments. Expose rejected rows and reasons. The reviewer checks at
least one AZFP sample's metadata-to-unit path. If interpretation remains uncertain, downgrade
claims and retain engineering work; do not fabricate confidence.

## AZFP environmental calibration preflight

Official compute_Sv documentation [S20] requires environmental parameters for AZFP, such as
sound speed and absorption or the variables needed to derive them. Configuration XML alone
is not sufficient evidence of those water conditions. Resolve these from source-provided
metadata/manual or an explicitly verified compatible environmental record. Do not reuse the
surface buoy's internal temperature as water temperature. Record measured versus assumed
parameters, sources and units, and report sensitivity for any justified approximation. Missing
instrument calibration or unjustified environmental inputs block physical-unit benchmark
promotion. A relative raw-count visualization can still be delivered, labelled diagnostic,
without being substituted for a verified calibrated forecast benchmark.
