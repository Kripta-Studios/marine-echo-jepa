# Native acoustic SSL: first strong development comparison

Date: 2026-09-30. Study: native_acoustic_ssl_v1. Actual execution completed
exit0 after distinct reconstruction approval and verification of all58 bindings.
This is development evidence; reserved AEON2 final-test values remain unopened.
The distinct reviewer subsequently reconstructed the actual eight prediction
artifacts independently with NumPy: every point score and7,584 daily losses
matched exactly. The bootstrap sequence hash matched; interval endpoints
differed by at most1.11e-16dB. The closed review is recorded in
development-numeric-and-budget-review-final.json, numeric_reconstruction.

The comparison uses exactly7,593 issued windows from AEON4_JOB:61937266,
native0–225m hourly38kHz integrated Sv, with horizons1/3/6 and five quantiles.
Every method has identical rows, observed target masks/values, source dates and
queries. There are7,581/7,557/7,521 observed targets and316 eligible target-source
dates at each horizon. The fixed minimum is18 observations per date/horizon;
scores weight source dates, horizons and deployments equally. No depth relabel,
rescaling, support intersection, target filling or finite-forecast filtering.

| Method | Endpoint | Development pinball dB |
| --- | --- | ---: |
| LightGBM | fixed15 quantile boosters |0.5052735501 |
| CF-JEPA | selected6000-SSL-screen500-update frozen probe |0.7457265899 |
| Direct supervision | fresh3000-update end-to-end trajectory |0.7651851961 |
| Shared temporal SSL | fresh3000-update full fine-tuning |0.7725943368 |
| Shared temporal SSL | fresh2000-update frozen readout |0.7857153105 |
| Shared temporal SSL | selected6000-SSL-screen500-update frozen probe |0.8246468455 |
| Persistence | context-only point forecast at every quantile |0.9891759562 |
| Seasonal24 | context-only source-period forecast |1.0317384666 |

The shared frozen endpoint trails LightGBM by0.2804417604dB; its paired95%
development difference interval is[0.2565263174,0.3044787833]dB. Shared full
fine-tuning trails by0.2673207867dB,[0.2387174496,0.2992089163]. Direct trails
by0.2599116460dB,[0.2313125386,0.2922709960]. CF's short probe trails by
0.2404530398dB,[0.2085548483,0.2791036294]; it is not its strong transfer
endpoint. These comparisons do not equalize pretraining compute or readout
budgets. All methods produce forecasts that vary across issuance.

The frozen recipe uses2000 jointly paired bootstrap draws, seed20260929,
nonoverlapping two-source-calendar-day blocks representing48 nominal hourly
intervals, and retains gaps and sampled multiplicities. All2000 replicates
are assessable. There is only one development deployment; source timezone and
absolute clock remain unknown. Intervals are conditional on these rows and
source-calendar proxies, not independent sites, verified UTC durations,
final-test generalization or seed variance.

No JEPA-value or SOTA claim follows. The shared frozen representation still
needs strong matched masked, random and pairing-permuted comparisons. The CF
strong endpoints have separate completed-parent prefit approval but have not
run. A synthetic proof establishes a band-information limitation of the
original shared encoder; it does not explain the actual score gap. Its proposed
single revision remains behind budget, durable implementation and prefit gates.

Provenance: native_development_comparison_v1.json lists the exact eight
prediction paths; development-reconstruction-review-final.json binds those
bytes, source and protocol. development-comparison-result-v1.json preserves
all exact support, source hashes, geometry, variation and resampling evidence.
development-comparison-execution-v1.log records the closed execution. Historical
evidence, approved r3, the live owned-process ledger and app/release boundaries
are preserved.
