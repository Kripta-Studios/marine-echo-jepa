# ADR 0002: primary calibration blocked; bounded fallback discovery

Date: 2026-09-26. No model selection or held-out outcomes have been accessed.

Echopype 0.11.1 parsed `20021614.01A` with `20021600.XML`: 185 pings,
four expected channels, 999 sample positions. Calling compute_Sv without environmental
parameters fails. The XML supplies configured sound speed 1465 m/s, but verified
seawater conditions and absorption provenance are absent. The pressure-sensor flag
is false. Neither buoy-computer temperature nor undocumented defaults resolve this.

Independent R0 inspection found that the platform manual labels the downward AZFP
55169 while both archive XML/DPL pairs identify 55170. Preserve both statements.
Use the manual as platform context only, not instrument-specific calibration evidence.
Do not redistribute its operational credentials. Its local hash remains registered.

Per docs/02_DATASETS_ACCESS.md and docs/21_RISKS_FALLBACKS.md, this establishes the
calibration-block fallback criterion. Authorize bounded metadata access to the exact
documented OOI monthly/day routes, with no credential bypass and no automatic switch
of active dataset. A viable fallback requires verified access/calibration, adjacent
days, separate configuration and prospective split/protocol. A one-day sample is
not benchmark eligibility. All primary physical-unit experiments remain BLOCKED.

Continue software and a clearly labelled raw-count diagnostic replay. Raw sample
indices are not verified physical range/depth, and raw counts are never Sv or dB.
Missing experiments are NOT_RUN, not negative JEPA results.
