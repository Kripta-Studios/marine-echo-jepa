# ADR0024: Prefix transfer uses native source interval identities

Status: IMPLEMENTED_ENGINEERING_PENDING_INDEPENDENT_PREFIT.

ADR0021 fixes source-calendar prefix boundaries before numerical selection. Its
executor must follow the existing native reader's adjacency rule: consecutive
source interval indices with timestamp differences from 55 through 65 minutes.
Source-centred timestamps retain the publisher clock; no UTC is invented.

The separately bound raw interval registry therefore includes a nonnegative
integer `source_interval_index` for each deployment/channel observation. Context
contains 96 consecutive indices, all four channels share each actual timestamp,
and the cutoff equals the last timestamp. Forecast targets must have the exact
cutoff index plus 1, 3 or 6, with the corresponding elapsed source-clock bounds.
Missing indices, index aliases, timestamp aliases, configuration changes and
invalid adjacency fail before numerical fitting. Nominal source horizons are
not renamed to exact wall-clock hours.

Registry timestamps and indices must come from the native source metadata. They
must never be reconstructed by assuming exact hourly centres. Native bounds and
the 165-ping quarantine remain unchanged. Prefix/context and suffix/context raw
identities and timestamps remain disjoint; calendar gaps alone are insufficient.

The closed builder snapshot is preserved. Root's focused checks first failed
on clock jitter and missing/wrong indices, then passed after this repair. The
expanded 102-check CPU suite passed with actual optimizer/resume cases. A new
Unicode durable synthetic fit and safe saved-weight replay passed after adding
the explicit evidence kind to completion receipts; its failed verification is
retained. These are engineering checks, not public-data training approval.

Real prefix fitting still requires separate exact numerical-access and distinct
prefit reviews, approved resource accounting and immutable ancestry/scaler/data
bindings. No reserved suffix values or final-site fitting are admitted here.
