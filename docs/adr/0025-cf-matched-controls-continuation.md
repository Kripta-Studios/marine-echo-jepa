# ADR0025: Owner-adopted CF matched-control extension

Status: OWNER_AUTHORIZED_SCOPE; exact source/prefit approval remains required.
Date: 2026-10-01.

The owner adopted `native_ssl_continuation_dd012f0/CODEX_CONTINUE_NATIVE_SSL.md`
and explicitly authorized a separate `CF_MATCHED_CONTROLS` study. This decision
adds to ADR0022; it does not modify that historical proposal, ADR0023, the Band
eleven-recipe reservation, fitted sources, or the original 43-neural/47-method
campaign. ADR0022's older unresolved screen-count discussion does not govern
this separately authorized extension.

Default fits are exactly six: CF scratch supervised and CF random-frozen strong
readout at seeds 7, 13 and 23. Respectively use 3,000/750 and 2,000/500
updates/checkpoint cadence. There is no new SSL pretraining, backbone change,
history/crop change, hyperparameter search or model selection on reserved data.
The extension has a separate maximum 12 full-owned GPU-hours inside the unchanged
aggregate 96-hour cap. Historical failure charges remain counted. At least
12 aggregate hours must remain reserved for transfer/evaluation/review replay.
Band charges and its 12-hour sub-cap are neither reset nor expanded.

Reuse the existing CF architecture and its strong endpoint feature semantics:
the forecasting encoder is the selected EMA branch in the trained comparator;
controls construct the corresponding branch from fresh initialization. All arms
consume full H96 observed native contexts and query metadata. This preserves the
documented mismatch between full-H96 inference and sampled SSL training crops;
it does not repair one arm. Match the existing query head, seed+100000 head
construction, TRAIN scalers, masks, supervised sample stream, optimizer schedule,
checkpoint selection and target inverse transform. Prove initial head equality
against the existing source factory; seed equality alone is insufficient.

Random-frozen controls must retain every encoder parameter and buffer exactly;
scratch supervised controls must update encoder parameters. Neither executes
SSL losses, future crops, predictor optimization or EMA updates. Distinct artifact
kinds and zero SSL update counts prevent controls from being labelled pretrained
SSL encoders. Random controls still have fitted readout and scaler ancestry.

Implementation and synthetic CPU verification may proceed while independent
review is unavailable. Real fits require an actual distinct
`APPROVED_CF_CONTROL_PREFIT`, exact source/data/config/runtime/resource bindings,
and the guarded root executor. Owner scope authorization is not that approval.
The final extension manifest must remain separate from the original 47 methods.
Final source access and adaptation policies remain separate gates.
