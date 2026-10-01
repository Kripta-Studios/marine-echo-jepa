# Native acoustic SSL: seven pretrained models and remaining gates

Recorded on 2026-10-01 after the completed-run checkpoint in
[V3](NATIVE_SSL_MODEL_RESEARCH_REPORT_ADDENDUM_V3.md). Seven real pretrained
encoders, their latent inference weights, TRAIN scalers and reusable source are
now packaged and have passed local isolated CPU correctness checks. The study
remains incomplete. Whole-site comparisons, public prefix transfer and three
planned training endpoints are NOT_RUN; SOTA is not established.

The local deliverable is
[`native_pretrained_model_snapshot_v2.zip`](../outputs/native_pretrained_model_snapshot_v2.zip),
109,616,204 bytes, SHA-256
`22f116a426dbba384610b0bd60e9d450660dcf50d4111a56040989dd4e2d2d48`.
It contains the original Shared7 encoder, CF-JEPA7/13/23 encoders, and native
Band shared-temporal SSL7/13/23 encoders. Every endpoint is retained; no best seed
was substituted. All97 ZIP entries match both their recorded hashes and the
copied-directory bytes. The original four-model snapshot remains byte-identical.
The approved r3 archive and earlier scientific evidence are preserved. This is
a research weights snapshot, with no app or release work.

The archive's `MODEL_CARD.md` documents the load-only APIs and their semantics.
Selected encoders and saved latent predictors are both available, with exact
TRAIN scalers. Shared models accept H96 raw-dB contexts, observation masks and
native measurement/query metadata. Their forecast outputs are latent blocks,
not acoustic-dB forecasts. The CF selected encoder uses EMA features; CF ONLINE
heads predict ordinal zones, and full H96 inference lies outside sampled
training crops. These limitations remain material to representation assessment.
The short development selection probes do not replace strong frozen readouts.

Root contract checks passed54 tests in15.22 seconds. Production packaging exited
zero. Isolated copied-source CPU checks exited zero, completed in4.125 seconds,
used628,912,128 bytes of peak owned RAM and verified owned-process cleanup.
All seven selected encoders reproduced their latent artifact encoder outputs
bit-identically. Imports resolved to copied source; CPU RNG remained unchanged
and CUDA was not initialized. Native0–230m was accepted and its relabeling as
0–200m was rejected for all applicable shared models. CF ordinal zones do not
provide that native-horizon query interface.

These were synthetic-input correctness checks using real pretrained weights;
they were not public-data transfer or performance measurements. Independent
package/source compatibility review is NOT_RUN. The immutable package card and
creation receipt retain their original QA-pending timestamp; the external
[closeout](../evidence/ssl-research-v1/native-pretrained-snapshot-closeout-v2.json)
records the subsequently completed checks.

There are40 completed neural endpoints, independently from the packaging count,
and21 fitted-parent links in the completed artifact inventory. Five additional
real jobs and their runner development scores are recorded in V3. The latest
independently reconstructed numerical evidence remains the28-method development
comparison in [V2](NATIVE_SSL_MODEL_RESEARCH_REPORT_ADDENDUM_V2.md). It favours
LightGBM over all three CF full fine-tuning seeds. Original shared-versus-random
and shared-versus-masking frozen comparisons do not establish the required
representation advantage. The expanded development comparison is NOT_RUN;
runner scores from new jobs cannot be promoted to independently verified or
whole-site results. No JEPA transfer advantage is established.

The fixed43-neural/47-method programme still lacks the Band supervised seed13
retry and Band seed23 strong frozen/full readouts. Distinct reviewer requests
for these prefits were automatically rejected before review with the stated
reason, "Potentially unintended activity." Native source/access reviews also
remain unavailable. Those rejections confer no scientific approval; no missing
fit or reserved-site evaluation was launched. The owner authorization remains
recorded, but it does not override the required independent reviews or safety
restrictions. The planned all-ancestor inventory tooling is implemented and
guarded against incomplete inputs, rather than represented as completed evidence.

AEON2 deployments61937278 and61937269 remain reserved; their numerical acoustic
values remain unopened. No public prefix head was fitted. Native geometry,
unknown source timezone and integrated-product eligibility remain as registered.
Nothing here establishes species, biomass, causal intervention or business value.

Closed full-owned GPU accounting remains8.318836 hours within the Band12-hour
cap and17.802356 hours within the aggregate96-hour cap, including failed attempts.
No scientific GPU owner or unreconciled journal is active. The unchanged original
GPU guard was used for the completed jobs; the proposed VLC exception policy
remains unapproved and unused. No foreign process was terminated, filesystem
restriction bypassed, cloud purchase made or credential changed.

The next scientific actions require genuine independent prefit/source/access
review, completion of the three missing endpoints, fair saved-development
comparison, all-ancestor selection freeze, public prefix transfer and reserved
whole-site evaluation. A final scientific review must assess those comparisons
before a performance claim. The current deliverable does not satisfy those gates.

Evidence: the snapshot creation and closeout receipts in
`evidence/ssl-research-v1`, `native-pretrained-snapshot-isolated-cpu-v3`, the
root54-test log under `evidence/ssl-pretrained-snapshot-builder-v2`, the closed
real-job receiptV5 and completed inventoryV5, and the preserved direct13/seed23
automatic rejection closeouts. The earlier128-test source/metadata suite and
its failure history remain in their original evidence paths.
