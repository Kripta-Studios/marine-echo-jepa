# ADR 0018: separate native acoustic representation transfer endpoints

Date: 2026-09-30. Status: frozen before downstream fitting, subject to distinct
exact prefit review. This extends ADR0015/0016; historical protocols and first
screen evidence remain immutable. The first shared screen has completed, with
step4500 selected by its original scheduled short probes. No strong downstream
trajectory, final-test numerical access or comparative JEPA claim has occurred.

The 500-update probes select encoders; they do not establish representation
transfer. For each selected encoder, fit a fresh query-conditioned quantile head
for2000 TRAIN updates with frozen encoder parameters and buffers, evaluating at
500/1000/1500/2000. Separately train full-finetune endpoints and a fresh matched
direct end-to-end model for3000 updates, evaluating at750/1500/2250/3000.
These are additional finite trajectories, not continuations of selection probes.
The fresh direct model has no fitted ancestor. A direct-screen frozen encoder
may additionally be probed as a representation comparator; its ancestry remains
explicitly supervised. Random and pairing-permuted frozen controls are mandatory.

All shared-backbone heads use the same seed+100000 initializer and seed-specific
TRAIN sampling sequence, independent of pretraining RNG state. CF uses its
author128-dimensional EMA encoder and the correspondingly dimensioned head;
its head parameter count is disclosed. Configuration fixes batch64, AdamW
LR0.0003, weight decay0.0001, gradient clipping1,10% warmup and cosine decay
to10% peak, patience4 and at least18 anchors per native assessment day. The
earliest strictly improved scheduled daily development pinball wins. Only the
head is optimized in frozen mode; full mode updates its encoder and fresh head.
All supervision, label presentations, updates and additional compute are logged.

Scalers originate solely in admitted TRAIN data. A pretrained endpoint verifies
its selected encoder, original completed run, inference, membership, source
configuration and distinct ancestor review; it reuses the original TRAIN scaler.
All ancestor TRAIN deployment/archive/row membership is checked before fitting.
Final AEON2 site data remains closed and excluded from every ancestor. Native
200/220/225/230m products retain their actual geometry; development is225m.

A narrowly approved historical model source snapshot may retain an earlier
shared encoder's original source hash after a mathematically equivalent CF-only
pooling repair in the same module. The snapshot is byte-identical historical
source, bound in the new distinct review alongside current code. The reviewer
must explicitly authorize shared-backbone compatibility and inspect the unchanged
shared class definitions. The exception applies only to this model source; it
cannot substitute corpora, scalers, configuration, protocol, membership or other
helpers. Model tensor/config identities must still agree exactly with original
inference. Historical artifacts and reviews are never edited to match new code.

Initial downstream admission requests seed7/history96 shared frozen/full and
fresh direct endpoints. Every later parent encoder requires renewed artifact
bindings before its own downstream fitting. Finalist seeds7/13/23 and optional
history24 comparisons require separately bound approvals. At most two SSL
finalists are selected using development frozen-readout evidence; full-finetune
and direct performance remain separately reported endpoints. No repeated recipe
or unbounded sweep is authorized by this ADR.

One root supervisor serializes all scientific fitting and GPU comparisons.
The aggregate96 owned GPU-hour programme ceiling includes failed and resumed
attempts. Allocated/reserved CUDA memory remains below10GiB and process RAM
below22GiB, with monitoring of actual interpreter descendants on Windows.
CPU correctness fixtures are labelled synthetic and never scientific results.
Cloud, payment, credentials, app/release and filesystem/junction restrictions
remain unchanged. Strong readout success alone does not establish SOTA.
