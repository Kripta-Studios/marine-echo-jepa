"""Refresh coordinator-owned current status while retaining historical text."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    path = ROOT / "orchestration/STATUS.md"
    text = path.read_text(encoding="utf-8")
    if text.startswith("## Active model-first research"):
        for boundary in ("All historical status content follows intact.\n\n",
                         "All prior STATUS content follows intact.\n\n"):
            _, separator, after = text.partition(boundary)
            if separator:
                text = after
                break
        else:
            raise ValueError("Current research status boundary is ambiguous.")
    current = """## Active model-first research — native acoustic SSL, 30 September 2026

The owner activated the new model-first mission on `research/marine-jepa-vnext`.
App/release work is frozen. Historical protocols, evidence and approved r3 remain
preserved. The current research authorization supersedes historical meeting-only
and diagnostic-only priorities for this separately versioned study.

Seven exact local archives are reserved before new model scores: AEON2 whole-site
final assessment; the latest AEON4 deployment for development; earlier AEON4 and
declared AEON3 TRAIN dates for fitting. Distinct numerical-access review approved
the repaired native-geometry contract. Real materialization completed with18,593
TRAIN windows (18,312 SSL eligible) and7,593 development windows, peak RSS4.04GiB.
Final-test values remain unopened. New weights/scalers start without fitted
ancestors. Native products retain their actual200/220/225/230m bounds.

Shared temporal SSL, source-grounded author-zone CF-JEPA, matched direct and
masked SSL, random and pairing-permuted controls are implemented. The shared
model has2,457,621 parameters. Parent verification passed34 durable model/runner
tests and22 data/metric/reference tests; the subsequent operational repair suite
passed49 checks including native inference and process ownership. Exact first-screen schedules/configs
are frozen under ADR0016 with four matched checkpoint opportunities and bounded
readout updates. The distinct model-prefit review is running; scientific fitting
and numerical model scoring are NOT_RUN until approval. A fixed Windows desktop
allowlist repairs the WDDM guard; strict baseline scoring now rejects nonfinite
outputs instead of reducing assessment support. These changes require renewed
exact prefit bindings. A bounded builder is implementing separately accounted
strong frozen readouts and supervised fine-tuning. Actual primary, builder
and reviewer runtime is GPT-6.1 Sol/high in three distinct sessions. Builder
scratch/Git denials are retained; no restricted path or denied junction action
was bypassed. Coordinator integration used new paths in its own checkout.

Next: execute the first approved shared SSL CUDA trajectory, then matched real
comparisons, stronger frozen readouts, held-out transfer and scientific package.
This is an active research programme, not another app or meeting deliverable.
Ledger: `orchestration/native_ssl_run_ledger_v1.json`; evidence:
`evidence/ssl-research-v1/`. No new trained model, result or SOTA claim yet.

All historical status content follows intact.

"""
    path.write_text(current + text, encoding="utf-8")


if __name__ == "__main__":
    main()
