"""Refresh coordinator-owned current status while retaining historical text."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    path = ROOT / "orchestration/STATUS.md"
    text = path.read_text(encoding="utf-8")
    if text.startswith("## Active model-first research"):
        for boundary in (
            "All historical status content follows intact.\n\n",
            "All prior STATUS content follows intact.\n\n",
        ):
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
readout updates. The distinct model-prefit review approved the six bound screens
and deterministic references; all73 exact bindings passed. A fixed Windows desktop
allowlist repairs the WDDM guard; strict baseline scoring now rejects nonfinite
outputs instead of reducing assessment support. These renewed bindings are now
approved. The first CUDA attempt exited1 before optimizer updates because its
device needed an explicit index. The preserved retry uses unchanged approved
code/configuration with `cuda:0`. This real job completed exit0:6000 SSL updates
and2000 total frozen-probe updates. Final selected encoder step4500, daily
development pinball0.8246468454704822, peak reserved0.2305GiB and RSS4.821GiB,
synchronized device-owned elapsed3371.063seconds. Real pretrained weights,
resumable checkpoints and reusable inference exist. Controls and strong readouts
are needed before any JEPA-value claim. Completed
deterministic development references: persistence0.989176 and seasonal24
1.031738dB pinball. These are development values, not final-test results.
The bounded downstream builder delivered37 CPU checks; coordinator reproduced
and repaired4 contract issues, then passed41 durable CPU checks. The downstream
CLI device index fix separately passed its focused check. Strong downstream
fitting remains NOT_RUN. CF first backward failed before optimizer updates on
deterministic CUDA adaptive pooling. LightGBM failed saving its first booster
under the Unicode checkout. Both failures are preserved. The supervisor's former
launcher-only RAM observation missed the Windows interpreter child; a real tree
test reproduced the defect and five checks now pass with descendant monitoring
and owned-tree termination. The bounded platform builder delivered equivalent
pooling and Python UTF-8 booster serialization. Parent127 CPU checks and34 core
regression checks passed; actual deterministic CUDA backward replay passed with
zero optimizer updates. Distinct neural reviewV3 approved the remaining five
screens and all104 exact bindings passed. The CF retry is running through the
bound indexed-device wrapper and owned-tree supervisor in a new output directory.
Comparator renewalV3 approved LightGBM and local Chronos2, with130 exact bindings
verified. Separate downstream prefit approved the actual shared encoder's fresh
2000-update frozen readout and3000-update full fine-tuning, plus a matched
3000-update supervised trajectory from scratch; all83 bindings passed. These
strong endpoint fits remain NOT_RUN while CF owns the scientific process.
The new encoder-only API passed81 parent CPU checks and reproduced the actual
shared pretrained features bit-identically on64 real development prefixes using
only context values, masks and native metadata; a separate inference review is
running. CF's second candidate completed3000 SSL updates and its second500-update
probe, with interim best still step1500 at0.745726589903934 development pinball.
Actual primary, builder
and reviewer runtime is GPT-6.1 Sol/high in three distinct sessions. Builder
scratch/Git denials are retained; no restricted path or denied junction action
was bypassed. Coordinator integration used new paths in its own checkout.

Next: complete the reviewed CF-JEPA trajectory and matched real
comparisons, stronger frozen readouts, held-out transfer and scientific package.
This is an active research programme, not another app or meeting deliverable.
Ledger: `orchestration/native_ssl_run_ledger_v1.json`; evidence:
`evidence/ssl-research-v1/`. Real SSL checkpoints exist; a completed comparative
model package, held-out result and SOTA claim remain unestablished.

All historical status content follows intact.

"""
    ledger = json.loads((ROOT / "orchestration/native_ssl_run_ledger_v1.json").read_text())
    active = [
        record
        for record in ledger["runs"]
        if record.get("status") in ("RUNNING_CUDA", "RUNNING_CPU_FIT")
    ]
    live = (
        "Current scientific process: "
        + (
            "; ".join(
                f"{record['id']} ({record['status']}, owned launcher PID{record.get('pid')})"
                for record in active
            )
            if active
            else "none; all prior attempts journalled"
        )
        + ".\n\n"
    )
    current = current.replace(
        "All historical status content follows intact.\n\n",
        live + "All historical status content follows intact.\n\n",
    )
    path.write_text(current + text, encoding="utf-8")


if __name__ == "__main__":
    main()
