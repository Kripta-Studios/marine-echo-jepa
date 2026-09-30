"""Refresh current research status without writing the active run ledger."""

import datetime
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / "orchestration/STATUS.md"
    original = path.read_text(encoding="utf-8")
    historical = original
    if original.startswith("## Active model-first research"):
        for marker in ("All historical status content follows intact.\n\n",
                       "All prior STATUS content follows intact.\n\n"):
            if marker in original:
                historical = original.split(marker, 1)[1]
                break
        else:
            raise ValueError("Ambiguous historical status boundary")
    ledger = json.loads((ROOT / "orchestration/native_ssl_run_ledger_v1.json").read_text(encoding="utf-8"))
    queue_path = ROOT / "evidence/ssl-research-v1/approved-queue-attempt-01/queue.json"
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    active = [run["id"] for run in ledger["runs"]
              if run.get("status") in ("RUNNING_CUDA", "RUNNING_CPU_FIT")]
    lines = ["## Active model-first research — native acoustic SSL",
             "", f"Updated {datetime.datetime.now(datetime.UTC).isoformat()}.", "",
             "Branch `research/marine-jepa-vnext`; owner-authorized local model research.",
             "App/release work is frozen; historical protocols, evidence and approved r3 are preserved.",
             "AEON2 whole-site final-test values remain unopened. Native geometry is retained.",
             "TRAIN18,593 issued/18,312 SSL eligible; DEV7,593 issued at native0–225m.",
             "Three distinct verified GPT-6.1 Sol/high sessions perform coordination, implementation and review.",
             "Builder scratch/Git and junction restrictions have not been bypassed.", "",
             "Completed real SSL: shared6000+2000 probes, CF6000+2000, masked6000+2000.",
             "Shared selected4500; CF/masked selected1500. Real safe weights and context-only inference exist.",
             "Strong shared frozen2000/full3000 and matched scratch supervised3000 completed.",
             "LightGBM15 quantile boosters and deterministic persistence/seasonal references completed.",
             "Shared/CF encoder CPU replay and LightGBM all15-booster CPU replay passed.",
             "Independent learning-path audit found no defect within inspected scope.", "",
             f"Active scientific job: {', '.join(active) or 'none recorded'}.",
             f"Original serial queue: {queue['status']}; {len(queue['jobs'])}/9 reports inspected.",
             f"Charged owned GPU-hours at ledger snapshot: {ledger.get('gpu_hours_spent_owned_scientific_jobs', 0):.6f}/96.",
             "The active wrappers own ledger writes; coordinator does not overwrite live accounting.",
             "All20 seed7 endpoints and controls completed; each serial queue exited0 with owned cleanup.",
             "CF strong frozen/full DEV0.614422/0.541814; masked frozen/full0.786842/0.691806dB.",
             "Permuted/random/supervised frozen DEV0.854690/0.812868/0.776118dB.",
             "CF seeds13/23 completed under67 exact bindings; queue exited0 with owned cleanup.",
             "Short-probe DEV seed7/13/23:0.745727/0.925193/0.714585dB; no best-seed selection.",
             "Seed13 strong frozen/full have separate93-binding completed-parent approval.",
             "First frozen13 attempt exited native0xc000070a after164.14s with no checkpoint/report; cause unknown.",
             "Exact captured dead-owner lock reconciled; bounded CUDA smoke passed34.172s full-owned.",
             "Fresh-output exact-recipe retry completed: frozen13 DEV0.668064; full13 DEV0.558057dB.",
             "Original native failed artifacts preserved; retry queue exited0 with verified owned cleanup.",
             "Seed23 strong endpoints completed under96 exact bindings: frozen DEV0.602785/full0.576860dB.",
             "All six CF strong endpoints across7/13/23 verified; raw DEV reports trail LightGBM0.505274dB.",
             "Shared scratch supervised13/23 completed under76 bindings, queue exited0 with owned cleanup.",
             "Direct DEV seed7/13/23:0.765185/0.741446/0.765869dB, all preserved without seed selection.",
             "Original Chronos zero-shot comparator completed475 shards/7,593 rows at0.506916dB DEV pinball;",
             "its external pretraining ancestry remains unknown and no clean local-ancestry guarantee follows.", "",
             "Exact eight-method DEV reconstruction passed all58 bindings and actual execution exit0.",
             "Common support:7,593 issued;7,581/7,557/7,521 targets;316 eligible dates per horizon.",
             "Recomputed daily pinball: LightGBM0.505274; CF short probe0.745727; direct0.765185;",
             "shared full0.772594; shared frozen0.785715; shared short0.824647;",
             "persistence0.989176; seasonal1.031738dB. Short probes are not strong matched endpoints.",
             "All forecasts vary across issuance. Paired2000-draw48-nominal-hour DEV intervals favour LightGBM.",
             "These conditional single-deployment development results do not establish JEPA-value or SOTA.",
             "Distinct actual numerical reconstruction verified all eight scores/7,584 daily losses exactly;",
             "bootstrap sequence identical and interval endpoints within1.11e-16dB. Final-test evidence NOT_RUN.", "",
             "One parameter-free per-channel GELU revision is proposed under ADR0020 after a synthetic",
             "frequency-cancellation proof. Eight authored source/tests integrated; root83 synthetic CPU checks",
             "passed including real optimizers, exact resume, frozen/updated weights and inference replay.",
             "Six initial virtual-export fixture failures are retained; only synthetic transport was repaired.",
             "All prior scientific sources unchanged. Owner resolved the band budget under ADR0023:",
             "11 total seed7 recipes including controls; all band operations capped12 GPU-hours inside96.",
             "The proof does not identify the cause of the real score gap. Exact source-prefit remains required.",
             "Band executor integrated;96 root policy checks passed163.70s and exclusive config generation exited0.",
             "Five exact per-job band prefits approved:195 approval bindings plus16 engineering-proof bindings verified.",
             "Approved five-job serial band queue launched, shared SSL first, within12/96 full-owned GPU hours.",
             "Band shared SSL completed6000+2000 real updates, selected4500, DEV0.789188dB.",
             "Its wrapper exited0 with owned cleanup;1.243043 full-owned GPU-hours charged, real weights preserved.",
             "Reusable latent inference source approved by distinct reviewer;40 exact bindings verified.",
             "Root89 synthetic CPU checks and actual Unicode disk replay passed for shared/CF/band.",
             "Real64-context CPU replay approved after metadata transport recovery;96 bindings verified.",
             "Actual Shared7/CF7/13/23 encoder and saved latent-head CPU replay passed, CLI exited0.",
             "No history24/latent128/extra architecture or final numerical access is approved.", "",
             "The20-method DEV reconstruction completed under150-binding admission; distinct numerical",
             "review verified20 scores/18,960 daily losses, all19 paired intervals and common support.",
             "Shared-vs-random strong frozen paired interval includes0; JEPA representation value is unproven.",
             "Held-out executor integrated; original native split/indexedCUDA repairs and115 CPU checks passed.",
             "Distinct assessment software/16-node ancestry review approved; all244 exact bindings verified.",
             "976 encoder tensors and61,500 TRAIN batches independently checked; scientific execution not authorized.",
             "Prefix delivery integrated; root102 actual CPU checks passed including six optimizer/resume cases.",
             "Root source-interval repair accepts native55–65minute adjacency and exact ordinal horizon IDs.",
             "Unicode durable prefix synthetic CPU fit/replay passed after explicit evidence-label repair.",
             "Prefix v2 typed supervised parents and structural gap guards integrated;171 root CPU checks passed.",
             "Eight actual optimizer/resume checks and new Unicode saved-artifact CPU fit/replay passed.",
             "Native150/180 configuration maps and nominal missing-ID compatibility are active builder work.",
             "Actual assessment resource-worker synthetic CPU smoke exited0 and removed its exclusive lock.",
             "CF backbone extra controls remain proposed/unfitted and deferred outside authorized11 recipes.",
             "Finalist and matched-direct seeds13/23 remain required; no extra architecture search is approved.",
             "Next: complete approved replications and independently approved band fits, then freeze",
             "bounded finalists and evaluation before held-out AEON2 assessment. The integrated executor",
             "has strict ancestor/native230m/dropout guards; software approved and final numeric access unopened.",
             "Complete the model-only package, provenance, independent result review and scientific report.",
             "Cloud disabled; local paid spend0; software/experiment/JEPA-value/business gates remain separate.",
             "", "All historical status content follows intact.", ""]
    path.write_text("\n".join(lines) + "\n" + historical, encoding="utf-8")
    print(json.dumps({"status": "UPDATED", "ledger_written": False, "active": active}))


if __name__ == "__main__":
    main()
