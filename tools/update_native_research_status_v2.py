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
             "CF seeds13/23 are approved under67 exact bindings; seed13 is executing, seed23 follows serially.",
             "Their strong endpoints require separately bound actual completed-parent review.",
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
             "All prior scientific sources unchanged. Budget clarification and source-prefit remain pending.",
             "The proof does not identify the cause of the real score gap. No revision fit is authorized yet.",
             "The proposal explicitly lists11 total seed7 recipes versus three A/B configurations;",
             "the reviewer requires owner clarification of the finite-screen interpretation before revised training.",
             "No history24/latent128/extra architecture or final numerical access is approved.", "",
             "The20-method DEV manifest is frozen; distinct source/admission review is in progress.",
             "Held-out executor builder reports92 synthetic CPU checks; source integration/review pending.",
             "Finalist and matched-direct seeds13/23 remain required; no extra architecture search is approved.",
             "Next: complete approved replications, resolve the optional revision gate, then freeze",
             "bounded finalists and evaluation before held-out AEON2 assessment. The bounded builder is",
             "implementing a separate held-out executor with strict ancestor and native230m/dropout guards.",
             "Complete the model-only package, provenance, independent result review and scientific report.",
             "Cloud disabled; local paid spend0; software/experiment/JEPA-value/business gates remain separate.",
             "", "All historical status content follows intact.", ""]
    path.write_text("\n".join(lines) + "\n" + historical, encoding="utf-8")
    print(json.dumps({"status": "UPDATED", "ledger_written": False, "active": active}))


if __name__ == "__main__":
    main()
