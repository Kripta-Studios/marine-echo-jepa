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
             "CF strong frozen/full endpoints have separate completed-parent89-binding prefit approval, pending serial execution.",
             "Other strong control endpoints require their completed-parent review before fitting.", "",
             "Exact eight-method DEV reconstruction passed all58 bindings and actual execution exit0.",
             "Common support:7,593 issued;7,581/7,557/7,521 targets;316 eligible dates per horizon.",
             "Recomputed daily pinball: LightGBM0.505274; CF short probe0.745727; direct0.765185;",
             "shared full0.772594; shared frozen0.785715; shared short0.824647;",
             "persistence0.989176; seasonal1.031738dB. Short probes are not strong matched endpoints.",
             "All forecasts vary across issuance. Paired2000-draw48-nominal-hour DEV intervals favour LightGBM.",
             "These conditional single-deployment development results do not establish JEPA-value or SOTA.",
             "A distinct actual numerical reconstruction is in progress; final-test evidence is NOT_RUN.", "",
             "One parameter-free per-channel GELU revision is proposed under ADR0020 after a synthetic",
             "frequency-cancellation proof. Its code, budget interpretation and source-prefit remain pending.",
             "The proof does not identify the cause of the real score gap. No revision fit is authorized yet.",
             "The proposal explicitly lists11 total seed7 recipes versus three A/B configurations;",
             "the reviewer must resolve the finite-screen interpretation before any revised training.",
             "No history24/latent128/extra architecture or final numerical access is approved.", "",
             "Next: finish serial controls/Chronos, execute separately approved CF endpoints, review strong",
             "controls/revision, freeze bounded finalists and evaluation before held-out AEON2 assessment.",
             "Complete the model-only package, provenance, independent result review and scientific report.",
             "Cloud disabled; local paid spend0; software/experiment/JEPA-value/business gates remain separate.",
             "", "All historical status content follows intact.", ""]
    path.write_text("\n".join(lines) + "\n" + historical, encoding="utf-8")
    print(json.dumps({"status": "UPDATED", "ledger_written": False, "active": active}))


if __name__ == "__main__":
    main()
