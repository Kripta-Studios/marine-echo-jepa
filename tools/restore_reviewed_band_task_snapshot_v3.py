"""Restore an exact reviewed task snapshot from retained reviewer events, preserving later edits."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / "evidence/ssl-research-v1"
    review = json.loads((folder / "band-downstream-prefit-compact-review-final-v3.json").read_bytes())
    target = ROOT / "orchestration/ssl_vnext_band_downstream_prefit_compact_verdict_v3.txt"
    expected = review["proof_bindings"][str(target)]
    matches = []
    current = target.read_text(encoding="utf-8")
    paragraphs = current.split("\n\n")
    for drop_canonical in (False, True):
        kept = [p for p in paragraphs if not p.startswith("Current compact-format root tooling")
                and not (drop_canonical and p.startswith("Canonical metadata transport"))]
        value = "\n\n".join(kept)
        for candidate in (value, value.rstrip("\n") + "\n", value.replace("\n", "\r\n")):
            raw = candidate.encode("utf-8")
            if hashlib.sha256(raw).hexdigest() == expected:
                matches.append(raw)
    for line in (folder / "band-downstream-prefit-compact-review-events-v3.jsonl").read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        item = event.get("item", {})
        if event.get("type") != "item.completed" or item.get("id") != "item_5":
            continue
        for block in item.get("result", {}).get("content", []):
            text = block.get("text", "")
            start = text.find("ROOT EXPLICITLY RESUMES SAME distinct reviewer")
            end = text.find('\n{\n  "status": "SECOND_PREFIT_REVIEW_CAPACITY_FAILURE_BEFORE_ARTIFACT"', start)
            if start < 0 or end < 0:
                continue
            value = text[start:end]
            for candidate in (value, value + "\n", value.rstrip("\n") + "\n"):
                raw = candidate.encode("utf-8")
                if hashlib.sha256(raw).hexdigest() == expected:
                    matches.append(raw)
    if not matches or any(raw != matches[0] for raw in matches):
        raise ValueError("Exact retained reviewer-read bytes could not be reconstructed")
    backup = folder / "band-compact-task-later-unreviewed-snapshot-v3.txt"
    with backup.open("xb") as stream:
        stream.write(target.read_bytes())
    target.write_bytes(matches[0])
    receipt = {"status": "EXACT_REVIEWED_TASK_RESTORED_FROM_RETAINED_EVENTS", "target": str(target),
               "restored_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
               "preserved_later_snapshot": str(backup), "scientific_changes": False}
    with (folder / "band-compact-task-restoration-receipt-v3.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
