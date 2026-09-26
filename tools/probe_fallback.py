"""Bounded metadata-only OOI discovery after recorded primary calibration failure."""

import json
from datetime import datetime, timezone
from pathlib import Path

from probe_data import probe


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    if not (root / "docs/decisions/0002-calibration-block-and-fallback.md").is_file():
        raise RuntimeError("Documented fallback decision is required.")
    urls = [
        "https://rawdata.oceanobservatories.org/files/CE04OSPS/PC01B/ZPLSCB102/2017/08/",
        "https://rawdata.oceanobservatories.org/files/CE04OSPS/PC01B/ZPLSCB102/2017/08/21/",
    ]
    results = [probe(url, {"rawdata.oceanobservatories.org"}, timeout=15) for url in urls]
    report = {"checked_at": datetime.now(timezone.utc).isoformat(), "basis": "ADR0002", "binary_downloads": 0, "active_dataset_unchanged": True, "results": results}
    destination = root / "evidence/data/fallback_access.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if all(r["status"] == "PREFIX_ACCESSIBLE_NOT_FULL_DOWNLOAD" for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
