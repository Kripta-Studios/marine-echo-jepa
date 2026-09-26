"""Bounded OOI fallback sample; never changes the primary protocol."""

import hashlib
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from probe_data import SafeRedirect


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    evidence = root / "evidence/data"
    opener = urllib.request.build_opener(SafeRedirect({"rawdata.oceanobservatories.org"}))
    base = "https://rawdata.oceanobservatories.org/files/CE04OSPS/PC01B/ZPLSCB102/2017/08/"
    month = opener.open(base, timeout=20).read(1000000)
    (evidence / "ooi_201708_listing.html").write_bytes(month)
    present_days = re.findall(r'href="(\d{2})/"', month.decode())
    records = []
    for day in ("20", "21", "22"):
        if day not in present_days:
            continue
        listing = opener.open(base + day + "/", timeout=20).read(1000000)
        (evidence / f"ooi_201708{day}_listing.html").write_bytes(listing)
        files = re.findall(r'href="([^"/]+\.raw)"', listing.decode())
        records.append({"day": "2017-08-" + day, "files": files})
    first = next(r for r in records if r["day"] == "2017-08-21")["files"][0]
    url = base + "21/" + first
    target = root / "data/raw/ooi/diagnostic" / first
    target.parent.mkdir(parents=True, exist_ok=True)
    limit = 64 * 1024**2
    if not target.exists():
        partial = target.with_suffix(".raw.part")
        with opener.open(url, timeout=30) as response, partial.open("wb") as output:
            count = 0
            if "html" in response.headers.get("Content-Type", "").lower():
                raise ValueError("Expected raw data, received HTML.")
            while chunk := response.read(1024**2):
                count += len(chunk)
                if count > limit:
                    raise ValueError("Diagnostic download exceeds 64 MiB cap.")
                if count == len(chunk) and chunk[4:8] != b"CON0":
                    raise ValueError("Not an EK60 CON0 raw file.")
                output.write(chunk)
        partial.rename(target)
    with target.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    report = {"checked_at": datetime.now(timezone.utc).isoformat(), "dataset_id": "ooi_ce04osps_ek60_2017", "purpose": "one_file_calibration_feasibility_only", "primary_dataset_unchanged": True, "source_url": url, "path": str(target.relative_to(root)), "bytes": target.stat().st_size, "sha256": digest, "adjacent_inventory": records, "benchmark_eligible": False, "reason": "One diagnostic file cannot establish full-day support or required chronological partitions."}
    (evidence / "ooi_diagnostic_inventory.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
