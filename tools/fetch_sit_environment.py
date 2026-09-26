"""Bounded downloads of the four recovered MOSAiC SIT records for applicability audit.

This extends environmental provenance only; it does not switch acoustic datasets,
claim co-location, open acoustic test values, or authorize physical calibration.
"""

import hashlib
import json
import os
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from continuation_records import write_report

ROOT = Path(__file__).resolve().parents[1]
RECORDS = {"2019O1": 940271, "2019O3": 940282, "2019O4": 940291, "2019O6": 940296}
CAP = 160 * 1024**2


def main() -> int:
    folder = ROOT / "data/raw/environment/mosaic_sit"
    folder.mkdir(parents=True, exist_ok=True)
    inventory = ROOT / "evidence/continuation/sit_inventory.json"
    old = json.loads(inventory.read_text()) if inventory.exists() else {"files": []}
    rows = []
    for buoy, identifier in RECORDS.items():
        path = folder / f"{identifier}-{buoy}.tsv"
        url = f"https://doi.pangaea.de/10.1594/PANGAEA.{identifier}?format=textfile"
        started = time.monotonic()
        row = {"buoy": buoy, "doi": f"10.1594/PANGAEA.{identifier}", "url": url}
        try:
            if path.exists():
                prior = next((item for item in old["files"] if item.get("buoy") == buoy), None)
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if prior is None or prior.get("sha256") != digest:
                    raise ValueError("Existing unregistered or changed source; no overwrite.")
                rows.append(prior)
                print(json.dumps({"buoy": buoy, "status": "REUSED_VERIFIED"}), flush=True)
                continue
            temporary = path.with_suffix(".partial")
            if temporary.exists():
                raise FileExistsError("Preserved partial download requires reconciliation.")
            with urllib.request.urlopen(
                urllib.request.Request(url, method="HEAD"), timeout=30
            ) as header:
                expected = int(header.headers["Content-Length"])
                if not 0 < expected <= CAP:
                    raise ValueError("Source exceeds the fixed 160 MiB per-record limit.")
            with urllib.request.urlopen(url, timeout=30) as response:
                if "tab-separated-values" not in response.headers.get("Content-Type", ""):
                    raise ValueError("Expected the publisher's tab-separated source.")
                count = 0
                digest_state = hashlib.sha256()
                with temporary.open("xb") as stream:
                    while block := response.read(1024**2):
                        count += len(block)
                        if count > CAP or time.monotonic() - started > 300:
                            raise ValueError("Download exceeded the byte/time bound.")
                        stream.write(block)
                        digest_state.update(block)
                    stream.flush()
                    os.fsync(stream.fileno())
                if count != expected:
                    raise ValueError("Source length differs from HTTP metadata.")
            # Single writer, preserved original bytes; never replace a prior source.
            temporary.rename(path)
            row.update(
                status="DOWNLOADED_UNVALIDATED_APPLICABILITY",
                path=path.relative_to(ROOT).as_posix(),
                bytes=count,
                sha256=digest_state.hexdigest(),
                fetched_at=datetime.now(UTC).isoformat(),
                license="CC-BY-4.0",
                seconds=time.monotonic() - started,
            )
        except (OSError, ValueError, KeyError) as error:
            row.update(status="ACCESS_OR_LIMIT_BLOCKED", error=f"{type(error).__name__}: {error}")
        rows.append(row)
        write_report(
            inventory,
            {
                "files": rows,
                "selection": "All four recovered SIT units; no outcome selection",
                "benchmark_eligible": False,
            },
        )
        print(json.dumps(row), flush=True)
    write_report(
        inventory,
        {
            "files": rows,
            "selection": "All four recovered SIT units; no outcome selection",
            "benchmark_eligible": False,
        },
    )
    return 0 if all("sha256" in item for item in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
