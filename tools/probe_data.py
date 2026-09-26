"""Bounded HTTPS access probe. Never downloads a full archive or bypasses login."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def validate_url(url: str, allowed_hosts: set[str]) -> None:
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in allowed_hosts
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in (None, 443)
    ):
        raise ValueError("Only allowlisted HTTPS hosts on port443 without credentials are allowed.")


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def __init__(self, allowed_hosts: set[str]):
        super().__init__()
        self.allowed_hosts = allowed_hosts

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_url(newurl, self.allowed_hosts)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def probe(url: str, allowed_hosts: set[str], *, timeout: float = 8.0) -> dict[str, Any]:
    validate_url(url, allowed_hosts)
    started = time.monotonic()
    request = urllib.request.Request(
        url,
        headers={"Range": "bytes=0-65535", "User-Agent": "marine-echo-jepa-access-probe/1.0"},
    )
    result: dict[str, Any] = {"url": url, "full_download_verified": False}
    try:
        opener = urllib.request.build_opener(SafeRedirect(allowed_hosts))
        with opener.open(request, timeout=timeout) as response:
            sample = response.read(65536)
            result.update(
                status="PREFIX_ACCESSIBLE_NOT_FULL_DOWNLOAD",
                http_status=response.status,
                content_type=response.headers.get("Content-Type"),
                content_length_header=response.headers.get("Content-Length"),
                sampled_bytes=len(sample),
                prefix_sha256=hashlib.sha256(sample).hexdigest(),
            )
            if url.endswith(".zip") and not sample.startswith(b"PK"):
                result["status"] = "UNEXPECTED_ARCHIVE_PREFIX"
    except urllib.error.HTTPError as exc:
        result.update(status="HTTP_ERROR", http_status=exc.code, error=str(exc.reason))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        # Never log response bodies, cookies or credentials.
        result.update(status="ACCESS_NOT_VERIFIED", error=str(exc)[:400])
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/datasets.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/preflight/data_access.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    hosts = set(config["allowed_hosts"])
    primary, fallback = config["datasets"]
    urls = [primary["catalog_url"], primary["download_items"][0]["url"],
            primary["download_items"][1]["url"], fallback["inventory_url"]]
    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Prefix/metadata access only; not parsing, calibration or full download.",
        "results": [probe(url, hosts) for url in urls],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if all(r["status"] == "PREFIX_ACCESSIBLE_NOT_FULL_DOWNLOAD" for r in report["results"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
