"""Read four small AZFP XML/DPL ZIP members by verified HTTP range."""

from __future__ import annotations

import hashlib
import json
import struct
import zlib
from pathlib import Path

import requests

from tools.v2_companion_probe import URL, get_range

NAMES = {"20012206.XML", "20080307.XML", "20012206.DPL", "20080307.DPL"}
OUT = Path("data/raw/pangaea/mosaic_azfp_up_2020_metadata")
REPORT = Path("evidence/v2/companion_document_inventory.json")


def main() -> None:
    head = requests.head(URL, timeout=30)
    head.raise_for_status()
    size = int(head.headers["Content-Length"])
    tail = get_range(size - 131072, size - 1, size)
    zip64 = tail.rfind(bytes([80, 75, 6, 6]))
    if zip64 < 0:
        raise ValueError("ZIP64 record absent")
    record = struct.unpack_from("<4sQ2H2L4Q", tail, zip64)
    count, directory_size, directory_offset = record[7:10]
    if directory_size > 1024 * 1024 or count != 5015:
        raise ValueError("Archive directory changed")
    directory = get_range(directory_offset, directory_offset + directory_size - 1, size)
    cursor = 0
    found = {}
    while cursor < len(directory):
        fields = struct.unpack_from("<4s6H3L5H2L", directory, cursor)
        if fields[0] != bytes([80, 75, 1, 2]):
            raise ValueError("Central directory malformed")
        name_length, extra_length, comment_length = fields[10:13]
        name_start = cursor + 46
        name = directory[name_start : name_start + name_length].decode("utf-8")
        if name in NAMES:
            found[name] = {
                "method": fields[4],
                "crc32": fields[7],
                "compressed_bytes": fields[8],
                "uncompressed_bytes": fields[9],
                "local_header_offset": fields[16],
            }
        cursor = name_start + name_length + extra_length + comment_length
    if cursor != len(directory) or set(found) != NAMES:
        raise ValueError("Expected four metadata members not found")
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"source_url": URL, "archive_bytes": size, "documents": {}, "acoustic_payload_bytes_read": 0}
    for name in sorted(NAMES):
        entry = found[name]
        if entry["compressed_bytes"] > 4096 or entry["uncompressed_bytes"] > 65536:
            raise ValueError("Metadata member unexpectedly large")
        offset = entry["local_header_offset"]
        header = get_range(offset, offset + 29, size)
        local = struct.unpack("<4s5H3L2H", header)
        if local[0] != bytes([80, 75, 3, 4]) or local[3] != entry["method"]:
            raise ValueError("Local member header differs")
        name_len, extra_len = local[-2:]
        actual_name = get_range(offset + 30, offset + 29 + name_len, size).decode("utf-8")
        if actual_name != name:
            raise ValueError("Local member name differs")
        begin = offset + 30 + name_len + extra_len
        compressed = get_range(begin, begin + entry["compressed_bytes"] - 1, size)
        if entry["method"] == 8:
            payload = zlib.decompress(compressed, -15)
        elif entry["method"] == 0:
            payload = compressed
        else:
            raise ValueError("Unsupported metadata compression")
        if len(payload) != entry["uncompressed_bytes"] or zlib.crc32(payload) != entry["crc32"]:
            raise ValueError("Metadata CRC or size differs")
        path = OUT / name
        with path.open("xb") as stream:
            stream.write(payload)
        report["documents"][name] = {
            **entry,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "path": str(path),
        }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
