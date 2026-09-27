"""Bounded HTTP-range ZIP directory probe; never downloads acoustic members."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import requests

URL = "https://download.pangaea.de/dataset/953621/files/azfp55171-bioacoustics.zip"
MAX_TRANSFER = 2 * 1024 * 1024
OUT = Path("evidence/v2/companion_archive_metadata.json")


def get_range(start: int, end: int, size: int) -> bytes:
    if start < 0 or end < start or end >= size or end - start + 1 > MAX_TRANSFER:
        raise ValueError("Invalid bounded ZIP metadata range")
    response = requests.get(URL, headers={"Range": f"bytes={start}-{end}"}, timeout=30)
    if response.status_code != 206 or len(response.content) != end - start + 1:
        raise ValueError("Publisher did not serve the exact metadata range")
    if response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
        raise ValueError("Publisher range total differs from HEAD")
    return response.content


def main() -> None:
    head = requests.head(URL, timeout=30, allow_redirects=True)
    head.raise_for_status()
    size = int(head.headers["Content-Length"])
    tail_start = max(0, size - 131072)
    tail = get_range(tail_start, size - 1, size)
    zip64_index = tail.rfind(bytes([80, 75, 6, 6]))
    if zip64_index < 0:
        raise ValueError("ZIP64 directory missing in bounded tail")
    signature, record_size, made_by, needed, disk, cd_disk, entries_disk, count, cd_size, cd_offset = (
        struct.unpack_from("<4sQ2H2L4Q", tail, zip64_index)
    )
    if (
        signature != bytes([80, 75, 6, 6])
        or record_size != 44
        or disk != 0
        or cd_disk != 0
        or entries_disk != count
        or cd_size > MAX_TRANSFER
        or cd_offset + cd_size > size
    ):
        raise ValueError("Unexpected ZIP64 directory shape")
    directory = get_range(cd_offset, cd_offset + cd_size - 1, size)
    members = []
    cursor = 0
    while cursor < len(directory):
        if directory[cursor : cursor + 4] != bytes([80, 75, 1, 2]):
            raise ValueError("Invalid ZIP central directory entry")
        fields = struct.unpack_from("<4s6H3L5H2L", directory, cursor)
        compressed = fields[8]
        uncompressed = fields[9]
        name_length, extra_length, comment_length = fields[10:13]
        name_start = cursor + 46
        name_bytes = directory[name_start : name_start + name_length]
        if len(name_bytes) != name_length:
            raise ValueError("Truncated ZIP member name")
        name = name_bytes.decode("utf-8" if fields[3] & 0x800 else "cp437")
        members.append({"name": name, "compressed_bytes": compressed, "uncompressed_bytes": uncompressed})
        cursor = name_start + name_length + extra_length + comment_length
    if cursor != len(directory) or len(members) != count:
        raise ValueError("ZIP directory count or size mismatch")
    documentation = [
        member for member in members if member["name"].lower().endswith((".xml", ".dpl", ".pdf", ".txt"))
    ]
    result = {
        "source_url": URL,
        "head_status": head.status_code,
        "archive_bytes": size,
        "single_file_cap_bytes": 8 * 1024**3,
        "archive_exceeds_single_file_cap": size > 8 * 1024**3,
        "metadata_bytes_transferred": len(tail) + len(directory),
        "zip64_member_count": count,
        "zip64_directory_bytes": cd_size,
        "documentation_members": documentation,
        "first_five_member_names": [member["name"] for member in members[:5]],
        "acoustic_payload_bytes_read": 0,
        "archive_downloaded": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
