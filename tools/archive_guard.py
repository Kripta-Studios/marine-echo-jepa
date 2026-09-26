"""Fail-closed ZIP metadata inspection. This is not a complete extraction service."""
from __future__ import annotations

import stat
import zipfile
from pathlib import PurePosixPath

RESERVED = {"CON", "PRN", "AUX", "NUL"} | {
    f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10)
}


def validate_zip_metadata(
    archive: zipfile.ZipFile,
    *,
    max_bytes: int = 80 * 1024**3,
    max_members: int = 200_000,
    max_ratio: int = 1000,
) -> dict[str, int]:
    """Reject unsafe names/types/size claims before any extraction.

    An actual extractor must additionally count bytes written, check CRC, enforce destination
    containment, reject existing symlinks and publish atomically. Metadata alone is not enough.
    """
    if min(max_bytes, max_members, max_ratio) <= 0:
        raise ValueError("All limits must be positive.")
    members = archive.infolist()
    if len(members) > max_members:
        raise ValueError("Archive has too many members.")
    seen: set[str] = set()
    files: set[str] = set()
    total = 0
    for info in members:
        name = info.filename.replace("\\", "/")
        parts = name.rstrip("/").split("/")
        if not parts or not all(parts) or name.startswith("/") or ":" in name:
            raise ValueError("Absolute, empty, UNC or drive-qualified member path.")
        if any(part in (".", "..") or part.endswith((".", " ")) for part in parts):
            raise ValueError("Unsafe relative or Windows-normalized member path.")
        if any(any(ord(char) < 32 for char in part) for part in parts):
            raise ValueError("Control characters in member path.")
        if any(part.split(".")[0].upper() in RESERVED for part in parts):
            raise ValueError("Reserved Windows device name.")
        key = "/".join(parts).casefold()
        if key in seen:
            raise ValueError("Duplicate case-insensitive member path.")
        mode = stat.S_IFMT(info.external_attr >> 16)
        if mode not in (0, stat.S_IFREG, stat.S_IFDIR):
            raise ValueError("Symlink or special archive member rejected.")
        if info.flag_bits & 1:
            raise ValueError("Encrypted members are unsupported.")
        parents = [str(parent).casefold() for parent in PurePosixPath(key).parents]
        if any(parent in files for parent in parents):
            raise ValueError("A file is used as an ancestor directory.")
        if not info.is_dir() and any(existing.startswith(key + "/") for existing in seen):
            raise ValueError("A file shadows an existing directory tree.")
        seen.add(key)
        if not info.is_dir():
            files.add(key)
        total += info.file_size
        if total > max_bytes:
            raise ValueError("Uncompressed-byte limit exceeded.")
        if info.file_size > max_ratio * max(info.compress_size, 1):
            raise ValueError("Suspicious compression ratio.")
    return {"members": len(members), "uncompressed_bytes": total}
