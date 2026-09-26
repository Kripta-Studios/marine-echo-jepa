"""Inventory and transactionally extract a locally supplied acoustic ZIP."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
import unicodedata
import zipfile
import zlib
from pathlib import Path, PurePosixPath
from typing import Any

_CHUNK = 1024 * 1024
_RESERVED = {"CON", "PRN", "AUX", "NUL"} | {
    f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10)
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_crc(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    crc = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(_CHUNK), b""):
            digest.update(chunk)
            crc = zlib.crc32(chunk, crc)
    return digest.hexdigest(), crc


def _stable_sha256(path: Path) -> tuple[int, str]:
    before = path.stat()
    digest = _sha256(path)
    after = path.stat()
    identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
    if identity(before) != identity(after):
        raise OSError(f"Source changed during hashing: {path.name}")
    return before.st_size, digest


def _safe_parts(name: str) -> tuple[str, ...]:
    normalized = name.replace("\\", "/")
    parts = normalized.rstrip("/").split("/")
    if not parts or normalized.startswith("/") or any(not part for part in parts):
        raise ValueError("Archive contains an absolute or empty path.")
    for part in parts:
        if (
            part in (".", "..")
            or ":" in part
            or part.endswith((".", " "))
            or any(ord(char) < 32 for char in part)
            or part.split(".")[0].upper() in _RESERVED
        ):
            raise ValueError("Archive contains an unsafe Windows path.")
    return tuple(parts)


def _validated_members(
    archive: zipfile.ZipFile, max_expanded_bytes: int
) -> tuple[list[zipfile.ZipInfo], int]:
    if max_expanded_bytes <= 0:
        raise ValueError("Expanded-byte limit must be positive.")
    members = archive.infolist()
    if len(members) > 200_000:
        raise ValueError("Archive member count exceeds the limit.")
    seen: set[str] = set()
    files: set[str] = set()
    total = 0
    for member in members:
        parts = _safe_parts(member.filename)
        key = unicodedata.normalize("NFC", "/".join(parts)).casefold()
        if key in seen:
            raise ValueError("Archive contains duplicate normalized paths.")
        if any(str(parent) in files for parent in PurePosixPath(key).parents):
            raise ValueError("Archive member has a file ancestor.")
        if not member.is_dir() and any(
            existing.startswith(key + "/") for existing in seen
        ):
            raise ValueError("Archive file shadows a directory.")
        file_type = stat.S_IFMT(member.external_attr >> 16)
        if file_type not in (0, stat.S_IFREG, stat.S_IFDIR):
            raise ValueError("Archive contains a link or special file.")
        if member.flag_bits & 1:
            raise ValueError("Encrypted archives are unsupported.")
        if member.file_size > 1000 * max(1, member.compress_size):
            raise ValueError("Archive member exceeds compression ratio limit.")
        seen.add(key)
        if not member.is_dir():
            files.add(key)
        total += member.file_size
        if total > max_expanded_bytes:
            raise ValueError("Archive exceeds the expanded-byte limit.")
    return members, total


def inventory_local_source(source_dir: Path) -> dict[str, Any]:
    """Hash existing local files without modifying or downloading them."""
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    files = []
    for path in sorted(source_dir.iterdir()):
        if not path.is_file():
            continue
        size, digest = _stable_sha256(path)
        files.append({"name": path.name, "bytes": size, "sha256": digest})
    return {"source_dir": str(source_dir.resolve()), "files": files}


def extract_local_zip(
    archive_path: Path,
    destination: Path,
    *,
    max_expanded_bytes: int,
    min_free_bytes: int,
) -> dict[str, Any]:
    """Validate, CRC-check and hash every member before atomic publication.

    An existing complete destination is verified and reused. A failure leaves no
    published directory, and the supplied source ZIP remains untouched.
    """
    if min_free_bytes < 0:
        raise ValueError("Free-space reserve cannot be negative.")
    archive_path = archive_path.resolve(strict=True)
    if destination.is_symlink():
        raise ValueError("Destination symlink is forbidden.")
    destination = destination.resolve()
    if destination == archive_path or archive_path in destination.parents:
        raise ValueError("Destination must be separate from the source archive.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_size, source_hash = _stable_sha256(archive_path)
    source_stat = archive_path.stat()
    with zipfile.ZipFile(archive_path) as archive:
        members, total = _validated_members(archive, max_expanded_bytes)
        free = shutil.disk_usage(destination.parent).free
        if free - total < min_free_bytes:
            raise OSError("Insufficient free space for expanded archive and reserve.")
        if destination.exists():
            manifest_path = destination / ".extraction-manifest.json"
            if not manifest_path.is_file():
                raise FileExistsError("Destination exists without a complete manifest.")
            result: dict[str, Any] = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            if result.get("archive_sha256") != source_hash:
                raise FileExistsError(
                    "Destination belongs to a different source archive."
                )
            expected = {
                "/".join(_safe_parts(member.filename)): member.CRC
                for member in members
                if not member.is_dir()
            }
            actual_names = [file["path"] for file in result["files"]]
            if len(actual_names) != len(expected) or set(actual_names) != set(expected):
                raise ValueError("Extraction manifest does not match archive contents.")
            for file in result["files"]:
                parts = _safe_parts(file["path"])
                actual = destination.joinpath(*parts)
                for parent in (destination, *list(actual.parents)[:-1]):
                    if parent.is_symlink():
                        raise ValueError(
                            "Extracted member contains a symlinked directory."
                        )
                if actual.is_symlink() or not actual.resolve().is_relative_to(
                    destination
                ):
                    raise ValueError("Extracted member escapes destination.")
                if not actual.is_file() or actual.stat().st_size != file["bytes"]:
                    raise OSError(
                        f"Previously extracted member failed integrity: {file['path']}"
                    )
                actual_sha, actual_crc = _sha256_crc(actual)
                if (
                    actual_sha != file["sha256"]
                    or actual_crc != expected[file["path"]]
                    or file["crc32"] != f"{expected[file['path']]:08x}"
                ):
                    raise OSError(
                        f"Previously extracted member failed integrity: {file['path']}"
                    )
            return result
        stage = Path(
            tempfile.mkdtemp(
                prefix=destination.name + ".stage.", dir=destination.parent
            )
        )
        try:
            files = []
            written_total = 0
            for member in members:
                relative = PurePosixPath(member.filename.replace("\\", "/"))
                target = stage.joinpath(*relative.parts)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                size = 0
                with archive.open(member) as source, target.open("xb") as sink:
                    for chunk in iter(lambda: source.read(_CHUNK), b""):
                        size += len(chunk)
                        written_total += len(chunk)
                        if (
                            written_total > max_expanded_bytes
                            or size > member.file_size
                        ):
                            raise ValueError(
                                "Actual extracted bytes exceed metadata or limit."
                            )
                        sink.write(chunk)
                        digest.update(chunk)
                if size != member.file_size:
                    raise ValueError("Extracted size differs from ZIP metadata.")
                files.append(
                    {
                        "path": relative.as_posix(),
                        "bytes": size,
                        "sha256": digest.hexdigest(),
                        "crc32": f"{member.CRC:08x}",
                    }
                )
            result = {
                "schema_version": "1.0",
                "archive_name": archive_path.name,
                "archive_bytes": source_size,
                "archive_sha256": source_hash,
                "members": len(members),
                "expanded_bytes": written_total,
                "files": files,
            }
            current_stat = archive_path.stat()
            if (
                current_stat.st_dev,
                current_stat.st_ino,
                current_stat.st_size,
                current_stat.st_mtime_ns,
            ) != (
                source_stat.st_dev,
                source_stat.st_ino,
                source_stat.st_size,
                source_stat.st_mtime_ns,
            ):
                raise OSError("Source archive changed during extraction.")
            (stage / ".extraction-manifest.json").write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8"
            )
            os.replace(stage, destination)
            return result
        finally:
            if stage.exists():
                shutil.rmtree(stage)
