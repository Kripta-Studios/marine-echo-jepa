"""Independent archive safety expectations for local PANGAEA ingestion."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import zipfile
from pathlib import Path

import pytest

from marine_echo.data import local_archive
from marine_echo.data.local_archive import extract_local_zip, inventory_local_source


def _zip(path: Path, members: list[tuple[str, bytes]], *, symlink: bool = False) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members:
            info = zipfile.ZipInfo(name)
            if symlink:
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, payload)


@pytest.mark.parametrize(
    "name",
    ["../escape", "/absolute", "C:/drive", "a/../escape", "CON.txt", "a\\..\\escape"],
)
def test_rejects_unsafe_names_without_publishing(tmp_path: Path, name: str) -> None:
    archive = tmp_path / "source.zip"
    _zip(archive, [(name, b"bad")])
    destination = tmp_path / "published"
    with pytest.raises(ValueError):
        extract_local_zip(archive, destination, max_expanded_bytes=1024, min_free_bytes=0)
    assert not destination.exists()
    assert list(tmp_path.glob("published.stage.*")) == []


def test_rejects_duplicate_casefold_and_symlink(tmp_path: Path) -> None:
    archive = tmp_path / "dup.zip"
    _zip(archive, [("A.xml", b"one"), ("a.XML", b"two")])
    with pytest.raises(ValueError):
        extract_local_zip(archive, tmp_path / "out", max_expanded_bytes=1024, min_free_bytes=0)
    _zip(archive, [("link", b"target")], symlink=True)
    with pytest.raises(ValueError):
        extract_local_zip(archive, tmp_path / "out", max_expanded_bytes=1024, min_free_bytes=0)


def test_limit_and_corrupt_crc_never_publish(tmp_path: Path) -> None:
    archive = tmp_path / "source.zip"
    _zip(archive, [("record.01A", b"abcdefgh")])
    with pytest.raises(ValueError):
        extract_local_zip(archive, tmp_path / "out", max_expanded_bytes=7, min_free_bytes=0)
    data = archive.read_bytes()
    archive.write_bytes(data.replace(b"abcdefgh", b"abcxefgh"))
    with pytest.raises(zipfile.BadZipFile):
        extract_local_zip(archive, tmp_path / "out", max_expanded_bytes=1024, min_free_bytes=0)
    assert not (tmp_path / "out").exists()


def test_transaction_preserves_configs_and_inventory(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    archive = source / "data.zip"
    _zip(
        archive,
        [
            ("20021600.XML", b"config"),
            ("20021600.DPL", b"schedule"),
            ("20021609.01A", b"raw"),
        ],
    )
    (source / "manual.pdf").write_bytes(b"source document")
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
    inventory = inventory_local_source(source)
    assert {item["name"] for item in inventory["files"]} == set(before)
    destination = tmp_path / "published"
    result = extract_local_zip(archive, destination, max_expanded_bytes=1024, min_free_bytes=0)
    assert result["members"] == 3
    assert (destination / "20021600.XML").read_bytes() == b"config"
    assert (destination / "20021600.DPL").read_bytes() == b"schedule"
    assert (destination / "20021609.01A").read_bytes() == b"raw"
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()} == before
    assert (
        extract_local_zip(archive, destination, max_expanded_bytes=1024, min_free_bytes=0) == result
    )


def test_existing_manifest_cannot_read_outside_destination(tmp_path: Path) -> None:
    archive = tmp_path / "source.zip"
    _zip(archive, [("record.01A", b"raw")])
    destination = tmp_path / "published"
    extract_local_zip(archive, destination, max_expanded_bytes=1024, min_free_bytes=0)
    outside = tmp_path / "outside"
    outside.write_bytes(b"secret")
    manifest = destination / ".extraction-manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["files"][0] = {
        "path": "../outside",
        "bytes": 6,
        "sha256": hashlib.sha256(b"secret").hexdigest(),
    }
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises((ValueError, OSError)):
        extract_local_zip(archive, destination, max_expanded_bytes=1024, min_free_bytes=0)


def test_existing_member_symlink_is_not_verified_as_content(tmp_path: Path) -> None:
    archive = tmp_path / "source.zip"
    _zip(archive, [("record.01A", b"raw")])
    destination = tmp_path / "published"
    extract_local_zip(archive, destination, max_expanded_bytes=1024, min_free_bytes=0)
    target = tmp_path / "outside"
    target.write_bytes(b"raw")
    (destination / "record.01A").unlink()
    try:
        os.symlink(target, destination / "record.01A")
    except OSError:
        pytest.skip("Windows symlink privilege unavailable")
    with pytest.raises((ValueError, OSError)):
        extract_local_zip(archive, destination, max_expanded_bytes=1024, min_free_bytes=0)


def test_inventory_rejects_source_change_during_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    path = source / "data.bin"
    path.write_bytes(b"first")
    original_hash = local_archive._sha256

    def changing_hash(candidate: Path) -> str:
        digest = original_hash(candidate)
        candidate.write_bytes(b"second-longer")
        return digest

    monkeypatch.setattr(local_archive, "_sha256", changing_hash)
    with pytest.raises(OSError):
        inventory_local_source(source)


def test_manifest_rewrite_cannot_approve_tampered_member(tmp_path: Path) -> None:
    archive = tmp_path / "source.zip"
    _zip(archive, [("record.01A", b"raw")])
    destination = tmp_path / "published"
    extract_local_zip(archive, destination, max_expanded_bytes=1024, min_free_bytes=0)
    (destination / "record.01A").write_bytes(b"bad")
    manifest = destination / ".extraction-manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["files"][0]["sha256"] = hashlib.sha256(b"bad").hexdigest()
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises((ValueError, OSError)):
        extract_local_zip(archive, destination, max_expanded_bytes=1024, min_free_bytes=0)
