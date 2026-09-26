"""Keep immutable report versions while updating the current continuation pointer file."""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def write_report(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    versions = path.parent / "report-versions"
    versions.mkdir(exist_ok=True)
    payload = (json.dumps(document, indent=2, allow_nan=False) + "\n").encode("utf-8")
    for data in ([path.read_bytes()] if path.exists() else []) + [payload]:
        digest = hashlib.sha256(data).hexdigest()
        version = versions / f"{path.stem}-{digest}.json"
        if version.exists():
            if version.read_bytes() != data:
                raise ValueError("Existing immutable report bytes differ.")
        else:
            with version.open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
    handle, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".stage-")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)
