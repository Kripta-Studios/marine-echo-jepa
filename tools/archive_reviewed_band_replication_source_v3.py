"""Preserve actual reviewed source bytes independently of Git newline normalization."""

import base64
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    review_path = ROOT / "evidence/ssl-research-v1/band-fixed-replication-prefit-compact-final-v3.json"
    review = json.loads(review_path.read_bytes())
    if review.get("status") != "APPROVED_BAND_FIXED_REPLICATION_REFERENCES":
        raise ValueError("Closed independent approval required")
    files = {}
    for group in ("bindings", "proof_bindings"):
        for name, expected in review.get(group, {}).items():
            path = Path(name)
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected:
                raise ValueError("Reviewed source changed before archival")
            if (path.is_relative_to(ROOT / "src") or path.is_relative_to(ROOT / "tools")
                    or path.is_relative_to(ROOT / "configs") or path.is_relative_to(ROOT / "docs")
                    or path.is_relative_to(ROOT / "orchestration")) and path.suffix in (".py", ".json", ".md", ".txt"):
                raw = path.read_bytes()
                files[str(path.relative_to(ROOT))] = {"sha256": actual, "base64_bytes": base64.b64encode(raw).decode("ascii")}
    target = ROOT / "evidence/ssl-research-v1/band-replication-reviewed-source-byte-archive-v3.json"
    with target.open("x", encoding="utf-8") as stream:
        json.dump({"status": "EXACT_REVIEWED_SOURCE_BYTES_ARCHIVED", "review_sha256": hashlib.sha256(review_path.read_bytes()).hexdigest(),
                   "files": files, "numerical_model_or_corpus_bytes_archived": False,
                   "scientific_source_changes": False}, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "EXACT_REVIEWED_SOURCE_BYTES_ARCHIVED", "files": len(files)}))


if __name__ == "__main__":
    main()
