"""Safely extract the approved immutable package into a new meeting scratch."""
import hashlib
import json
import re
import stat
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = Path(__file__).resolve().parent
ARCHIVE = ROOT / 'release/aeon-offline-scale-expanded-20260928-r3.zip'
EXPECTED = '5e265e08bf1041bbe0fb4dd2a5f3503bae98f996096577f976beaf9384d6bb57'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

assert sha(ARCHIVE) == EXPECTED
scratch = ROOT / 'outputs/meeting-readiness-20260929-r3'
suffix = 2
while scratch.exists():
    scratch = ROOT / f'outputs/meeting-readiness-20260929-r3-{suffix}'
    suffix += 1
with zipfile.ZipFile(ARCHIVE) as archive:
    members = archive.infolist()
    names = [m.filename for m in members]
    assert len(names) == len(set(n.casefold() for n in names)) == 108
    assert sum(m.file_size for m in members) < 100_000_000
    for member in members:
        path = PurePosixPath(member.filename)
        assert not path.is_absolute() and '..' not in path.parts
        assert '\\' not in member.filename and ':' not in member.filename
        assert path.parts[0] == 'aeon-offline-scale-expanded-20260928-r3'
        assert not stat.S_ISLNK(member.external_attr >> 16)
        for component in path.parts:
            assert component.rstrip(' .') == component
            assert not re.match(r'^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)', component, re.I)
    scratch.mkdir(parents=True)
    for member in members:
        destination = scratch.joinpath(*PurePosixPath(member.filename).parts)
        assert destination.resolve().is_relative_to(scratch.resolve())
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as out:
            out.write(archive.read(member))
package = scratch / 'aeon-offline-scale-expanded-20260928-r3'
manifest = {}
for line in (package / 'SHA256SUMS').read_text().splitlines():
    digest, name = line.split('  ', 1)
    assert name not in manifest
    manifest[name] = digest
    assert sha(package / name) == digest, name
assert len(manifest) == 107
assert {p.relative_to(package).as_posix() for p in package.rglob('*') if p.is_file()} == set(manifest) | {'SHA256SUMS'}
sources = [
    'orchestration/reports/AEON_CHECKPOINT_OBJECTIVE_DIAGNOSTIC_20260929.md',
    'orchestration/reports/AEON_CHECKPOINT_DIAGNOSTIC_TECHNICAL_20260929.md',
    'orchestration/reviews/AEON_CHECKPOINT_OBJECTIVE_DIAGNOSTIC_REVIEW_20260929.json',
    'orchestration/reviews/AEON_SCALE_EXPANDED_OFFLINE_R3_REVIEW_20260929.json',
    'orchestration/reports/AEON_SCALE_EXPANDED_OFFLINE_R3_VALIDATION_20260929.md',
    'orchestration/reports/AEON_RETROSPECTIVE_OUTCOME_20260927.md',
    'orchestration/reports/AEON_EXTERNAL_SECONDARY_NUMERIC_OUTCOME_20260928.md',
]
selection = {'recorded_at': datetime.now(timezone.utc).isoformat(), 'indices': [0, 100],
             'rule': 'Zero-based chronological cutoff indices fixed before opening replay forecasts or later observations; all three packaged model families, no superiority search.'}
(EVIDENCE / 'example-selection.json').write_text(json.dumps(selection, indent=2) + '\n', encoding='utf-8')
(EVIDENCE / 'status-original.txt').write_bytes((ROOT / 'orchestration/STATUS.md').read_bytes())
record = {'recorded_at': datetime.now(timezone.utc).isoformat(), 'archive': str(ARCHIVE),
          'archive_sha256_before': EXPECTED, 'scratch': str(scratch), 'package': str(package),
          'assets_verified': len(manifest), 'manifest_sha256': sha(package / 'SHA256SUMS'),
          'source_revision_sha256': sha(package / 'SOURCE_REVISION.json'),
          'sources': {s: sha(ROOT / s) for s in sources},
          'head_before': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
(EVIDENCE / 'preparation.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print(json.dumps(record, indent=2))
