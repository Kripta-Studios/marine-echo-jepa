"""Verify the prose correction without rereading protected sources or replacing prior evidence."""
import hashlib
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
BASE = '71cab57d4118a59d1b9acbebb31078e6c032bd97'
DOCUMENTS = ['README.md', 'INDEX.md', 'START_HERE.md',
             'CODEX_MEETING_READY_AFTER_DIAGNOSTIC.md',
             'meeting/20260929/MEETING_BRIEF.md', 'meeting/20260929/DEMO_RUNBOOK.md']
OUTPUT = Path(__file__).with_name('correction-verification.json')
assert not OUTPUT.exists(), 'Preserve an existing correction result'
assert not subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=ROOT, text=True).strip()
changed = subprocess.check_output(['git', 'diff', BASE, '--name-only'], cwd=ROOT, text=True).splitlines()
assert set(changed) == set(DOCUMENTS), changed
links = 0
document_hashes = {}
for name in DOCUMENTS:
    data = (ROOT / name).read_bytes()
    assert b'\r' not in data, name
    text = data.decode('utf-8')
    original = subprocess.check_output(['git', 'show', BASE + ':' + name], cwd=ROOT).decode('utf-8')
    assert re.findall(r'\d+(?:\.\d+)*', text) == re.findall(r'\d+(?:\.\d+)*', original), name
    assert not re.search(r'\b(?:is|remains) pending\b|\breviewer pending\b', text, re.I), name
    assert 'checklist' in text and 'separate review record' in text, name
    for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)', text):
        target = target.strip('<>')
        if urlparse(target).scheme or target.startswith('#'):
            continue
        assert ((ROOT / name).parent / unquote(target.split('#', 1)[0])).resolve().exists(), (name, target)
        links += 1
    document_hashes[name] = hashlib.sha256(data).hexdigest()
runbook = (ROOT / DOCUMENTS[-1]).read_text(encoding='utf-8')
primary = next(line for line in runbook.splitlines() if line.startswith('| 1:00'))
assert 'switch direct/EMA' in primary and 'LightGBM' not in primary
assert 'then inspect EMA at fixed index 100' in primary
assert 'LightGBM was not selected in the fresh browser smoke' in runbook
assert 'against the live scratch server' in runbook
assert 'after owned server shutdown, using already loaded SPA data from the initial packaged API response' in runbook
assert 'No new live API call occurred at that later stage' in runbook
assert 'spoken eight-minute rehearsal remain NOT_RUN' in runbook
assert 'complete persisted network census is **NOT_RUN**' in runbook
assert 'denied snapshot/log writes into the main checkout' in runbook
assert 'No denied main write was retried through another channel' in runbook
assert '| 7:30–8:00 |' in runbook
brief_words = len((ROOT / DOCUMENTS[-2]).read_text(encoding='utf-8').split())
assert brief_words <= 600
result = {'status': 'PASSED', 'base_commit': BASE, 'changed_documents': DOCUMENTS,
          'relative_links_checked': links, 'brief_word_count': brief_words,
          'all_numeric_tokens_unchanged': True, 'authored_documents_lf': True,
          'live_aeon_vs_cached_raw_chronology_explicit': True,
          'primary_script_only_browser_tested_direct_ema': True,
          'lightgbm_api_static_scope_explicit': True,
          'full_spoken_rehearsal': 'NOT_RUN', 'network_permission_limits_preserved': True,
          'review_disposition': 'Referenced through checklist/current separate review record; no approval claimed',
          'original_evidence_logs_scripts_and_all_other_tracked_files_unchanged': True,
          'protected_source_reads': False, 'research_launch_browser_reruns': False,
          'document_sha256': document_hashes}
OUTPUT.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
print(json.dumps(result, indent=2))
