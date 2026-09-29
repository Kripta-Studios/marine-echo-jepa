"""Focused document, numeric, hash, diff-scope and preservation checks only."""
import hashlib
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlparse

E = Path(__file__).resolve().parent
R = E.parents[1]
M = R/'meeting/20260929'

def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

prep = load(E/'preparation.json')
baseline = load(E/'state-before.json')
rehearsal = load(E/'rehearsal.json')
shutdown = load(E/'shutdown.json')
owned = ['README.md','INDEX.md','START_HERE.md','CODEX_MEETING_READY_AFTER_DIAGNOSTIC.md','orchestration/STATUS.md']
owned += [p.relative_to(R).as_posix() for p in M.glob('*')]
documents = [R/p for p in owned if p.endswith('.md') and p != 'orchestration/STATUS.md']
links = []
for doc in documents:
    text = doc.read_text(encoding='utf-8')
    for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)',text):
        target = target.strip('<>')
        if urlparse(target).scheme or target.startswith('#'):
            continue
        path = (doc.parent/unquote(target.split('#',1)[0])).resolve()
        assert path.exists(), (doc.relative_to(R),target)
        links.append({'document':doc.relative_to(R).as_posix(),'target':target})
fallback = E/'static-fallback.html'
for target in re.findall(r'href="([^"]+)"',fallback.read_text(encoding='utf-8')):
    assert (E/unquote(target)).resolve().exists(),target
brief_words = len((M/'MEETING_BRIEF.md').read_text(encoding='utf-8').split())
assert brief_words <= 600, brief_words
diagnostic = (R/'orchestration/reports/AEON_CHECKPOINT_OBJECTIVE_DIAGNOSTIC_20260929.md').read_text(encoding='utf-8')
technical = (R/'orchestration/reports/AEON_CHECKPOINT_DIAGNOSTIC_TECHNICAL_20260929.md').read_text(encoding='utf-8')
retrospective = (R/'orchestration/reports/AEON_RETROSPECTIVE_OUTCOME_20260927.md').read_text(encoding='utf-8')
transfer = (R/'orchestration/reports/AEON_EXTERNAL_SECONDARY_NUMERIC_OUTCOME_20260928.md').read_text(encoding='utf-8')
numeric_sources = {'0.639577':diagnostic,'0.639062':diagnostic,'0.642170':diagnostic,
                   '0.533954':retrospective,'0.563387':retrospective,
                   '0.678922':f"{0.6789216779097978:.6f}",'0.670178':f"{0.6701783099148351:.6f}"}
assert '0.6789216779097978' in transfer and '0.6701783099148351' in transfer
for value, source in numeric_sources.items():
    assert value in source,value
    assert value in (R/'README.md').read_text(encoding='utf-8'), value
    assert value in (M/'MEETING_BRIEF.md').read_text(encoding='utf-8'),value
seed_scores = {}
for family, seed, score in re.findall(r'\| (direct_supervised1500|direct_supervised3000|ema_jepa_supervised1500) \| (7|13|23) \| ([0-9.]+) \|',technical):
    seed_scores[(family,seed)] = float(score)
assert len(seed_scores) == 9
for seed in ('7','13','23'):
    assert seed_scores[('direct_supervised1500',seed)] < seed_scores[('ema_jepa_supervised1500',seed)]
    assert seed_scores[('direct_supervised1500',seed)] < seed_scores[('direct_supervised3000',seed)]
decision = (M/'NEXT_RESEARCH_DECISION.md').read_text(encoding='utf-8')
for fragment in ('0.03','6000','20-minute','25%','1%','DRAFT / NOT_AUTHORIZED / NOT_QUEUED','NO_FURTHER_FITTING_BEFORE_MEETING'):
    assert fragment in decision,fragment
for fragment in ('0.03','6000','20-minute','25%','1%'):
    assert fragment in diagnostic,fragment
assert shutdown['owned_port_closed'] is True
for path, digest in prep['sources'].items():
    assert sha(R/path) == digest,path
preserved = []
for path, digest in baseline['historical_tracked_hashes'].items():
    if path == 'orchestration/STATUS.md':
        assert sha(E/'status-original.txt') == digest
        assert (R/path).read_bytes().endswith((E/'status-original.txt').read_bytes())
    else:
        assert sha(R/path) == digest,path
    preserved.append(path)
assert sha(Path(prep['archive'])) == prep['archive_sha256_before']
assert (R/'orchestration/STATUS.md').read_bytes().endswith((E/'status-original.txt').read_bytes())
untracked_preserved = []
for line in baseline['git_status'].splitlines():
    if line.startswith('?? '):
        path = line[3:].strip('"')
        assert (R/path).exists(),path
        untracked_preserved.append(path)
changed = subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=R,text=True).splitlines()
for path in changed:
    if path.startswith('evidence/meeting-readiness-20260929/'):
        assert not any(token in Path(path).name for token in ('contract','routing','state-before','coordinator-','review','events')),path
    else:
        assert path in owned,path
check = subprocess.run(['git','diff','--check'],cwd=R,text=True,capture_output=True)
assert check.returncode == 0,check.stdout+check.stderr
rows = []
for example in rehearsal['examples']:
    for model,quantiles in example['models'].items():
        rows.append({'index':example['chronological_index'],'model':model,
                     'q05':f'{quantiles[0][0]:.6f}','q50':f'{quantiles[0][2]:.6f}','q95':f'{quantiles[0][4]:.6f}'})
        for value in rows[-1].values():
            if isinstance(value,str) and value.startswith('-'):
                assert value in (M/'DEMO_RUNBOOK.md').read_text(encoding='utf-8'),value
checklist = load(M/'READINESS_CHECKLIST.json')
assert checklist['fresh_review']['status'] == 'NOT_RUN'
assert checklist['self_approval'] is False
assert set(c['status'] for c in checklist['checks']) <= {'PASSED','FAILED','NOT_RUN','NOT_APPLICABLE'}
for pin in checklist['source_pins']:
    assert sha(R/pin['path']) == pin['sha256'],pin['path']
supplement = load(E/'supplement-verification.json')
raw_row = supplement['raw_first_row']
assert raw_row['horizons'][0]['truth_code'] is None
for row in raw_row['horizons'][1:]:
    value = f"{row['truth_code']:.6f}"
    assert value in (M/'DEMO_RUNBOOK.md').read_text(encoding='utf-8')
    assert value in fallback.read_text(encoding='utf-8')
for row in raw_row['horizons']:
    for key in ('ridge_quantiles_code','direct_quantiles_code'):
        for value in row[key]:
            assert f'{value:.6f}' in fallback.read_text(encoding='utf-8')
result = {'status':'PASSED','relative_links_checked':len(links), 'static_fallback_links_checked':5,
          'brief_word_count':brief_words,'numeric_statements_checked':list(numeric_sources),
          'technical_seed_scores':{f'{k[0]}/{k[1]}':v for k,v in seed_scores.items()},
          'served_example_table_checked':rows,'historical_baseline_hashes_preserved':len(preserved),
          'separate_packaged_raw_later_observations_checked':True,'coordinator_browser_evidence_pins_checked':True,
          'untracked_baseline_entries_still_present':untracked_preserved,
          'untracked_preservation_limit':'Existence checked; baseline did not supply hashes for these untracked paths. No writes/deletes performed there.',
          'status_original_bytes_preserved':True,'archive_sha256_after':sha(Path(prep['archive'])),
          'source_report_hashes_unchanged':prep['sources'],'git_diff_scope':changed,'git_diff_check_exit':check.returncode,
          'tdd':'NOT_APPLICABLE: no production behavior change','scientific_suite':'NOT_RUN: prohibited in this lane'}
(E/'document-verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
for check_item in checklist['checks']:
    for path in check_item.get('evidence',[]):
        assert (R/path).exists(),path
print(json.dumps(result,indent=2))
