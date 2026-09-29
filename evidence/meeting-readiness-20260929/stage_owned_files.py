"""Stage only explicit authorized paths, preserving coordinator and unrelated work."""
import hashlib
import json
import subprocess
from pathlib import Path

E=Path(__file__).resolve().parent
R=E.parents[1]
paths=['README.md','INDEX.md','START_HERE.md','CODEX_MEETING_READY_AFTER_DIAGNOSTIC.md','orchestration/STATUS.md',
       'meeting/20260929/MEETING_BRIEF.md','meeting/20260929/DEMO_RUNBOOK.md',
       'meeting/20260929/PILOT_DATA_REQUEST.md','meeting/20260929/NEXT_RESEARCH_DECISION.md',
       'meeting/20260929/READINESS_CHECKLIST.json']
evidence_names=[
    '.gitattributes',
    'prepare_scratch.py','start_scratch.ps1','stop_scratch.ps1','rehearse_static.py',
    'update_entrypoints.py','build_checklist.py','verify_documents.py',
    'incorporate_coordinator_evidence.py','stage_owned_files.py',
    'preparation.json','example-selection.json','launch.json','health.json','rehearsal.json',
    'launcher.stdout.log','launcher.stderr.log','package-verification.log',
    'final-runtime-verification.json','shutdown.json','shutdown-first-attempt.json',
    'shutdown-second-attempt.json','browser-availability.json',
    'document-build-first-attempt.json','document-verification-encoding-failure.json',
    'document-verification.json','supplement-verification.json','static-fallback.html',
    'staging-byte-preservation-first-attempt.json',
    'staging-byte-binding-first-attempt.json',
    'status-original.txt','served-index.html','served-index-4wqFJhvG.css','served-index-DJlTwxbi.js',
    'served-study.json','replay-0-core_direct_equal_three_seed_ensemble.json',
    'replay-0-core_ema_equal_three_seed_ensemble.json','replay-0-post_hoc_lightgbm.json',
    'replay-100-core_direct_equal_three_seed_ensemble.json',
    'replay-100-core_ema_equal_three_seed_ensemble.json','replay-100-post_hoc_lightgbm.json',
]
paths += ['evidence/meeting-readiness-20260929/'+name for name in evidence_names]
already_staged = subprocess.check_output(['git','diff','--cached','--name-only'],cwd=R,text=True).splitlines()
assert set(already_staged) <= set(paths) | {'evidence/meeting-readiness-20260929/implementation-scope.json'}, 'Foreign staged work: stop without changing it'
for path in paths:
    assert (R/path).is_file(),path
    if path.startswith('evidence/'):
        name=Path(path).name
        assert not any(word in name for word in ('contract','routing','state-before','coordinator-','review-schema','implementation-events','review-events'))
record={'status':'STAGED_SCOPE_VERIFIED','base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),
        'paths':paths,'working_tree_sha256':{p:hashlib.sha256((R/p).read_bytes()).hexdigest() for p in paths},
        'root_owned_evidence_excluded':True,'raw_event_stream_excluded':True,'unrelated_subsequent_work_excluded':True,
        'fresh_review':'PENDING; no self-approval','commit_scope':'Meeting documentation and implementation evidence only'}
record_path='evidence/meeting-readiness-20260929/implementation-scope.json'
(R/record_path).write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
paths.append(record_path)
subprocess.run(['git','add','--',*paths],cwd=R,check=True)
subprocess.run(['git','add','--renormalize','--',*paths],cwd=R,check=True)
staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=R,text=True).splitlines()
assert set(staged)==set(paths),(set(staged)-set(paths),set(paths)-set(staged))
subprocess.run(['git','diff','--cached','--check'],cwd=R,check=True)
for path in paths:
    if path.startswith('evidence/'):
        staged_bytes = subprocess.check_output(['git','show',':'+path],cwd=R)
        assert staged_bytes == (R/path).read_bytes(),path
print(f'Staged and checked {len(paths)} explicit owned files; coordinator/routing/events and unrelated work excluded.')
