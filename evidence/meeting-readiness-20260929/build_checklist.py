"""Bind actual meeting readiness to collected execution evidence."""
import hashlib
import json
from pathlib import Path

E = Path(__file__).resolve().parent
R = E.parents[1]
M = R/'meeting/20260929'

def load(name):
    return json.loads((E/name).read_text(encoding='utf-8-sig'))

prep,launch,rehearsal,shutdown,final = [load(n) for n in ('preparation.json','launch.json','rehearsal.json','shutdown.json','final-runtime-verification.json')]
# Correct historical screenshot names without repeating the completed smoke.
rehearsal['browser']['historical_screenshots'] = ['evidence/browser/aeon-scale-expanded-r3-desktop.png','evidence/browser/aeon-scale-expanded-r3-mobile.png']
(E/'rehearsal.json').write_text(json.dumps(rehearsal,indent=2)+'\n',encoding='utf-8')
prefix = 'evidence/meeting-readiness-20260929/'
checks = []

def add(name,status,command,exit_code,result,evidence):
    checks.append({'id':name,'status':status,'command':command,'exit_code':exit_code,'result':result,'evidence':[prefix+p for p in evidence]})

add('safe_extraction_and_107_asset_hashes','PASSED','.venv/Scripts/python.exe -B evidence/meeting-readiness-20260929/prepare_scratch.py',0,'Archive hash matched before extraction; safe new scratch; 107 exact manifest assets; no package changes.',['preparation.json','prepare_scratch.py'])
add('offline_launch','PASSED','powershell.exe -NoProfile -ExecutionPolicy Bypass -File evidence/meeting-readiness-20260929/start_scratch.ps1',0,'Free 8784; offline Python 3.12.13 venv; 13 bundled app distributions; ready true; loopback only; no Node runtime.', ['launch.json','launcher.stdout.log','launcher.stderr.log'])
add('api_static_quantiles_byte_binding','PASSED','.venv/Scripts/python.exe -B evidence/meeting-readiness-20260929/rehearse_static.py',0,'HTTP 200; served HTML/JS/CSS exact package bytes; study matches pinned artifact; two fixed indices, all three families, all five ordered quantiles at three horizons; truth-free.',['example-selection.json','rehearsal.json','static-fallback.html','served-study.json','health.json'])
add('in_app_browser_connection','NOT_RUN','MCP browser setup; getForUrl(http://127.0.0.1:8784); discovery agent.browsers.list()',None,'No in-app browser is available; discovery []; coordinator used independently available MCP Playwright, not a denied-action workaround.',['browser-availability.json'])
add('fresh_browser_navigation_rendering_and_fallback','PASSED','Coordinator MCP Playwright against scratch http://127.0.0.1:8784',None,'Coordinator executed real direct/EMA quantiles, index100 pagination, comparisons/limits, packaged raw engineering fallback navigation and desktop/mobile fit. Supplied evidence inspected by implementation writer; no second smoke.',['coordinator-browser.json','coordinator-browser-verification.json'])
add('fresh_screenshots','PASSED','Coordinator MCP Playwright screenshot',None,'Fresh desktop/mobile/raw-observation screenshots retained only in explicitly permitted core/.playwright-mcp; absolute paths and supplied hashes in coordinator evidence. No copies into denied main destination.',['coordinator-browser.json','coordinator-browser-verification.json'])
add('main_checkout_browser_artifact_writes','FAILED','Coordinator MCP Playwright save snapshot and console log to main evidence path',None,'Tool denied those writes; retained at explicitly allowed core paths. No denied write retried through another channel; no screenshots copied into main.',['coordinator-browser.json'])
add('complete_persisted_browser_network_census','NOT_RUN',None,None,'Initial coordinator inspection observed 20 loopback requests, but final saved network buffer was empty after navigation. Do not claim final log proves complete census.',['coordinator-browser-verification.json'])
add('full_eight_minute_spoken_rehearsal','NOT_RUN',None,None,'Planned eight-minute runbook exists; timed automated API/static traversal was 1.973620 seconds, not speaking rehearsal.',['rehearsal.json'])
add('row_level_later_truth_reveal','NOT_APPLICABLE',None,None,'Truth-free replay and retrospective forecast NPZs lack later observations for selected cutoffs; separate static packaged aggregate evidence displayed as such. No new outcomes accessed.',['rehearsal.json','static-fallback.html'])
add('separate_packaged_engineering_later_observation','PASSED','.venv/Scripts/python.exe -B evidence/meeting-readiness-20260929/incorporate_coordinator_evidence.py',0,'Coordinator predetermined first chronological raw row before values; package byte hash and exact row verified. +1 missing/ineligible retained; +3/+6 observed. Uncalibrated response code, not Sv_mean, AEON or JEPA. Browser rendered by coordinator; separate static table generated.',['coordinator-raw-example-selection.json','coordinator-raw-example.json','supplement-verification.json','static-fallback.html'])
add('shutdown_first_attempt','FAILED','powershell.exe -NoProfile -ExecutionPolicy Bypass -File evidence/meeting-readiness-20260929/stop_scratch.ps1',1,'Decoded DateTime was incorrectly regex-parsed; ownership guard refused before mutation; corrected.',['shutdown-first-attempt.json'])
add('shutdown_second_attempt','FAILED','powershell.exe -NoProfile -ExecutionPolicy Bypass -File evidence/meeting-readiness-20260929/stop_scratch.ps1',1,'Null child frontier produced invalid CIM filter before mutation; corrected.',['shutdown-second-attempt.json'])
add('owned_shutdown_and_existing_servers','PASSED','powershell.exe -NoProfile -ExecutionPolicy Bypass -File evidence/meeting-readiness-20260929/stop_scratch.ps1',0,'Validated PID/time/parentage/command/listener; stopped owned serving PID 32600; own chain gone and 8784 closed; protected existing servers unchanged.',['shutdown.json','final-runtime-verification.json'])
add('package_and_archive_after','PASSED',final['package_verifier_command'],0,'Packaged verifier passed 107 assets after serving; archive SHA256 before/after unchanged.',['package-verification.log','final-runtime-verification.json'])
add('document_build_first_attempt','FAILED','build_checklist.py followed by verify_documents.py before yielded prerequisite completed',1,'Prerequisite final-runtime JSON not yet present; subsequent missing-checklist link failure. Waited for prerequisite exit 0, then proceeded sequentially.',['document-build-first-attempt.json'])
add('document_verification_encoding_attempt','FAILED','.venv/Scripts/python.exe -B evidence/meeting-readiness-20260929/verify_documents.py',1,'Default cp1252 document read failed; corrected verifier reads UTF-8 explicitly.',['document-verification-encoding-failure.json'])
add('staging_byte_preservation_first_attempt','FAILED','.venv/Scripts/python.exe -B evidence/meeting-readiness-20260929/stage_owned_files.py',1,'Captured CRLF preservation initially triggered Git trailing-CR reports; scoped cr-at-eol recognition added while retaining substantive whitespace checks.',['staging-byte-preservation-first-attempt.json'])
add('staging_byte_binding_first_attempt','FAILED','.venv/Scripts/python.exe -B evidence/meeting-readiness-20260929/stage_owned_files.py',1,'Ordinary git add retained a previously normalized index entry; staged-byte assertion caught it. Corrected by explicit-path git add --renormalize before byte comparison.',['staging-byte-binding-first-attempt.json'])
add('documents_links_numbers_preservation_scope','PASSED','.venv/Scripts/python.exe -B evidence/meeting-readiness-20260929/verify_documents.py',0,'Focused document/link/number/baseline/diff-scope validation; see actual result record.',['document-verification.json','verify_documents.py'])
add('whitespace','PASSED','git diff --check',0,'No whitespace errors.',['document-verification.json'])
for name,reason in [('production_tdd','No production behavior changed.'),('model_fitting','Outside scope; no scientific execution permission.'),('scientific_suite_repeat','Prohibited by owner task.'),('new_biological_or_commercial_validation','Outside this public-data handoff.'),('package_patch_rebuild_new_release','Prohibited; immutable r3 retained.')]:
    add(name,'NOT_APPLICABLE',None,None,reason,[])
source_pins = [{'path':p,'sha256':h} for p,h in prep['sources'].items()]
for name in ('coordinator-browser.json','coordinator-browser-verification.json','coordinator-raw-example-selection.json','coordinator-raw-example.json','verified-implementation-routing.json'):
    source_pins.append({'path':prefix+name,'sha256':hashlib.sha256((E/name).read_bytes()).hexdigest(),'owner':'root'})
coordinator_browser=load('coordinator-browser.json')
coordinator_verification=load('coordinator-browser-verification.json')
raw_example=load('coordinator-raw-example.json')
historical_browser = []
for path in ['evidence/browser/aeon-scale-expanded-r3-desktop.png','evidence/browser/aeon-scale-expanded-r3-mobile.png','evidence/browser/aeon-scale-expanded-r3-browser-smoke.json']:
    historical_browser.append({'path':path,'sha256':hashlib.sha256((R/path).read_bytes()).hexdigest(),'scope':'Historical approved r3 validation; not new meeting scratch browser evidence.'})
result = {'schema_version':'1.0','meeting_date':'2026-09-29','implementation_status':'partial','objective':'meeting-ready handoff around approved r3',
          'assignment_head':prep['head_before'],'owner_requested_writer':{'model':'gpt-6.1-sol','effort':'high','sole_writer':True,'recursive_agents':False,'runtime_binding':'Verified by root from exact session runtime turn_context; root-owned verified-implementation-routing.json; not self-review.'},
          'self_approval':False,'fresh_review':{'status':'NOT_RUN','disposition':'PENDING_FRESH_DISTINCT_SESSION','owner':'root','scope':'This meeting documentation/evidence increment; existing r3 and diagnostic approvals remain valid in their scopes.'},
          'deliverables':[f'meeting/20260929/{n}' for n in ['MEETING_BRIEF.md','DEMO_RUNBOOK.md','PILOT_DATA_REQUEST.md','NEXT_RESEARCH_DECISION.md','READINESS_CHECKLIST.json']],
          'entrypoints':['README.md','INDEX.md','START_HERE.md','CODEX_MEETING_READY_AFTER_DIAGNOSTIC.md','orchestration/STATUS.md'],
          'package':{'archive':'release/aeon-offline-scale-expanded-20260928-r3.zip','archive_sha256_before':prep['archive_sha256_before'],'archive_sha256_after':final['archive_sha256_after'],
                     'source_commit':'f975d2b0a425a9cbdadbacf9f50d0cad216d93c1','scratch_absolute':prep['package'],'scratch_relative':Path(prep['package']).relative_to(R).as_posix(),
                     'sha256sums_sha256':prep['manifest_sha256'],'source_revision_sha256':prep['source_revision_sha256'],'hashed_assets':107,
                     'internal_review_pending':'Immutable build-time state; external exact-byte approval governs unchanged r3.',
                     'external_review':'orchestration/reviews/AEON_SCALE_EXPANDED_OFFLINE_R3_REVIEW_20260929.json'},
          'diagnostic':{'source_commit':'991368d6a64c7189d803828a15a39828a00e8ea0','implementation_commit_prefix':'239a8b4','integration_commit_prefix':'70834f8',
                        'scope':'Completed approved exposed TRAIN/VAL diagnostic; later than r3; no new scientific execution authorized.'},
          'source_pins':source_pins,'prerequisites':{'shell':'Windows PowerShell','uv':launch['uv'],'python':'Already cached offline CPython 3.12; actual 3.12.13','node':'NOT_REQUIRED','browser':'In-app connection unavailable; coordinator available MCP Playwright supplied fresh visual smoke without dependency installation.'},
          'launch':launch,'shutdown':shutdown,'rehearsal':{'kind':rehearsal['rehearsal_kind'],'actual_seconds':rehearsal['elapsed_seconds'],'planned_speaking_minutes':8,'selected_indices':[0,100],
                     'examples':[{'index':x['chronological_index'],'interval':x['cutoff_interval_id'],'source_timestamp':x['cutoff_source_timestamp'],'row_id':x['row_id']} for x in rehearsal['examples']],
                     'full_quantiles_and_provenance':prefix+'rehearsal.json','later_truth':'Absent for chosen AEON cutoffs; static released aggregate report is the fallback.'},
          'checks':checks,'historical_browser_evidence':historical_browser,
          'fresh_coordinator_browser_evidence':{'executor':'coordinator','browser':coordinator_browser,'verification':coordinator_verification,'independent_final_review':False},
          'separate_raw_engineering_example':raw_example,
          'gates':{'software':'Existing exact r3 offline research approval; fresh API/static checks and coordinator MCP Playwright visual checks passed; in-app Browser attempt NOT_RUN.',
                   'experiment':'Preserved completed bounded studies with original mixed outcomes and limitations; no new experiment.',
                   'jepa_value':'Positive original within-study retrospective rule only; negative original secondary transfer, mixed development and unresolved mechanism attribution.',
                   'biology':'NOT_EVALUATED','marine_operational_commercial':'NOT_EVALUATED'},
          'scope_limits':['No fitting, checkpoint inference, evaluation, diagnostic rerun, new data or outcomes.','Saved truth-free AEON replay; no AEON row-level later observation reveal. Separate already released MOSAiC response-code engineering observations are explicitly labelled.','Publisher-conditioned Sv_mean; source intervals; source timezone/availability unresolved.','Prior-year data is TRAIN for expanded models and affected descendants.','No generalization, SOTA, species, biomass, catch, fuel saving, causal device effect, production integration or business-value claim.','Original archives/extractions and scientific ledgers/reports/models/predictions unchanged.'],
          'blockers':['Spoken approximately eight-minute rehearsal NOT_RUN; timed automated steps and coordinator actual browser navigation recorded without fabricated speaking time.','Fresh distinct meeting review pending.'],
          'remaining_not_run':['Unavailable in-app Browser attempt','Complete persisted network census after buffer reset','Spoken full eight-minute rehearsal','Browser clicks on separately generated static HTML; its data and links were locally verified.'],
          'decision':'NO_FURTHER_FITTING_BEFORE_MEETING','conditional_intervention':'DRAFT / NOT_AUTHORIZED / NOT_QUEUED; unchanged diagnostic thresholds; not automatic authorization.',
          'taskfile_judgment':'Missing taskfile consolidated from full owner message; no historical scientific execution permission inferred.',
          'local_commit':'Containing local implementation commit, reported by git rev-parse HEAD after commit; root owns final approval bookkeeping.',
          'cost_usd':0,'publishing_contact_or_push':False}
(M/'READINESS_CHECKLIST.json').write_text(json.dumps(result,indent=2,ensure_ascii=True)+'\n',encoding='utf-8')
print('Wrote checklist from actual execution evidence; fresh review remains pending.')
