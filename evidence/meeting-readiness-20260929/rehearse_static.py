"""Focused served-byte/API smoke and timed static traversal; no model imports."""
import hashlib
import html
import json
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

E = Path(__file__).resolve().parent
R = E.parents[1]
prep = json.loads((E/'preparation.json').read_text())
launch = json.loads((E/'launch.json').read_text(encoding='utf-8-sig'))
package = Path(prep['package'])
selection = json.loads((E/'example-selection.json').read_text())
base = launch['address']
started = time.perf_counter()
steps = []
responses = []

def request(path, name):
    tick = time.perf_counter()
    with urllib.request.urlopen(base+path, timeout=10) as response:
        body = response.read()
        status = response.status
    assert status == 200
    (E/name).write_bytes(body)
    responses.append({'path': path, 'status': status, 'seconds': time.perf_counter()-tick,
                      'evidence': name, 'sha256': hashlib.sha256(body).hexdigest()})
    return body

def step(name, tick):
    steps.append({'step': name, 'elapsed_seconds': time.perf_counter()-tick})

tick = time.perf_counter()
health = json.loads(request('/health','health.json'))
assert health['ready'] is True
index = request('/','served-index.html')
assert index == (package/'web/index.html').read_bytes()
for asset in re.findall(r'(?:src|href)="(/assets/[^\"]+)"', index.decode()):
    data = request(asset,'served-'+asset.rsplit('/',1)[1])
    assert data == (package/'web'/asset.lstrip('/')).read_bytes()
step('Readiness and exact served HTML/JS/CSS byte binding',tick)
tick = time.perf_counter()
study = json.loads(request('/api/v1/studies/aeon','served-study.json'))
assert study == json.loads((package/'artifacts/aeon-study.json').read_text())
assert study['retrospective_test']['issued_rows'] == 1216
assert study['limitations']
step('Complete packaged study comparison and limitations',tick)
tick = time.perf_counter()
packaged_replay = json.loads((package/'artifacts/aeon-test-replay.json').read_text())
ids = [r['cutoff_interval_id'] for r in packaged_replay['rows']]
assert ids == sorted(ids)
models = list(study['retrospective_test']['models'])
examples = []
for chosen in selection['indices']:
    expected = packaged_replay['rows'][chosen]
    example = {k: expected[k] for k in ('row_id','cutoff_interval_id','cutoff_source_timestamp')}
    example.update({'chronological_index': chosen, 'models': {}})
    for model in models:
        query = urllib.parse.urlencode({'model_id': model,'offset':chosen,'limit':1})
        path = '/api/v1/studies/aeon/replay?'+query
        replay = json.loads(request(path,f'replay-{chosen}-{model}.json'))
        assert replay['total'] == 1216
        assert replay['quantile_levels'] == [.05,.25,.5,.75,.95]
        assert replay['horizon_source_interval_steps'] == [1,3,6]
        row = replay['rows'][0]
        assert row['row_id'] == expected['row_id']
        assert row['quantiles_db'] == expected['predictions'][model]
        assert not any(k in row for k in ('truth','targets','later_observations'))
        for q in row['quantiles_db']:
            assert len(q) == 5 and all(a<=b for a,b in zip(q,q[1:]))
        example['models'][model] = row['quantiles_db']
    examples.append(example)
step('Two predetermined chronological examples, all model quantiles, truth-free schema',tick)
tick = time.perf_counter()
report_path = package/'provenance/retrospective_test/AEON_RETROSPECTIVE_OUTCOME_20260927.md'
report = report_path.read_text(encoding='utf-8')
limits = (package/'LIMITATIONS.md').read_text(encoding='utf-8')
assert '0.533954' in report and '0.563387' in report
later = {'status':'NOT_APPLICABLE', 'reason':'AEON replay and packaged retrospective forecast NPZs contain no later observations for these cutoffs. Static fallback presents already released aggregate outcome evidence, not per-row truth.',
         'inspected_npz_keys':['row_ids','quantiles_db'], 'source': report_path.relative_to(R).as_posix(),
         'source_sha256':hashlib.sha256(report_path.read_bytes()).hexdigest()}
parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Meeting static fallback</title>',
         '<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:16px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #999;padding:6px}pre{white-space:pre-wrap}a{overflow-wrap:anywhere}</style>',
         '<h1>Meeting static fallback: reviewed r3 saved forecasts</h1>',
         '<p>Publisher-conditioned Sv_mean in dB re 1 m^-1; +1/+3/+6 source intervals. Source timezone and availability unresolved. No live inference, later-truth reveal or new evaluation.</p>']
for example in examples:
    parts.append(f"<h2>Chronological index {example['chronological_index']}: {example['cutoff_source_timestamp']}</h2><p>Interval {example['cutoff_interval_id']}; row {example['row_id']}</p>")
    parts.append('<table><tr><th>Model</th><th>Source horizon</th><th>q05</th><th>q25</th><th>q50</th><th>q75</th><th>q95</th></tr>')
    for model, forecasts in example['models'].items():
        for horizon, values in zip([1,3,6],forecasts):
            parts.append('<tr><td>'+html.escape(model)+'</td><td>'+str(horizon)+'</td>'+''.join(f'<td>{value:.6f}</td>' for value in values)+'</tr>')
    parts.append('</table>')
parts += ['<h2>Later observations</h2><p>'+html.escape(later['reason'])+'</p>',
          '<h2>Already released retrospective evidence</h2><pre>'+html.escape(report)+'</pre>',
          '<h2>Package limits</h2><pre>'+html.escape(limits)+'</pre>',
          '<h2>Fallback navigation</h2>']
for name in ('REPORT.md','LIMITATIONS.md','MODEL_CARD.md','MARINE_DATA_REQUEST.md','provenance/retrospective_test/AEON_RETROSPECTIVE_OUTCOME_20260927.md'):
    import os
    link = Path(os.path.relpath(package/name,E)).as_posix()
    assert (E/link).exists()
    parts.append(f'<p><a href="{urllib.parse.quote(link)}">{html.escape(name)}</a></p>')
parts.append('</html>')
(E/'static-fallback.html').write_text('\n'.join(parts)+'\n',encoding='utf-8')
step('Read packaged outcome/limits and generate separate static evidence navigation',tick)
record = {'started_at': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds':time.perf_counter()-started,
          'rehearsal_kind':'Timed automated API/static traversal; not a spoken eight-minute rehearsal',
          'steps':steps,'responses':responses,'examples':examples,'later_observation_display':later,
          'browser':{'status':'NOT_RUN','reason':'MCP browser connection returned No browser is available; discovery returned []; no dependencies installed or alternate browser channel used.',
                     'fresh_screenshots':'NOT_RUN', 'historical_screenshots':['evidence/browser/aeon-scale-expanded-r3-desktop.png','evidence/browser/aeon-scale-expanded-r3-mobile.png']},
          'static_fallback':'evidence/meeting-readiness-20260929/static-fallback.html'}
(E/'rehearsal.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'elapsed_seconds':record['elapsed_seconds'],'steps':steps,'examples':examples,'later_observation_display':later},indent=2))
