"""Incorporate supplied coordinator browser evidence without modifying its files."""
import hashlib
import html
import json
from pathlib import Path

E=Path(__file__).resolve().parent
R=E.parents[1]
M=R/'meeting/20260929'

def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def replace(path, old, new):
    content=path.read_text(encoding='utf-8')
    assert old in content,(path,old)
    path.write_text(content.replace(old,new),encoding='utf-8',newline='\n')

prep=load(E/'preparation.json')
package=Path(prep['package'])
coordinator=load(E/'coordinator-browser.json')
verification=load(E/'coordinator-browser-verification.json')
raw_example=load(E/'coordinator-raw-example.json')
raw_selection=load(E/'coordinator-raw-example-selection.json')
raw_path=package/'artifacts/raw-development.json'
raw=load(raw_path)
assert coordinator['status']=='PASSED' and coordinator['base_url']=='http://127.0.0.1:8784'
assert hashlib.sha256(raw_path.read_bytes()).hexdigest()==raw_example['packaged_artifact_sha256']
assert raw_selection['fixed_sorted_row_index']==0
first=sorted(raw['rows'],key=lambda row:row['cutoff_utc'])[0]
assert first==raw_example['row']
assert raw['calibrated'] is False and raw['final_evaluation'] is False
for key in ('source_result_sha256','review_record_sha256'):
    assert raw[key]==raw_example[key]
for horizon in first['horizons']:
    for key in ('ridge_quantiles_code','direct_quantiles_code'):
        assert horizon[key]==sorted(horizon[key])
section=['<h2>Separate engineering later-observation example</h2>',
         '<p>FIRST chronological packaged MOSAiC TRAIN-development row, fixed by coordinator before inspecting values. UNCALIBRATED RESPONSE CODE, NOT Sv_mean, NOT AN AEON OR JEPA RESULT. No final evaluation. This observation table is separate from truth-free AEON replay.</p>',
         '<p>Cutoff '+first['cutoff_utc']+'</p>',
         '<table><tr><th>Horizon hours</th><th>Target start UTC</th><th>Eligibility</th><th>Later observation code</th><th>Ridge q05/q25/q50/q75/q95 codes</th><th>Direct q05/q25/q50/q75/q95 codes</th></tr>']
for row in first['horizons']:
    truth='MISSING / NOT SCORED' if row['truth_code'] is None else f"{row['truth_code']:.6f}"
    section.append('<tr><td>'+str(row['horizon_hours'])+'</td><td>'+row['target_start_utc']+'</td><td>'+str(row['eligible'])+'</td><td>'+truth+'</td><td>'+' / '.join(f'{v:.6f}' for v in row['ridge_quantiles_code'])+'</td><td>'+' / '.join(f'{v:.6f}' for v in row['direct_quantiles_code'])+'</td></tr>')
section+=['</table>','<p>Packaged raw-development.json SHA256 '+raw_example['packaged_artifact_sha256']+'</p>',
          '<p>Source result SHA256 '+raw_example['source_result_sha256']+'; review SHA256 '+raw_example['review_record_sha256']+'</p>']
replace(E/'static-fallback.html','</html>','\n'.join(section)+'\n</html>')
supplement={'status':'PASSED','executor':'implementation writer inspecting coordinator evidence and released package only',
            'coordinator_browser':{'path':'evidence/meeting-readiness-20260929/coordinator-browser.json','sha256':hashlib.sha256((E/'coordinator-browser.json').read_bytes()).hexdigest()},
            'coordinator_verification':{'path':'evidence/meeting-readiness-20260929/coordinator-browser-verification.json','sha256':hashlib.sha256((E/'coordinator-browser-verification.json').read_bytes()).hexdigest()},
            'raw_selection':raw_selection,'packaged_artifact_sha256':raw_example['packaged_artifact_sha256'],
            'raw_first_row':first,'scope':'Already released uncalibrated MOSAiC TRAIN development; no new outcomes, inference or evaluation; AEON remains truth-free.',
            'permissions':'No browser artifacts copied into denied main destination. External core paths and supplied hashes referenced only.',
            'complete_persisted_network_census':'NOT_RUN_FINAL_BUFFER_EMPTY_AFTER_NAVIGATION',
            'runtime_routing_evidence':'evidence/meeting-readiness-20260929/verified-implementation-routing.json'}
(E/'supplement-verification.json').write_text(json.dumps(supplement,indent=2)+'\n',encoding='utf-8')
replace(M/'MEETING_BRIEF.md','Historical MOSAiC raw-count views belong to a different engineering demonstration.',
        'A separate first chronological packaged MOSAiC response-code example supplies later observations (+1 missing, +3/+6 present), labelled uncalibrated engineering evidence, never AEON or JEPA truth.')
replace(R/'README.md','Fresh browser smoke and speaking rehearsal are NOT_RUN; launcher/API/static traversal passed.',
        'Fresh coordinator MCP Playwright smoke passed; the unavailable in-app browser attempt and spoken rehearsal remain NOT_RUN. Launcher/API/static checks passed. A separately labelled packaged MOSAiC response-code example supplies later observations.')
replace(R/'START_HERE.md','Fresh browser/screenshot checks and a spoken eight-minute rehearsal are NOT_RUN; launcher/API/static checks passed.',
        'Fresh coordinator MCP Playwright browser/screenshot checks passed; the unavailable in-app browser attempt and spoken eight-minute rehearsal remain NOT_RUN. Launcher/API/static checks passed.')
replace(R/'INDEX.md','Fresh meeting review is pending; current browser smoke and speaking rehearsal are NOT_RUN.',
        'Fresh meeting review is pending; coordinator MCP Playwright smoke passed, while the in-app browser attempt and spoken rehearsal remain NOT_RUN.')
replace(M/'DEMO_RUNBOOK.md','**Fresh browser smoke and a spoken eight-minute rehearsal are NOT_RUN** because the MCP browser connection returned no available browser.',
        '**Fresh coordinator MCP Playwright browser smoke passed; the unavailable in-app browser attempt and a spoken eight-minute rehearsal remain NOT_RUN.** The separate coordinator connector supplied actual scratch navigation and screenshots.')
replace(M/'DEMO_RUNBOOK.md','| 3:00–4:00 | Explain that replay has no later truth and performs no checkpoint inference. Open the separate static fallback\'s **Later observations** notice and **Already released retrospective evidence**. This is aggregate evidence, not a reveal for those examples. Do not use raw archives or new CAL/TEST outcomes. |',
        '| 3:00–4:00 | Explain that AEON replay has no later truth and performs no checkpoint inference. Navigate to **#experiment-lab / Raw response-code TRAIN development**: the separately predetermined first chronological MOSAiC row shows saved quantiles and later observations. State **UNCALIBRATED RESPONSE CODE, NOT Sv_mean, NOT AN AEON OR JEPA RESULT**; +1 is missing/ineligible, +3/+6 are observed. Use the separate static fallback table if needed. No raw archive or new CAL/TEST outcome is opened. |')
old='Fresh browser navigation, quantile rendering, comparison/limits rendering, fallback navigation and screenshots are **NOT_RUN**. MCP setup returned `No browser is available`; discovery returned `[]`. No alternative browser channel or browser dependency installation was used.'
new='The implementation session\'s in-app browser setup returned `No browser is available`; discovery returned `[]`, so that attempt remains **NOT_RUN**. Separately, the coordinator\'s available **MCP Playwright** connector completed real scratch navigation, direct/EMA quantile rendering, fixed-index-100 pagination, comparisons/limits, raw engineering fallback navigation and desktop/mobile screenshots. [Coordinator browser evidence](../../evidence/meeting-readiness-20260929/coordinator-browser.json) and [verification](../../evidence/meeting-readiness-20260929/coordinator-browser-verification.json) identify the executor and limits. Viewports were 1440/390 pixels with document widths 1425/375 (scrollbars), fitting both. A favicon 404 was nonblocking; no JavaScript console error was observed. The initial network inspection observed 20 requests, all loopback; the final saved network buffer was empty after navigation, so a complete persisted network census is **NOT_RUN**. Separate generated static HTML was checked for data/links locally, not browser-clicked. No browser dependency installation occurred.'
replace(M/'DEMO_RUNBOOK.md',old,new)
addition='''
### Fresh coordinator screenshots and permission boundary

The MCP tool denied snapshot/log writes into the main checkout. The coordinator retained artifacts only in its explicitly allowed `C:/Users/Álvaro Schwiedop/Desktop/KriptaStudios/marine-echo-jepa-core/.playwright-mcp/` directory. No denied main write was retried through another channel and no images were copied into main. Exact external paths/hashes are in coordinator-browser-verification.json. Fresh files include `meeting-ready-20260929-1440.png`, `meeting-ready-20260929-390.png`, `meeting-ready-20260929-raw-observation.png`, initial/raw snapshots and console/network logs. They are coordinator execution evidence, not an independent final review.

### Packaged later-observation example, separate scope

The coordinator fixed the **first chronological raw row** before inspecting values: cutoff `2020-04-02T00:00:00Z` in packaged `artifacts/raw-development.json`. Its SHA-256 is `ad3b69a29395fb83528cd2e015c86e6255a2e3be42b6b7a450ca0f37db310de1`; source result `4237f375ba129fce95be952325e56cf5bc9615a5d89b2ed9ba91dfc8771cb49d`; review record `18d8c890dfef8aacc274308d94251b91059c76fc743071d1cc1bcd61cc584b15`. [Selection](../../evidence/meeting-readiness-20260929/coordinator-raw-example-selection.json), [full quantiles/observation excerpt](../../evidence/meeting-readiness-20260929/coordinator-raw-example.json) and [implementation verification](../../evidence/meeting-readiness-20260929/supplement-verification.json) bind this choice to the package without new evaluation.

| Horizon hours | Target start UTC | Later response-code observation | Eligibility |
| --- | --- | ---: | --- |
| 1 | 2020-04-02T00:00:00Z | Missing / not scored | Ineligible |
| 3 | 2020-04-02T02:00:00Z | 11218.386223 | Eligible |
| 6 | 2020-04-02T05:00:00Z | 11229.620260 | Eligible |

The complete ridge/direct five-quantile values are in the static table and excerpt. This is **UNCALIBRATED RESPONSE CODE, NOT Sv_mean, NOT AN AEON OR JEPA RESULT**. It is TRAIN-development engineering evidence without final evaluation. Keep the missing +1 row visible; do not filter for favorable/error-free cases. The static table supplements the AEON aggregate fallback while AEON row-level truth remains absent.
'''
with (M/'DEMO_RUNBOOK.md').open('a',encoding='utf-8') as out:
    out.write(addition)
original=(E/'status-original.txt').read_bytes()
status=R/'orchestration/STATUS.md'
content=status.read_bytes()
assert content.endswith(original)
prepend=content[:-len(original)].decode('utf-8')
prepend=prepend.replace('Fresh MCP\nbrowser smoke/screenshots are NOT_RUN (no available browser); released static\nevidence and historical reviewed screenshots are linked.',
                        'Fresh coordinator MCP Playwright smoke/screenshots passed; in-app browser\nattempt was NOT_RUN. Packaged uncalibrated MOSAiC later observations supplement\ntruth-free AEON replay; denied main artifact writes remain explicitly recorded.')
status.write_bytes(prepend.encode('utf-8')+original)
print('Incorporated coordinator browser and fixed packaged engineering observation evidence; no root-owned files modified.')
