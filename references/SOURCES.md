# Source ledger

Preparation date: 2026-09-26. Primary sources below support dataset/model/tool facts. Engineering
choices, dates in the two-week plan and thresholds are our proposals, not claims from a paper.
Catalog/HTML access does not prove successful multi-gigabyte download, calibration or training.

| ID | Source and URL | What it supports / limitation |
|---|---|---|
| S01 | Marine M3iGO: https://www.marineinstruments.es/news/marine-instruments-launches-the-m3igo-the-first-satellite-fishing-buoy-with-artificial-intelligence/ ; MSB: https://www.marineinstruments.es/products/software-msb/ | Existing acoustic AI/software context. Proposed integration only. |
| S02 | De La Torre et al.,2022, PANGAEA949811: https://doi.org/10.1594/PANGAEA.949811 ; HTML https://doi.pangaea.de/10.1594/PANGAEA.949811?format=html | Arctic downward AZFP, catalog168days,38/125/200/455kHz, raw/config/GPS/Iridium,4.4GB archive,CC BY4. Binary unverified. |
| S03 | Official Echopype OOI example: https://echopype-examples.readthedocs.io/en/latest/OOI_eclipse.html | Real upward moored EK60 and raw path. The example day is not a long benchmark. |
| S04 | OOI data acknowledgment: https://oceanobservatories.org/how-to-use-acknowledge-and-cite-data/ ; instrument: https://oceanobservatories.org/instrument-class/zpls/ | Freely available data with source/NSF acknowledgment; sonar/platform context. Verify chosen archive. |
| S05 | Echopype releases: https://github.com/echostack-org/echopype/releases ; docs: https://echopype.readthedocs.io/ | Calibration/conversion library and current compatibility. Pin tested version. |
| S06 | Echopype paper: https://arxiv.org/abs/2111.00187 | Open-source sonar processing background; not our implementation results. |
| S07 | LeJEPA: https://arxiv.org/abs/2511.08544 | SIGReg/predictive representation motivation; original domain/results differ. |
| S08 | LeWorldModel: https://arxiv.org/abs/2603.19312 ; code: https://github.com/lucas-maes/le-wm | Compact end-to-end visual world model; our passive acoustic adaptation is not direct reproduction. |
| S09 | TS-JEPA: https://arxiv.org/abs/2509.25449 | Temporal JEPA reference; not a public marine benchmark. |
| S10 | Chronos-2: https://arxiv.org/abs/2510.15821 | Optional frozen numerical forecasting baseline; inspect current weights/license. |
| S11 | Official Codex subagents: https://learn.chatgpt.com/docs/agent-configuration/subagents | Native standalone agent TOML and global agent settings checked at preparation. |
| S12 | Official models: https://developers.openai.com/api/docs/models/gpt-6-astra ; https://developers.openai.com/api/docs/models/gpt-6-sol ; https://developers.openai.com/api/docs/models/gpt-5.6-sol ; https://developers.openai.com/api/docs/models/gpt-6-luna | Requested model identifiers/efforts; account access remains to verify. |
| S13 | Codex config: https://developers.openai.com/codex/config-reference | Configuration contract, version-dependent. |
| S14 | Codex CLI: https://developers.openai.com/codex/cli/reference | Model,working-directory and config override flags. |
| S15 | PyTorch installation selector: https://pytorch.org/get-started/locally/ | Official compatible package selection; actual SM120 kernel test required. |
| S16 | Runpod: https://www.runpod.io/pricing | Dated listed rates0.74USD/h4090 and0.99USD/h5090, not an all-in quote. |
| S17 | User repository, read through GitHub connector: https://github.com/Kripta-Studios/Kaleido-Project/blob/main/README.md | Bounded hybrid trajectory evidence and rejected ETA/delay scope. Not rerun here. |
| S18 | JEPA-Anything: https://arxiv.org/abs/2609.20800 | Recent predictive-factorization reference, deliberately outside P0. |
| S19 | OOI access announcement,2026-08-26: https://oceanobservatories.org/2026/08/ooi-user-accounts-coming-for-data-access/ | Future free account requirement announced; notice says implementation date will follow. Verify at execution, do not bypass. |
| S20 | Echopype compute_Sv: https://echopype.readthedocs.io/en/stable/api/echopype.calibrate.compute_Sv.html | AZFP requires supplied environmental parameters; instrument XML alone does not settle water environment. |

## Dataset attribution to preserve

De La Torre, Pedro R; Berge, Jørgen; Granskog, Mats A; Katlein, Christian; Divine, Dmitry V;
Raphael, Ian; Geoffroy, Maxime; Vogedes, Daniel; Itkin, Polona; Daase, Malin; Zolich, Artur;
Cottier, Finlo (2022): Data from downward looking Acoustic zooplankton and fish profiler (AZFP)
deployed on drifting sea ice in the Arctic during MOSAiC expedition. PANGAEA.
https://doi.org/10.1594/PANGAEA.949811 . License: CC BY4.0.

For a derived release add the exact preprocessing version and state modifications, aggregation,
masking and included time range. OOI uses its own prescribed source/NSF acknowledgment.

## Verification boundary

No external archive/checkpoint is in this ZIP. The supplied access evidence records DNS failure
from the preparation container and503 responses from direct download attempts; the catalog and
linked filenames were visible. The owner's machine must perform the actual access/parse test.
No source date or model result should be updated by guessing. Record retrieval timestamps and
pin revised artifacts when the implementation agent checks them.
