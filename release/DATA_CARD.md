# Data card — Marine Echo JEPA engineering diagnostic

**Snapshot:** 2026-09-26
**Status:** Engineering-only, incomplete P0. The diagnostic is real public acoustic data, but it
is raw-count data and is not a calibrated forecast benchmark.

## Primary source

The primary source is the owner-supplied local copy of dataset `mosaic_azfp_down_2020`, catalogued
by PANGAEA as a 2020 MOSAiC downward-looking AZFP deployment. The catalog reports four nominal
frequencies (38, 125, 200, and 455 kHz) and a 168-day deployment period from 2020-02-16 to
2020-08-02. Catalog duration is not evidence that every day passed quality or benchmark
eligibility checks. Source record: [PANGAEA 949811](https://doi.org/10.1594/PANGAEA.949811).

The five local input files were inventoried in `data/manifests/mosaic_azfp_down_2020/source_inventory.json`.
Hashes below identify those local copies; no publisher-provided SHA-256 or signature was
available.

| File | Bytes | Local SHA-256 |
|---|---:|---|
| `azfp55170-bioacoustics.zip` | 4,676,515,350 | `bc9cf56bd92e7dd1c83cfca6d296585a832e13e27635c04f14ff864bf2831ab9` |
| `POPE306-308-mosaic-manual.pdf` | 705,497 | `d491cb47ad5d569c60a057751eb741eba2acc8f9075460c9b7e6fe9efe48c2bf` |
| `iridium-msgs.txt` | 206,050 | `44096a09096aa3bed2cfb6253948152c39a6bc4f3b8e495ba100069c08eb5d6a` |
| `xeos-rover-mosaicdown-transponder-16feb-02aug2020.csv` | 135,837 | `031e563c1d8aa2b851997f5f239ace571561b12a58c98eaecc56676b0efdc946` |
| `readme.txt` | 1,779 | `16b9fc8dc0d76b2631aa60c3dccb0ca89e6d8942c8c8fb0376be68bccbbae5fb` |

The archive SHA-256 and ZIP CRC checks establish integrity of the local copy, not publisher
authenticity. The archive contained 3,744 members and 10,169,592,780 declared expanded bytes.
The full ZIP integrity scan returned no bad member. Safe extraction streamed member CRC and
SHA-256 checks before publishing the extracted tree. Two instrument XML and two DPL files were
preserved. The original ZIP remains intact in the local source directory. The instrument manual
is not included in this meeting package because it contains legacy operational details; its
integrity hash is recorded above.

## Diagnostic derivative

The demonstrator uses a bounded one-day derivative recorded at
`evidence/data/diagnostic_raw_replay_2020-02-17.json` (SHA-256
`68240f4e6d75eef0c6890dbbb658468b45ff0f4ea95eb2ec1e41084de200beea`). It covers 2020-02-17 and
uses 24 `.01B` chunks, 5,564 raw pings, and 96 observed trailing 15-minute bins. Every bin is
represented by four channel summaries over 64 sample-index bands. Values are mean raw digitizer
counts, with unit `raw_counts`, `calibrated=false`, and range convention `sample_index`.

This is a diagnostic aggregation, not the canonical dataset pipeline or an eligible evaluation
set. Same-hour `.01A` chunks remain in the source inventory; they were excluded from this single
mode-specific replay only. No sample index is converted to metres or biological depth. No
fish/species, biomass, catch, or operational label is present in this derivative.

## Calibration and measurement units

The archive XML and DPL identify downward AZFP serial 55170 and include a configured sound speed
of 1465 m/s. The platform manual lists a different downward instrument serial, 55169. The manual
is treated as platform context, not as instrument-specific calibration evidence. The parsed
AZFP data lacks verified seawater temperature, salinity, pressure, and absorption provenance;
the XML reports no installed pressure sensor. Internal/sonar thermistor values are not accepted
as seawater measurements. Required environmental fields for physical calibration were absent or
not verified, so `Sv`, dB, physical range, and a physical backscatter index are unavailable.

The replay therefore remains in raw counts on sample-index bands. Any future calibrated output
requires instrument-matched configuration plus authoritative environmental/calibration evidence
and review; this card does not supply or infer those values.

## Separate OOI feasibility sample

A distinct OOI upward-looking EK60 fallback probe used one raw file from 2017-08-21, 52,436,316
bytes, local SHA-256 `27248617a52ace72f1d4a39d3d4c3d1e350316f842ed4e0446013a04ce399f7c`. A pinned
Echopype 0.11.1 code path used indicative instrument/environment fields carried in that raw file;
an xarray output round-trip equality check passed. This is a code-path feasibility result, not
independent field calibration, a complete daily corpus, or an eligible benchmark. No broad OOI
download occurred and the primary dataset was not switched. A coarse 90-day acquisition estimate
of about 63.6 GiB exceeds the configured 40 GiB raw-data limit; further OOI work is blocked pending
an owner-approved resource plan. OOI data, if later used, requires the source/NSF acknowledgment
described by [OOI data guidance](https://oceanobservatories.org/how-to-use-acknowledge-and-cite-data/).

## Intended access, retention, and citation

PANGAEA lists the primary source as CC BY 4.0. Cite it as:

> De La Torre, Pedro R.; Berge, Jørgen; Granskog, Mats A.; Katlein, Christian; Divine, Dmitry V.;
> Raphael, Ian; Geoffroy, Maxime; Vogedes, Daniel; Itkin, Polona; Daase, Malin; Zolich, Artur;
> Cottier, Finlo (2022): Data from downward looking Acoustic zooplankton and fish profiler (AZFP)
> deployed on drifting sea ice in the Arctic during MOSAiC expedition. PANGAEA.
> https://doi.org/10.1594/PANGAEA.949811. License: CC BY 4.0.

The source archive and extracted instrument records are local research inputs, not contents of
this documentation package. Any derived release must carry this attribution and its own
preprocessing version, time range, aggregation, and integrity manifest. Do not redistribute the
manual or its operational details.


## Later evidence update

The internet search found and downloaded MOSAiC environmental profiles (DOI 10.18739/A21J9790B). Their spatial/temporal applicability and calibration procedure remain unapproved; see ADR0003. Earlier missing-input statements describe the initial supplied files, not the absence of public environmental data. Model software tests and a 100-update CUDA resource profile passed using synthetic fixtures, not benchmark data. All three configured native agents subsequently returned usage-limit errors. Final independent scientific/release review and Sol review of these Luna-authored drafts remain incomplete.
