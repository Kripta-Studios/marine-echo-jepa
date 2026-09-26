# Datasets and access plan

## D1 — primary: MOSAiC downward-looking AZFP

Source [S02]: https://doi.pangaea.de/10.1594/PANGAEA.949811?format=html

The catalog describes a 2020 deployment with four channels (38, 125, 200, 455 kHz), downward
orientation, hourly raw-file groups, configuration files, GPS and Iridium messages. The listed
acoustic ZIP is approximately 4.4 GB; licence CC BY 4.0. Preserve the catalog's complete citation.
There are no tuna catch labels. Buoy internal temperature must not be renamed seawater temperature.
A 5 m cable does not establish exact transducer depth or attitude at every instant.

Direct files are enumerated in `configs/datasets.json`, including:

- `azfp55170-bioacoustics.zip`
- `readme.txt`
- `xeos-rover-mosaicdown-transponder-16feb-02aug2020.csv`
- `iridium-msgs.txt`
- `POPE306-308-mosaic-manual.pdf`

**Access state at handoff:** catalog/links/licence verified; binary downloads not validated.
The web fetch returned 503 for sampled binary/text endpoints, and the local runtime could not
resolve external hosts. This is not proof that an end user's connection will fail. The archive
has not been downloaded, extracted, parsed or hashed here. The catalog's all-files ZIP/TAR
aggregator mentions account signup; direct-link access must be tested separately. Respect any
actual login/terms requirement instead of bypassing it.

## D2 — contingency: OOI moored EK60

Source [S03] is an executable Echopype example using an upward-looking OOI echosounder.
It gives the raw archive route:

`https://rawdata.oceanobservatories.org/files/CE04OSPS/PC01B/ZPLSCB102/2017/08/21/`

The documented example is only one day: **not enough for the intended forecast benchmark**.
After validating that exact route, discover permitted contiguous adjacent dates from the
actual archive index; do not invent file names or assume complete months. Save every listing.
Discover and record OOI use/acknowledgment terms [S04]. Instrument orientation and depths differ
from D1. Use a separate dataset contract, model run family and result label.

## Acquisition sequence

1. Probe the small metadata files and the archive without downloading all bytes. Record HTTP
   status, actual final URL, MIME, sizes when available and authentication requirements.
2. For D1, download readme/configuration documentation and the ZIP with an 8 GiB per-file
   ceiling. Download serially or with at most two workers. Store source and local SHA-256.
3. List archive members before extraction. Reject traversal, absolute paths, symlinks,
   duplicate normalized names and a total expanded size above the configured ceiling.
4. Extract to a new dataset-version directory. Inventory raw/config file pairs and sampling modes.
5. Parse one hour and calibrate one channel. Inspect metadata and values; then a 48-hour train-only slice.
6. Only after the parser/QC gate, process the full permitted D1 recording incrementally.
7. If D1 access or calibration is blocked, test D2 using its documented example. Limit fallback
   discovery to half a working day; let UI/contracts/tests continue in parallel.

## Data budget

Default total downloaded raw-data ceiling 40 GiB; D1 preferred. Default expanded-data ceiling
80 GiB and minimum free space 100 GiB before the complete pipeline. These are resource policies,
not claims about actual expanded size. If less disk is available, process hourly, retain the
original archive, preserve checksums and retain only approved needed intermediates; never delete
raw sources silently. OOI downloads can be much larger than the D1 archive; check sizes first.

## Eligibility

For D1, aim for the full available record; benchmark eligibility requires at least 90 usable
chronological days, at least 12 valid calibration days and 20 valid test days after purging.
If fewer exist, retain the original date-selection rule and report a diagnostic study rather
than shortening horizons or choosing dates after viewing errors to force a positive result.
D2 uses its own registered 60/15/10/15 chronological split and the same eligibility counts;
less data supports an engineering/diagnostic result, not full acceptance.

Do not use random ship-survey segments as a replacement for temporal buoy observations.
Do not download ImageNet, fish photographs, AIS or shrimp audio simply to make the project look
multimodal. Optional external environmental data require matching place/time and issuance metadata;
they are deliberately excluded from the two-week critical path.

## Access-policy update

OOI announced on 2026-08-26 that free user accounts will be introduced, with an implementation
date to be announced [S19]. Verify current requirements when downloading. Do not assume
anonymous access is permanent or infer that an account currently is required for every endpoint.

## Fallback is a different configuration

OOI has its own orientation, nominal channels and supported ranges. Generate a separate
dataset-specific model/preprocessing configuration before training; do not force its three
channels into D1's four-frequency tensor or rename120kHz as125kHz. Keep the shared experiment
logic, but do not compare unrelated habitats as matched outcomes or call this cross-domain
transfer without an independent experiment.

## Continuation source/access register — 26 September2026

The five original D1 source files and extracted XML/DPL remain preserved. The owner explicitly
authorized additional environmental research in the continuation. None of these ancillary
sources changes the acoustic deployment, chronological split or benchmark protocol.

| Source | Local evidence | Access and disposition |
|---|---|---|
| Arctic Data Center10.18739/A21J9790B daily profiles | `evidence/calibration/environment_inventory.json` | Existing bytes reused; TEOS units converted and verified; applicability unapproved |
| PANGAEA940271/940282/940291/940296 recovered SIT buoys | `evidence/continuation/sit_inventory.json` | Anonymous publisher TSV; CC-BY-4.0;488127156bytes; preserved under `data/raw/environment/mosaic_sit/` |
| ASL55170 factory certificate, O2A4308/resource980 | `evidence/continuation/factory_certificate.json` | Public207370-byte PDF; actual serial/coefficients independently verified; registry attachment association is swapped; license for redistribution not established |
| Combined Iridium/XEOS applicability audit | `evidence/continuation/trajectory_matches.json` | Metadata-only; no test acoustic outcomes;12/24 candidate-test composite profile days meet illustrative5km/24h screen |

The source manual is retained locally. It contains operational credential-like strings and
must not be copied into release payloads or public evidence. No credentials were used.
The certificate may be retained for local verification; redistribution requires a documented
license decision. No new OOI downloads, cloud resources, contacts or dataset substitutions occurred.


OOI fallback follow-up (27 September): anonymous metadata access still worked for the fixed
2017 example week. The independent reviewer used the existing HDF5 reader with an8MiB cap;
164478 bytes across three range requests exposed metadata/coordinates only, not Sv values.
The full product is contiguous uncompressed float32. Exact extraction of all three channels
at10–100m for the inspected day is484075560 bytes; extrapolating134 such days is60.4113GiB,
above the40GiB total raw-data cap. Seven daily HEAD sizes are similar. This is a checked-week
resource estimate, not proof about every possible future date range. The extended D2 range,
calibration lineage and separate protocol remain unfrozen. The small averaged product uses
median Sv and gap interpolation, so it cannot supply the unchanged linear-mean/support contract.
No fallback switch or bulk download occurred. See `evidence/continuation/ooi_fallback_disposition.json`
for exact entity, code-version and use-policy references.


The final bounded OOI metadata check located deployment4 (29 July2017 to17 July2018),
refdes CE04OSPS-PC01B-05-ZPLSCB102, asset ATOSU-63259-00002, sensor depth193m and
water depth574m in the official asset-management repository. No matching external
calibration entry exists in the inspected complete tree; the available00003 entry
is for a different asset and must not be substituted. A1,496-byte CON0-only read of
the existing21August raw file, through pinned Echopype, recovered embedded
38/120/200kHz acquisition coefficients. It did not establish asset/certificate
mapping, environmental provenance, or generation lineage for the20August FullNC.
No sample datagram or new acoustic payload was read. Discovery stopped after this
bounded check. D2 remains resource-plan and calibration-provenance blocked within
the current protocol, without asserting that every conceivable representation is
impossible. A single-channel representation would require its own material
input/protocol amendment and does not solve the unresolved lineage.
