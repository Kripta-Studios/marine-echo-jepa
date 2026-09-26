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
