# T02/T05 local source and ingestion increment

Status: T02 real access and parsing **partially complete; physical calibration blocked**. T05 transactional local ZIP extraction **implemented and executed**; canonical calibrated shards and D02/D03/D05 remain pending. This increment is not a completed P0 experiment.

Dependencies: T01 runtime preflight was supplied by the leader. Session `/root/core`, configured native role `sol_builder` in `.codex/agents/sol_builder.toml` (`gpt-6-sol`, effort `high`); no independent provider-side binding attestation is exposed. Branch `impl/core`, isolated worktree `marine-echo-jepa-core`. Commit: recorded in Git after this report is written.

Owned paths: `src/marine_echo/data/`, `tests/security/`, `tests/unit/`, `tests/integration/`, `tools/local_data*.py`, `data/manifests/`, `evidence/data/`, this report. No raw source file was modified or downloaded.

## Source and extraction

The immutable source is the owner's existing `data/raw/pangaea/949811` directory in the main worktree. [Source inventory](../../data/manifests/mosaic_azfp_down_2020/source_inventory.json) records all five file sizes and local SHA-256 hashes, including the 705,497-byte manual. The 4,676,515,350-byte ZIP has local SHA-256 `bc9cf56bd92e7dd1c83cfca6d296585a832e13e27635c04f14ff864bf2831ab9`. These local hashes and ZIP CRC establish local integrity; no publisher SHA-256 or signature was provided.

The ZIP metadata guard accepted 3,744 members and 10,169,592,780 claimed expanded bytes. A separate full `ZipFile.testzip()` scan returned `null` bad member after 339.04 seconds. The bounded extractor used a declared 20 GiB expanded cap and 10 GiB free-space reserve, streamed every member with CRC and SHA-256, and atomically published 3,744 files in 663.51 seconds to the ignored core worktree `data/raw/pangaea/mosaic_azfp_down_2020_extracted`. The original archive is retained. Two XML and two DPL files were preserved; both XML files are 6,012 bytes with identical SHA-256 `8efaf894f10b60a1e80885dd8e38366a72319cf6af64c3b0e914b28f6b6437f7`. The extracted directory includes its per-file integrity manifest. [Extraction summary](../../data/manifests/mosaic_azfp_down_2020/extraction_summary.json) records the final-code reuse check when completed.

Free C: space measured 66,663,546,880 bytes before the work and 49,347,776,512 bytes after extraction and concurrent dependency/frontend activity. This is below the protocol's 100 GiB full-pipeline threshold. The 20 GiB extraction cap and 10 GiB reserve permit bounded extraction only; full expanded intermediate retention is not authorized by the observed disk. Process-tree peak RAM was not measured, so resource compliance beyond the disk checks is not asserted. Paid spend and committed cloud cost were USD 0.

Reproduction (PowerShell; run in this worktree with the pinned main `.venv`):

```powershell
& 'C:\Users\Álvaro Schwiedop\Desktop\KriptaStudios\marine-echo-jepa\.venv\Scripts\python.exe' tools/local_data.py inventory --source 'C:\Users\Álvaro Schwiedop\Desktop\KriptaStudios\marine-echo-jepa\data\raw\pangaea\949811' --output data/manifests/mosaic_azfp_down_2020/source_inventory.json
& 'C:\Users\Álvaro Schwiedop\Desktop\KriptaStudios\marine-echo-jepa\.venv\Scripts\python.exe' tools/local_data.py extract --archive 'C:\Users\Álvaro Schwiedop\Desktop\KriptaStudios\marine-echo-jepa\data\raw\pangaea\949811\azfp55170-bioacoustics.zip' --destination data/raw/pangaea/mosaic_azfp_down_2020_extracted --output data/manifests/mosaic_azfp_down_2020/extraction_summary.json --max-expanded-gib 20 --reserve-gib 10
```

## Real AZFP parser and calibration gate

Pinned Echopype 0.11.1 parsed `20021614.01A` with `20021600.XML` as a selected diagnostic file on the catalog's partial first day: 185 pings from 14:13:54 to 14:59:54 UTC, 999 sample indices and channels `[38000,125000,200000,455000]` Hz. It is not a full train calendar day. The first complete train day is 2020-02-17; Echopype parsed both 00-hour chunks separately: `.01A` has 4 pings from 00:00:11 to 00:00:56 and `.01B` has 235 pings from 00:01:29 to 00:59:59. This validates frequency normalization despite misleading XML `units="hz"` on values 38/125/200/455. The [selected train A/B probes](../../evidence/data/selected_train_hour_a_probe.json) and [selected B probe](../../evidence/data/selected_train_hour_b_probe.json) record actual shapes and times.

The XML and DPL identify AZFP serial 55170 and configured sound speed 1465 m/s. The manual equipment table instead identifies a different downward unit serial 55169 (and upward 55171). An ADR is required before using the manual as instrument-specific calibration provenance; this report treats it as platform context only. The manual is retained only in the ignored immutable raw source and was not copied into Git or logs because it contains legacy operational details.

Echopype's `Environment.sound_speed_indicative` (m/s) and `absorption_indicative` (dB/m) are entirely NaN in the actual parsed chunks. Its `Environment.temperature` is finite near −1.86 to −1.77 °C and is derived from AZFP ancillary thermistor counts. Although Echopype labels it `sea_water_temperature`, source documentation describes internal/sonar temperature; it has no verified seawater meaning here. The XML reports no installed pressure sensor. In Echopype 0.11.1, `get_env_params_AZFP` explicitly requires user salinity and pressure and uses temperature to derive sound speed and absorption unless independently supplied values are given. `compute_Sv(ed)` without supplied water inputs failed with `TypeError: unsupported operand type(s) for -: 'NoneType' and 'float'`. No environmental values were guessed. Calibrated Sv, range in meters, 38 kHz physical backscatter index, and full protocol benchmark remain blocked pending authoritative compatible water parameters and sensitivity review.

Exact selected real probe command:

```powershell
& 'C:\Users\Álvaro Schwiedop\Desktop\KriptaStudios\marine-echo-jepa\.venv\Scripts\python.exe' tools/local_data_probe.py --raw data/raw/pangaea/mosaic_azfp_down_2020_extracted/20021700.01B --xml data/raw/pangaea/mosaic_azfp_down_2020_extracted/20021600.XML --output evidence/data/selected_train_hour_b_probe.json
```

## Bounded diagnostic replay and remaining data work

The [48-hour train archive metadata](../../evidence/data/train_48h_archive_metadata.json) covers only filename inventory: 85 files across 46 of 48 hour stems, 39 with dual chunks, and two absent stems (`20021822/23`). It does not prove decoded ping coverage. Full canonical preprocessing must audit every suffix by decoded UTC ping time, mode and overlap; it must not assume a filename-hour is an independent sample.

The [real one-day diagnostic replay](../../evidence/data/diagnostic_raw_replay_2020-02-17.json) is 629,805 bytes, SHA-256 `68240f4e6d75eef0c6890dbbb658468b45ff0f4ea95eb2ec1e41084de200beea`, with 96 observed trailing 15-minute bins from 24 `.01B` chunks and 5,564 raw pings. Each row has UTC bin-end `event_time_utc` and 4×64 mean raw digitizer counts across sample-index bands. It is explicitly `raw_counts`, `calibrated=false`, `range_convention=sample_index`, `data_kind=public_real`; no distance or biological quantity is inferred. `.01A` chunks remain in the raw inventory and are excluded only from this bounded same-mode diagnostic, not silently from a future canonical dataset.

```powershell
& 'C:\Users\Álvaro Schwiedop\Desktop\KriptaStudios\marine-echo-jepa\.venv\Scripts\python.exe' tools/local_data_replay.py --extracted data/raw/pangaea/mosaic_azfp_down_2020_extracted --day 2020-02-17 --chunk-suffix 01B --output evidence/data/diagnostic_raw_replay_2020-02-17.json
```

## TDD, validation and review

[RED/GREEN log](../../evidence/data/tdd_local_ingest.txt) records actual failed and passing commands and exit codes. The final focused suite used both real AZFP and real ZIP environment variables and exited 0 with 17 passed, 2 upstream deprecation warnings, no skips (20.39 seconds). Ruff check exited 0; mypy over `src/marine_echo/data` exited 0. Security tests cover traversal, Windows device names, duplicate names, symlinks, CRC failure, expansion limit, atomic publish, concurrent source mutation, manifest path tampering, changed extracted contents and idempotent reuse. Real integration asserts 38,000/125,000/200,000/455,000 Hz, duplicate `.01A/.01B` chunks and absent hour stems.

Independent R0 reviewer `/root/review` inspected source, ZIP metadata, extraction manifest and calibration evidence and raised the material findings addressed above. Review approval remains pending; a separate reviewer report and the serial-provenance ADR are leader-owned. Next engineering work is typed contracts/canonical count handling and compact model smoke tests. Required physical-unit D06 and full T05 canonical shards remain blocked by verified environmental provenance and available disk plan; no model outcome is claimed.
