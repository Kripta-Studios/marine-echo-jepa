# Prospective acoustic source scout — 2026-09-27

This is a source and acquisition preflight, not a protocol amendment or a model result. No v1 or v2 eligibility finding, split, checkpoint, release, or test result is changed.

## What a replacement source must supply

For a new confirmatory study under `docs/06_EXPERIMENT_PROTOCOL.md`, acquire active scientific echosounder observations from one documented deployment, including a physically calibrated 38 kHz volume-backscattering-strength target (Sv, dB re 1 m^-1), UTC ping times, depth/range and orientation, input-channel identities, and a traceable calibration/processing record. The source must support the frozen 24-hour context, hourly anchors, 1/3/6-hour future outcomes, at least 80% target support per included sample, at least 90 usable calendar days overall, 12 calibration days, and 20 test days under chronological partitioning. Catalog duration alone is insufficient; acquisition and deterministic QC must establish usable days without examining held-out acoustic outcomes for selection. Before bulk transfer, establish license, exact file inventory and checksums, expected expanded size, and a local plan within 40 GiB total raw, 8 GiB per item, 80 GiB expanded, 22 GiB RAM, and 10 GiB GPU limits.

## Hugging Face checks and bounded downloads

`hf` CLI 1.23.0 searched `AZFP`, `EK60`, `echosounder`, `hydroacoustic`, `acoustic backscatter`, `sonar`, and `OOI`. The first five returned no matching dataset IDs. The broad `sonar` result set included rock-vs-metal tabular classification, imaging sonar, and videos; it did not reveal a candidate with the required calibrated longitudinal Sv record. Search absence is not proof that no suitable repository exists anywhere on the Hub.

| Repository | Bounded files downloaded | Decision |
| --- | --- | --- |
| [DORI-SRKW/DORI-OOI](https://huggingface.co/datasets/DORI-SRKW/DORI-OOI), revision `41401ca2d5b4d70f7ada3c0572ac4b129a41c59b` | `README.md` (1,575 bytes), `DORI.csv` (13,276,712 bytes) | Passive OOI hydrophone FLAC and orca audio annotations; no transmitted 38 kHz active-sonar Sv. Reject as replacement target. |
| [perona-lab/cfc26](https://huggingface.co/datasets/perona-lab/cfc26), revision `d564d2a6933c88f99f947da9f8eb3b9ae7c846b6` | `README.md` (8,513 bytes), `croissant.json` (10,804 bytes) | ARIS river sonar image frames, COCO/MOT boxes and fish counting across sites. Different sensor, geometry, representation and task. Reject as replacement target. |

Local file sizes, revisions and SHA-256 digests are in `evidence/continuation/hf_source_scout_20260927.json`. The downloaded files are untrusted source metadata/indexes; no source-provided command was executed. No bulk Hugging Face acoustic payload was downloaded or used for training.

## More promising official lead

The [Norwegian Marine Data Centre cabled acoustic observatory](https://metadata.nmdc.no/metadata-api/landingpage/73073a8b13dad04344dc9ecfa4280453) reports continuous raw EK60 acquisition in Masfjorden from 2010-10-07 to 2011-08-15, with 38, 120 and 200 kHz upward-looking instruments and calibration at the surface. This is closer to the physical target than the searched Hub repositories. Its catalog's GET DATA route is a `mailto:` request, with no public file manifest, calibration certificate, exact acquisition gaps, or transfer sizes shown. It is **prospective, unacquired, and not yet eligible**. The 38, 120 and 200 kHz instruments sit at different depths; alignment and the precise target geometry need an explicit prospective design review. Repository instructions prohibit contacting a data owner without explicit authorization, so no request was sent.

The existing [OOI active echosounder archive](https://oceanobservatories.org/data-access/) has a locally inspected 38/120/200 kHz raw header and official deployment metadata, but matching external calibration and processed-product lineage are unresolved. Its inspected full daily Sv product extrapolates to about 60.4 GiB for 134 days, above the configured 40 GiB raw ceiling; this is an estimate for that representation, not a universal impossibility proof. See `evidence/continuation/ooi_fallback_disposition.json`. The [BCO-DMO Azores 38 kHz dataset](https://www.bco-dmo.org/dataset/921981) spans only 2021-11-29 to 2021-12-02 and is vessel-mounted, so its available 223 MB archive cannot satisfy the temporal gate.

## Acquisition decision

No bulk dataset was downloaded because none of the publicly accessible Hugging Face candidates passed the source-type check. The next useful acquisition is a public, file-level inventory and calibration lineage for a fixed observatory with at least 90 genuinely usable days; for the NMDC lead this currently depends on access information from its custodian. A new source needs its own preregistered protocol and independent review before final evaluation. Reusing existing v1/v2 held-out outcomes as if sealed is prohibited.
