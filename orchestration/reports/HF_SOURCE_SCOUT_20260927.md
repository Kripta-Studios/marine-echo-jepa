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

## Chinese research datasets checked

The [UATD dataset](https://figshare.com/articles/dataset/UATD_Dataset/21331143), associated with Peng Cheng Laboratory researchers, is a genuine 4.47 GB sonar dataset, but its 9,200 forward-looking sonar images and object boxes target detection at 720/1,200 kHz, not long-duration calibrated 38 kHz Sv forecasting. The [Chinese Academy of Sciences QiandaoEar22 study](https://arxiv.org/abs/2406.04353) is passive ship-radiated hydrophone audio, about 31 hours of recordings. Neither should be downloaded to fill this study's temporal/Sv requirement.

## Better open lead discovered after the Hub check

The [AEON AZFP Integrated Sv products](https://figshare.com/articles/dataset/AZFP/29247113) at Figshare are CC BY 4.0 and published as calibrated active 38 kHz/other-frequency products from fixed Gulf of Maine observatories. The source readme says the manufacturer coefficients are applied in Echoview, with additional absorption and TVG corrections, background/noise filtering and manual exclusions. It also documents a roughly 15-degree upward transducer tilt. The supplied data are *processed hourly and daily CSV exports*, not the raw 15-minute bins in the current frozen forecasting contract. Exact raw-to-product lineage, calibration config/`.ecs` files, exclusion masks, timezone, and valid-support semantics need verification before study admission.

I downloaded the 23,400-byte source readme and the 11,313,838-byte `AEON3 GEB Jul2021-Jan2022` Sv archive. Both publisher MD5 values and local SHA-256 hashes match the manifest. The ZIP has 70 CSV files and expands to 112,161,538 bytes. Its 38 kHz `60minPartition` filenames span seven months. A metadata-only pass found 174 distinct `Date_M` fields from 2021-07-23 through 2022-01-12, of which 172 have 24 distinct hourly `Interval` IDs, one has 23 and one has 14. This is **hourly record coverage, not 172 QC-usable days**. It does not establish 80% valid target support, a valid 15-minute input history, chronological calibration/test eligibility, or a sealed holdout. This specific ZIP contains 38, 200 and 455 kHz exports; the 125 kHz channel described in the general readme is absent from it. A prospective protocol amendment and independent review are required before any final evaluation based on these hourly products.

## Other official lead

The [Norwegian Marine Data Centre cabled acoustic observatory](https://metadata.nmdc.no/metadata-api/landingpage/73073a8b13dad04344dc9ecfa4280453) reports continuous raw EK60 acquisition in Masfjorden from 2010-10-07 to 2011-08-15, with 38, 120 and 200 kHz upward-looking instruments and calibration at the surface. This is closer to the physical target than the searched Hub repositories. Its catalog's GET DATA route is a `mailto:` request, with no public file manifest, calibration certificate, exact acquisition gaps, or transfer sizes shown. It is **prospective, unacquired, and not yet eligible**. The 38, 120 and 200 kHz instruments sit at different depths; alignment and the precise target geometry need an explicit prospective design review. Repository instructions prohibit contacting a data owner without explicit authorization, so no request was sent.

The existing [OOI active echosounder archive](https://oceanobservatories.org/data-access/) has a locally inspected 38/120/200 kHz raw header and official deployment metadata, but matching external calibration and processed-product lineage are unresolved. Its inspected full daily Sv product extrapolates to about 60.4 GiB for 134 days, above the configured 40 GiB raw ceiling; this is an estimate for that representation, not a universal impossibility proof. See `evidence/continuation/ooi_fallback_disposition.json`. The [BCO-DMO Azores 38 kHz dataset](https://www.bco-dmo.org/dataset/921981) spans only 2021-11-29 to 2021-12-02 and is vessel-mounted, so its available 223 MB archive cannot satisfy the temporal gate.

## Acquisition decision

No bulk Hugging Face dataset was downloaded because none of the inspected Hub candidates passed the source-type check. The bounded AEON archive is the next candidate for calibration-lineage, timestamp, support and representation audit; it is not yet an eligible replacement under the existing 15-minute protocol. The NMDC lead depends on custodian access information. A new source needs its own preregistered protocol and independent review before final evaluation. Reusing existing v1/v2 held-out outcomes as if sealed is prohibited.
