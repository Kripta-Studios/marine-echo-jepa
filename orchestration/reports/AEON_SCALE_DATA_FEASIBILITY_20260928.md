# AEON scaling data feasibility, 28 September 2026

This is a metadata-only inventory for a new post-hoc development study. It does
not change the frozen AEON3 2024–25 experiment, its retrospective result, the
prior-year transfer outcome, or the historical v1/v2 registries. The publisher
source is [AEON AZFP Integrated Sv products, version 2](https://figshare.com/articles/dataset/AZFP/29247113)
(CC BY 4.0). The exact machine-readable inventory is
`evidence/aeon_scale/metadata_scout_20260928.json`, SHA-256
`b6fd10e84019accc61db261675e6823ecb811e398ab34ac30674acce5a486195`.

Four additional official ZIP files (Figshare IDs 61937263, 61937266, 61937272,
61937278) were downloaded to `E:\marine-echo-jepa-scale\data`. Their combined
transfer was 81,879,095 bytes; publisher sizes and MD5 digests were verified.
The metadata-only scan found 25,409 new 38 kHz `60minFullDepth` rows, but none
has the frozen 0–200 m geometry. The AEON4 archives instead report 0–220 or
0–225 m; the AEON2 archive reports 0–230 m. Two AEON4 products also lack
channels in the original four-channel model input, and some use different
complete-hour ping counts. No acoustic `Sv_mean` values were interpreted or
retained in this scan. A distinct reviewer independently verified all four
archive MD5/SHA-256 values and spot-checked one 38 kHz member per archive,
its header, permitted metadata fields and geometry. This is a bounded
spot-check, not an independent reconstruction of every row.

The original TRAIN fit has 4,965 windows, so literal 3× and 5× require 14,895
and 24,825 windows respectively. Reclassifying the already scored AEON3
2023–24 prior-year transfer cohort as new TRAIN data yields at most 13,472
candidate windows (2.71×) before a new split and QC. It would cease to be an
external test for that new model. Adding the AEON3 2021 archive under a revised
180-ping and missing-channel contract yields at most 16,775 candidates (3.38×)
before further QC; it is not an unchanged-target augmentation. The four new
archives contribute zero unchanged 0–200 m candidates, so a 5× training
corpus is not supported by the currently published processed products under
the original observation contract.

`E:` had 70,550,622,208 free bytes at preflight. Storage easily accommodates
these processed ZIPs and bounded checkpoints. A multi-site expansion requires
a new independently reviewed target/representation and deployment-grouped
evaluation before any numeric acoustic access; a source-native FullDepth value
cannot be renamed a common 0–200 m measurement.
