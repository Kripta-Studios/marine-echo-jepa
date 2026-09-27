# AEON3 2024 processed-Sv provenance follow-up

Date: 2026-09-27. Scope: publisher and institutional metadata only. This review
did not inspect calibration/test acoustic outcomes or change ADR 0008.

The [Figshare collection](https://figshare.com/articles/dataset/AZFP/29247113)
identifies the selected processed archive as file 61937281, 44,727,557 bytes,
publisher MD5 `083769f09573164ea8f43ce273971ec0`, CC BY 4.0. The local
SHA-256 is
`4e72dd4dbec707b6bf15168e51f380cbe9145a78b595ef886d78cc3806c0ecde`.
The [UNH AEON portal](https://eos.unh.edu/aeon/data/aeon-data-portal) lists
AEON3 Georges Basin AZFP raw coverage for 6 March 2024 to 31 March 2025.
The [March/April 2024 cruise report](https://eos.unh.edu/sites/default/files/media/2024-05/rv-connecticut_marapr2024forwebsite.pdf)
records the lander deployment; its UTC deployment time does not establish the
timezone of the exported hourly product timestamps.

The source readme calls the `55144` filename field `SerialNumber`. The processed
ZIP contains monthly CSVs and Echoview project/workspace sidecars, but no AZFP
raw, deployment-specific `.cfg`, `.ecs`, or calibration certificate. The linked
[deployment information](https://ndownloader.figshare.com/files/55904774)
documents the AMAR501 recorder/lander, not an independent 2024 binding of
AZFP serial 55144. The [NOAA water-column-sonar viewer](https://www.ncei.noaa.gov/maps/water-column-sonar/)
is generic; a bounded [NCEI metadata query](https://www.ncei.noaa.gov/metadata/geoportal/opensearch?q=AZFP%26time=2024-03-01/2025-04-01%26bbox=-68.2,42.5,-68.0,42.7%26f=csv)
did not identify this deployment's AZFP raw accession or file manifest. This
absence from one query does not prove raw files are unavailable.

The readme describes coefficient, TVG and absorption processing followed by
noise removal, median filtering, manual exclusions, and surface correction.
It does not bind exact calibration/configuration files or project hashes to
these CSV exports. A local `.evwx` header says Echoview `13.1.120.0`, but
the exact CSV export processing version is unverified. Keep the target label
**source-reported conditioned Sv**. Treat 55144 as the source filename
identifier, source dates as timezone-unknown, and the dataset as one processed
deployment without a verified raw-to-product calibration chain. Do not infer
fish abundance, biomass or independently calibrated absolute backscatter.

The next bounded provenance step, if a direct public accession emerges, is to
compare its deployment serial and configuration/calibration files with the
processed export. No paid acquisition or external contact was performed.
