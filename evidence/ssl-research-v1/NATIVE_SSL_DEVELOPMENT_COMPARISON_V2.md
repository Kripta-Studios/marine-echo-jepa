# Native acoustic development comparison: twenty executed endpoints

Status: actual reconstruction exited 0 under distinct 150-binding admission review. Independent numerical reconstruction of this expanded matrix is NOT_RUN. The earlier eight-method reconstruction was independently verified.

All results concern one development deployment, AEON4_JOB:61937266, at its native 0–225 m integrated 38-kHz product. Whole-site AEON2 final-test numerical values remain unopened. These are development comparisons; they do not establish transfer, JEPA value, SOTA or causal biological effects.

The saved outputs preserve the same 7,593 issued rows, targets, masks, source dates and query geometry. There are 7,581/7,557/7,521 observed targets at horizons 1/3/6; floor18 yields 316 eligible dates per horizon and 7,572/7,548/7,512 scored rows. No prediction-based filtering, support intersection, imputation, geometry renaming or test threshold tuning was performed. All twenty forecasts vary across issuance.

| Endpoint | Development pinball (dB) | Difference vs LightGBM | Paired 95% interval (dB) |
|---|---:|---:|---|
| lightgbm | 0.505274 | 0 | reference |
| chronos2_zero_shot | 0.506916 | +0.001642 | [-0.008968, 0.011138] |
| cf_jepa_full_finetune | 0.541814 | +0.036540 | [0.028145, 0.044476] |
| cf_jepa_frozen_readout | 0.614422 | +0.109149 | [0.094598, 0.126816] |
| masked_ssl_full_finetune | 0.691806 | +0.186533 | [0.167036, 0.207284] |
| cf_short_probe | 0.745727 | +0.240453 | [0.208555, 0.279104] |
| direct_end_to_end | 0.765185 | +0.259912 | [0.231313, 0.292271] |
| shared_full_finetune | 0.772594 | +0.267321 | [0.238717, 0.299209] |
| direct_frozen_readout | 0.776118 | +0.270845 | [0.238601, 0.307933] |
| shared_frozen_readout | 0.785715 | +0.280442 | [0.256526, 0.304479] |
| masked_ssl_frozen_readout | 0.786842 | +0.281568 | [0.261033, 0.302690] |
| direct_short_probe | 0.789279 | +0.284005 | [0.250540, 0.322941] |
| masked_ssl_short_probe | 0.806670 | +0.301397 | [0.271019, 0.333079] |
| random_frozen_frozen_readout | 0.812868 | +0.307594 | [0.281101, 0.338265] |
| shared_short_probe | 0.824647 | +0.319373 | [0.297393, 0.342231] |
| permuted_ssl_frozen_readout | 0.854690 | +0.349417 | [0.325780, 0.373796] |
| random_frozen_short_probe | 0.882659 | +0.377385 | [0.345048, 0.414748] |
| permuted_ssl_short_probe | 0.895474 | +0.390201 | [0.362062, 0.419858] |
| persistence | 0.989176 | +0.483902 | [0.462487, 0.504158] |
| seasonal24 | 1.031738 | +0.526465 | [0.499634, 0.554365] |

Lower loss is better; a positive difference favours LightGBM. Intervals use the predeclared paired 2,000-draw bootstrap, seed20260929, 48 nominal source-calendar hours, with gaps and duplicate sampled-block multiplicities retained. They are conditional on this one deployment, not intervals over independent sites or verified UTC observations.

The shared frozen candidate scores0.785715, versus masked0.786842, random0.812868, permuted0.854690 and supervised features0.776118. Its3.34% improvement over random features is a development observation; supervised features perform better and the masked difference is small. Full shared fine-tuning0.772594 does not beat matched scratch supervised0.765185. Masked full fine-tuning0.691806 is stronger here.

CF frozen0.614422 and full fine-tuning0.541814 are the strongest local neural endpoints in this matrix, but both trail LightGBM0.505274. CF uses128-dimensional features and 18,565 head parameters; shared methods use64 and5,189. These controls therefore do not isolate the effect of the CF SSL objective. Chronos2 zero-shot0.506916 is close to LightGBM; its external pretraining ancestry is unknown and it does not inherit the local fitted-ancestor exclusion guarantee.

Short probes use500 supervised updates and are development checkpoint-selection instruments. Strong frozen endpoints use fresh2,000-update readouts; full and scratch endpoints use3,000 updates. Four scheduled selection opportunities, matched supervision/sample streams where specified, warmup and cosine schedules are retained. Equal updates are not equal compute. SSL pretraining/probes, full downstream costs, startup/evaluation/saving/failures and serial ownership are accounted separately in the immutable run/resource receipts and active96-hour ledger. Three-seed replication is still executing; this matrix contains seed7 only.

The structural band-pooling proof uses synthetic inputs and does not establish the cause of these numerical gaps. Its proposed revision requires the pending ADR0020 owner budget clarification and source-prefit approval. No revised fit has occurred.

Remaining evidence: independent expanded numerical reconstruction, three-seed strong endpoints/direct reference, frozen pretest selection, held-out native230m comparisons, secondary-channel robustness, supported prefix adaptation, relocated model-only inference package and independent scientific/package review. Missing runs are not positive results. Software, experiment, JEPA-value and business gates remain separate.

Result SHA256: `7a22604fb183c84c93ec4479ea57e703cdacf4fececa9c3e346170515c4384b7`.

Sources: `orchestration/native_development_comparison_v2.json`, `evidence/ssl-research-v1/development-comparison-v2-review-final.json`, `evidence/ssl-research-v1/development-comparison-result-v2.json`, `evidence/ssl-research-v1/development-comparison-execution-v2.log`.
