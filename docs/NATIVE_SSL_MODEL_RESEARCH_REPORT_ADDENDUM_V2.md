# Native acoustic SSL: verified development replications

This addendum preserves the original V1 report. It records completed development comparisons, not final-site transfer results. The reserved AEON2 acoustic values remain unopened. App and release work remain frozen.

The distinct reviewer independently decoded all 28 saved prediction files and reconstructed 28 point scores, 26,544 daily losses and 27 paired bootstrap intervals. Maximum discrepancies were 6.7e-16 dB for point scores, 1.4e-15 dB for daily losses and 5.6e-16 dB for interval endpoints. The coordinator subsequently verified all 228 review bindings. No corpus arrays, fitting or reserved-site values were used in this reconstruction.

All methods share 7,593 issued contexts from AEON4_JOB:61937266 at native 0–225 m. The observed target counts are 7,581, 7,557 and 7,521 for the 1-, 3- and 6-interval queries. Each horizon contributes 316 eligible source-calendar dates. Forecasts are finite and variable, query geometry and target masks agree, and no method-specific row intersection was used. Dates and nominal interval arithmetic do not establish UTC or exact elapsed hours.

Lower daily-balanced quantile pinball is better. The following means and sample standard deviations include every fixed seed, 7, 13 and 23; the denominator for sample variance is n−1.

| Fixed endpoint | Development mean (dB) | Sample SD (dB) |
| --- | ---: | ---: |
| CF-JEPA, strong frozen readout | 0.628424 | 0.034819 |
| CF-JEPA, full fine-tuning | 0.558910 | 0.017539 |
| Shared temporal backbone, fresh supervised | 0.757500 | 0.013907 |
| CF-JEPA, short selection probe | 0.795168 | 0.113676 |

The fixed LightGBM comparison scores 0.505274 dB. Every CF full-fine-tuning endpoint has a positive paired difference from LightGBM, with conditional 95% intervals of [0.028145, 0.044476], [0.041998, 0.063760] and [0.060252, 0.084210] dB for seeds 7, 13 and 23 respectively. These are intervals conditional on one development deployment and the reviewed predictions. They are not intervals across independent sites or across seeds. No population or SOTA claim follows.

CF improves on the fresh shared-temporal supervised endpoints in these results, but their backbones, parameter counts and readout dimensions differ. This comparison cannot isolate the value of the JEPA objective. The original matched shared-temporal frozen comparisons also do not establish JEPA representation value: its difference from the random encoder includes zero, and its difference from masking includes zero. Fully fine-tuned forecast performance alone is not evidence of transferable pretrained representations.

The single authorized nonlinear frequency-conditioning revision remains under evaluation. Its completed seed-7 short-probe selection scores are JEPA 0.789188, masking 0.701138, shuffled 0.893408 and random encoder 0.937210 dB. Its matched fresh supervised endpoint scores 0.550881 dB. These are runner scores awaiting reconstruction; matched strong readouts remain required. The lower masking score prevents treating the revised architecture's training completion as a JEPA advantage. All five screens completed with verified owned cleanup, consuming 3.171 full-owned GPU-hours. Seeds 13 and 23 are being implemented in separately versioned modules without changing fitted version-one sources. The owner permits eleven total screening recipes including controls and at most twelve full-owned Band GPU-hours within the aggregate ninety-six-hour programme allowance.

The CF seed-13 native training failure and its single reviewed exact-recipe retry remain in the historical records. The independent numerical review also encountered a model-capacity interruption before writing its final artifact. Its recovery used the same distinct session, retained independent arithmetic outputs and unchanged artifact hashes; it did not decode predictions again. All bindings were reverified. The prefix-contract coordinator check encountered a separate Windows native process failure; one unchanged retry subsequently passed all 237 checks and the physical saved-artifact CPU replay. Independent prefix software review and real fitting admission remain required. No failure was counted as positive evidence or removed from the record.

Next scientific gates are the matched Band endpoints, fixed-seed replication, finalist and control freeze, independent final-access review, native 0–230 m held-out assessment, and deployment-specific 1/7/30-day prefix transfer. Pretrained weights and reusable encoder/latent APIs already exist, with real CPU replay for the original Shared and CF checkpoints. A final report and portable model package require the outstanding held-out and transfer evidence.

Evidence: `evidence/ssl-research-v1/development-comparison-v3-numeric-review-final-v2.json`, `development-comparison-v3-numeric-review-closeout-v2.json`, the preserved capacity-failure witness, `orchestration/native_development_comparison_v3.json`, and the original V1 scientific report. The complete saved development result remains at `evidence/ssl-research-v1/development-comparison-v3.json`.
