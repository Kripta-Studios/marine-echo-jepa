# Independent review of the twenty-method development comparison

The distinct reviewer completed the expanded numerical reconstruction. The historical V2 report and its bindings remain unchanged; this addendum records the later verification.

All20 methods share7,593 unique issued rows, identical target masks, observed values, source dates and native0–225m queries. Independent JavaScript arithmetic reconstructed20 primary scores and18,960 daily losses. Maximum discrepancies were6.66e-16dB for primary scores,8.89e-16dB for daily losses, and4.17e-16dB for the19 paired interval endpoints. The2,000-draw resampling sequence matched exactly. No implementation metric module, Torch inference or fitting was invoked by this review.

The paired differences below use the same predeclared48-nominal-hour blocks. Negative differences favour the first endpoint. These intervals are conditional on the single observed development deployment.

| Comparison | Difference (dB) | Paired95% interval (dB) |
|---|---:|---|
| Shared frozen minus random frozen | -0.027153 | [-0.062203,0.001335] |
| Shared frozen minus permuted frozen | -0.068975 | [-0.092589,-0.049190] |
| Shared frozen minus masked frozen | -0.001127 | [-0.032181,0.029797] |
| Shared frozen minus supervised frozen | +0.009597 | [-0.017782,0.034273] |
| Shared full minus scratch supervised | +0.007409 | [-0.014873,0.030693] |
| Shared full minus masked full | +0.080788 | [0.053367,0.108150] |

The shared frozen representation's3.34% point improvement over random features is uncertain under this paired interval. Its advantage over permuted pairing is supported on this development deployment, but superiority over random, masked or supervised features is not established. Masked full fine-tuning outperforms shared full fine-tuning here. CF remains the strongest local neural endpoint, while its different backbone prevents objective-specific conclusions from the existing shared controls. Neither CF endpoint beats LightGBM on development.

The earlier eight-method independent review used Python/NumPy. This expanded review used persistent JavaScript with an independent ZIP/NPY decoder, metric arithmetic and PCG64 resampling. The different routes are preserved accurately.

Root observed the reconstruction CLI exit0 through tool session55183. Its empty console log alone is not independent evidence of process exit. The later `development-comparison-execution-receipt-v2.json` identifies that coordinator observation; the distinct review independently verifies the resulting numbers and files.

Evidence: `development-comparison-v2-numeric-review-final.json`, `development-comparison-result-v2.json`, and `development-comparison-execution-receipt-v2.json`. All75 numerical-review bindings were verified by root after completion. Final-site results, three-seed strong comparisons, supported adaptation, robustness and model-package review remain incomplete. No SOTA, successful transfer or biological-effect claim follows from these development findings.
