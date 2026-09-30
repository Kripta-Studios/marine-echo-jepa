# Real native-product CF-JEPA screen

The reviewed CF-JEPA adaptation completed an actual local CUDA trajectory with
exit0:6000 self-supervised updates and four fresh500-update frozen selection
probes. Development selection chose the1500-update encoder at0.745726589903934dB
daily pinball. This is a completed first screen, not a final transfer result or
a SOTA claim. Final AEON2 numerical data remains unopened.

The implementation follows the pinned [author CF-JEPA repository](https://github.com/WDSLab/CF-JEPA)
at `5d3d2fd1273c283fbfa03249c078619245e84033`: multiscale dilated temporal
convolutions, crop invariance, three forecast zones, normalized latent prediction,
variance/covariance terms, horizon annealing and a parameter-only cosine EMA.
The marine future-zone construction and four native acoustic measurement channels
are declared adaptations; this is not a complete reproduction of the paper's
original benchmarks. No flattened MLP encoder or moved prediction regularizer
is presented as a new model. The inference EMA encoder has412800 parameters;
pretraining optimizes462336 parameters, the readout has18565 and the complete
model has893701, including the EMA copy.

The separate native contract admits18593 TRAIN windows, of which18312 support
the SSL objective, and7593 issued development windows. All fitted data belongs
to reserved TRAIN deployments. Whole AEON2 deployments/site remain outside every
local fitted ancestor. Input geometry remains the publisher's actual200/220m
TRAIN products and225m development product, without dB depth scaling. A later
230m assessment product will remain230m. These integrated acoustic statistics
are neither depth profiles nor waveform measurements. Clock/orientation and
processing uncertainty remain disclosed.

| SSL checkpoint | Fresh probe250 pinball,dB | Fresh probe500 pinball,dB |
| --- | ---: | ---: |
| 1500 | 0.891647509 | 0.745726590 |
| 3000 | 0.918304363 | 0.795987559 |
| 4500 | 0.938176797 | 0.829410345 |
| 6000 | 0.944646614 | 0.840351894 |

All four candidates used the same TRAIN-only probe initialization, sample policy,
label budget and development opportunities. Selection did not assume that more
updates improve transfer, and no unchanged30000-update extension was used.
Source CF learning rate0.00034 and weight decay0.05 were preserved; the frozen
probes followed the separately fixed warmup/cosine policy in ADR0016. Strong
2000-update frozen readouts and3000-update fine-tuning/supervised endpoints are
separate trajectories under ADR0018. The short probes do not substitute for them.

The successful attempt owned the device for4727.313seconds; the wrapper charged
4743.094seconds of complete owned time. Peak allocated memory was1075154432bytes,
reserved1235222528bytes and conservative summed process-tree peak RSS5275938816bytes,
all below the10GiB/22GiB limits. The earlier deterministic-pooling failure,
its resource charge and the mathematical pooling-parity repair remain preserved.
No warning-only determinism, denied filesystem workaround or paid resource was used.

Actual CPU verification on64 real DEV prefixes reproduced128-dimensional
encoder features bit-identically through the query-free encoder API. Forecasts
from context and issued native queries replayed within the frozen CPU/CUDA
tolerances; maximum absolute difference was0.0001068115234375dB. No target or
future acoustic array was an inference input and no optimizer ran in this check.

Durable artifacts are in `outputs/native_acoustic_ssl_v1/cf_jepa_seed7_h96_deterministic/`:

- `selected_encoder.pt`, SHA256 `2f34c61a4e0d7937a56b97ec1a51ea5f50d45802ec0eff31057fa243e55d386a`.
- `inference.pt`, SHA256 `d6eef2c06807843848c029a0655ddb8b2694d9f1802141daace0c8c41a1812cc`.
- `run.json`, SHA256 `0c0865b9c8d605a6a5e8e57870d9e32168a3ac22152acd200a06fc88550fab62`.
- Safe resumable checkpoints, TRAIN membership, raw predictions and development metrics.

Proofs include `prefit-review-final-03.json`, `prefit-bindings-verification-03.json`,
the closed `cf-jepa-reviewed-retry.log`, `cf-context-only-cpu-replay.json` and the
machine-readable programme ledger. A genuinely distinct reviewer approved the
prefit source/data/selection/resource bindings. Separate strong CF downstream
prefit and independent result reconstruction remain required.

Matched direct, masked SSL, random and pairing-permuted controls, strong readouts,
LightGBM, local Chronos2, finalist seed variability, held-out assessment, robustness,
relocation and the final scientific model report are still outstanding. Chronos2's
external pretraining exposure is not comprehensively excluded and cannot inherit
the local ancestry guarantee. Acoustic prediction establishes no causal fish,
biomass, species, catch, fuel-saving or production-integration result. App/release
work and the approved r3 archive remain frozen.
