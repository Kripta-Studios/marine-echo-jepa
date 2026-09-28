# AEON same-data 30k development outcome

This post-hoc TRAIN/validation study executed the predeclared 30,000-update
seed-7 comparison on the unchanged AEON3 Georges Basin 2024–25 cohort. It is
separate from the historical v1/v2 findings and the original 3,000-update
campaign. A distinct reviewer independently replayed predictions, metrics,
checkpoint identities and the fixed Stage-2 gate. The exact machine review is
`orchestration/reviews/AEON_SCALE_30K_OUTCOME_REVIEW_20260928.json`.

| Family | Original corrected 3k daily pinball | 30k final daily pinball | Change in loss |
| --- | ---: | ---: | ---: |
| Direct neural, seed 7 | 0.651154 dB | 1.178379 dB | 80.97% worse |
| EMA-JEPA, seed 7 | 0.648805 dB | 0.913424 dB | 40.79% worse |

The 30k direct slot completed 30,000 supervised updates. EMA-JEPA completed
15,000 TRAIN-only SSL pretraining updates and 15,000 supervised updates. Both
saved final model checkpoints and finite, nonconstant, monotone validation
predictions with shape 1,219 × 3 horizons × 5 quantiles. The corrected primary
score uses at least 18 observed anchors per source date and horizon; it has 50
eligible dates per horizon, with 1,194, 1,192 and 1,189 eligible rows. The
reviewer reconstructed all 18 scheduled validation predictions and all 24
checkpoint hashes. No calibration or test outcomes were opened.

The diagnostic direct score rose from 0.650889 dB at 2,500 supervised updates
to 1.178379 dB at the frozen 30,000-update endpoint. The EMA-JEPA score rose
from 0.664914 dB at 2,500 supervised updates to 0.913424 dB at its frozen
15,000-supervised-update endpoint. The earlier checks cannot be substituted
after seeing validation. The 1% Stage-2 improvement condition failed for both
families. The executable gate recorded
`STAGE_2_NOT_AUTHORIZED_BY_PREDECLARED_RULE`; no seeds 13 or 23 were run.

EMA-JEPA's final pretraining representation had target effective-rank fraction
0.02867 and predictor/target RMS ratio 16.27, consistent with a severe
low-rank/scale mismatch but not exact zero-variance collapse. The progressive
validation degradation under a fixed 3e-4 learning rate is a plausible
overtraining diagnostic, not a proven mechanism. These data do not justify a
50,000-update extension. This result cannot establish independent
generalization, physical calibration or state of the art.

Direct peak process RSS was 1,887,002,624 bytes and PyTorch GPU reserved memory
25,165,824 bytes; EMA-JEPA peaked at 2,030,170,112 bytes RSS and 27,262,976
bytes reserved. Wall-clock training time was approximately 17.5 minutes for
direct and 25.4 minutes for EMA-JEPA, including pretraining. Additional VRAM
or RAM would not itself address this negative scaling result.
