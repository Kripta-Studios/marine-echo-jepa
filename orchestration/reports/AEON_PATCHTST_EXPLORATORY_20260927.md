# AEON masked patch transformer exploratory development outcome

This is a post-hoc TRAIN/validation architecture check, outside the frozen ADR 0011
selection and outside final CAL/retrospective TEST evaluation. The design uses the
channel-independent patching principle from Nie et al., ICLR 2023,
https://openreview.net/forum?id=Jbdc0vTOcol, adapted to six four-hour patches over
24 past hourly source products, four masked frequency channels, three Transformer
encoder layers (width 128, four heads), and a 38-kHz persistence skip. The head
issues sorted 0.05/0.25/0.5/0.75/0.95 quantiles for +1/+3/+6 source intervals.

The config was fixed before fitting at `configs/aeon_patchtst_exploratory.json`,
SHA-256 `995de7560ffd30fed24aad9654d77fea123797f9f3431153b1454d028225842f`.
The source archive SHA-256 is
`4e72dd4dbec707b6bf15168e51f380cbe9145a78b595ef886d78cc3806c0ecde`;
the frozen cohort SHA-256 is
`5e475f6af798025f187c70ae638710d2b4362f688db2ad43d60ec436dda3e25d`.
Normalization used TRAIN observations and TRAIN targets only. No CAL or TEST numeric
outcomes were opened by this run.

The first launch exited 1 without console output during early fit, after creating
only an empty `seed7` directory. It is `NOT_RUN_NO_OUTPUT`; no score or checkpoint
is attributed to it. The logged retry used a new output path and completed exit 0.
Its exact command was:

```powershell
$env:PYTHONPATH='src'
$env:OMP_NUM_THREADS='4'
$env:PYTHONFAULTHANDLER='1'
& '<main>/.venv/Scripts/python.exe' -u -m marine_echo.training.aeon_patchtst --archive '<main>/data/prospective/aeon-azfp-20260927/AEON3_GEB_Mar2024-Mar2025_AZFP_Sv.zip' --split-review orchestration/reviews/AEON_SPLIT_AMENDMENT_20260927.json --config configs/aeon_patchtst_exploratory.json --output outputs/aeon_patchtst_exploratory_20260927_r2
```

Seed 7 completed 3,000 supervised updates on 4,965 TRAIN rows. The final checkpoint
was the only validation endpoint. It issued 1,219 validation rows; each horizon had
50 eligible source dates. The protocol daily mean pinball loss was
**0.9299893352789196 dB**, with horizon losses 0.8216756248032775,
0.9513441587439235 and 1.0169482222895578 dB. The independently reviewed
three-seed direct ensemble reference is 0.6390617418 dB. This exploratory seed
was 0.2909275935 dB worse; it does not establish a general architecture ranking.

The fixed continuation threshold was 1.05 times that direct reference,
0.67101482889 dB. Seed 7 failed it, so seeds 13 and 23 are `NOT_RUN_BY_FIXED_GATE`.
An independent local re-score of the saved prediction NPZ reproduced
0.9299893352789196 dB. Row IDs, target masks/truth, target source times and
interval IDs matched the core direct seed-7 validation file exactly.

Artifacts reside at `outputs/aeon_patchtst_exploratory_20260927_r2/` in the
isolated worktree. `report.json` SHA-256 is
`0c40bc70e2adeaa0bb54a1f5ef8777505232d3fc694ad04b69d4f8e31edb29a6`;
the seed-7 checkpoint SHA-256 is
`e8728ae34be00ca18d2cc3571f470ca2092e77031e594f93d2eb80ccd9827d50`;
prediction NPZ SHA-256 is
`12ade00f0de191e61179cc59e55493f6104e46fb572839a78597dc171b80a9e7`.
Fit time was 296.82 seconds, measured peak process RSS 1,922,793,472 bytes and
peak CUDA reserved 113,246,208 bytes. The GPU owner was returned to `NONE`.

This result is pending distinct outcome review. It does not change the selected
operational family, JEPA comparison, app release, or any final-evaluation claim.
