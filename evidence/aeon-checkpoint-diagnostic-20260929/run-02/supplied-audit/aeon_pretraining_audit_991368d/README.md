# AEON pretraining gradient audit — commit 991368d

## What was actually executed

18 synthetic CPU backward-pass probes (3 seeds × 2 missingness patterns × 3 model modes)
and 3 paired initialization checks ran against the exact source blobs retrieved through
GitHub. The local copies were verified against Git blob IDs before execution.
No AEON acoustic data, trained checkpoints, user environment or GPU was used.
This is a source-behavior experiment, not an AEON forecasting benchmark.

## Observations

- EMA pretraining: all 6,144 first-layer weights attached to the final six positions
  have exactly zero loss gradient in every tested case. Earlier-position gradients
  are nonzero. This follows from zeroing those student inputs and stopping teacher gradients.
- Shared-SIGReg pretraining and full-context direct supervised control: those suffix
  columns have nonzero gradients on these synthetic inputs. This does not establish
  that either method trains good representations on the acoustic data.
- Same seed: initial encoder tensors match, initial forecast heads do not, for all
  three tested seeds. The SSL constructor consumes RNG for its predictor before its head.

The suffix result is a data-gradient finding, not a claim of absolutely frozen weights:
AdamW weight decay can alter them, later shared layers change, and supervised training
can learn them. It does not prove the cause of the observed rank, scale, or loss failures.

## Source identities

Repository: Kripta-Studios/marine-echo-jepa, commit 991368d

- src/marine_echo/models/aeon_ssl.py: Git blob 7bd8a94cde303d968df1e541af47b50ce969d342
- src/marine_echo/models/sigreg.py: Git blob ef6f77225d03c32daa8d6509401e998658f2b198

The source files themselves are not duplicated in this deliverable. The script reads the
existing source files and refuses different identities. The executed-results JSON records
the precise environment, gradients and source SHA-256 hashes.

## Reproduce in the existing project environment

Use the existing Python environment with PyTorch. No installation or network is needed.
For example, from the extracted audit directory:

```powershell
& 'C:\dev\marine-echo-jepa\.venv-gpu\Scripts\python.exe' `
  '.\audit_gradient_paths.py' `
  --source-root 'C:\dev\marine-echo-jepa\src' `
  --output '.\audit_results_local.json'
```

Replace the Python executable with the project's actual GPU/training environment path;
`.venv-gpu` is an example, not a verified local directory. The probe uses CPU even from a
GPU-capable environment. Use a new output filename on rerun; old evidence is not overwritten.
Do not reset your checkout merely to pass a hash check. If these two source files changed,
review the change and run a separately identified audit.

## Interpretation and next decision

Preserve historical results. Do not automatically execute the previous nine-trajectory
attribution campaign. First reproduce this source audit, inspect the saved 1,500-step direct
checkpoints and existing learning curves, and audit trained pretraining gradients/scales on
TRAIN-only batches. Treat loss-component gradients as diagnostics, not a forecast score.

The prior-year archive was included in expanded TRAIN. It cannot be external evaluation for
those expanded-data models or any descendants that use it. Previously frozen original-model
transfer results remain valid in their original descriptive scope.

A new architectural experiment should be justified by a specific finding and separately
reviewed, not selected to rescue a positive historical result. Avoid large sweeps, automatic
replication, or replacing this task with more release packaging.
