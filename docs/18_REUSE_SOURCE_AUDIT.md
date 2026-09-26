# Reuse and source audit

## Prefer a new small repository

Do not fork the whole logistics or industrial application. Reuse narrowly audited modules and
patterns with preserved license/attribution. Do not import old dataset loaders, source IDs,
customer naming, local paths, artifact claims or benchmark thresholds into the new domain.

The user's `Kripta-Studios/Kaleido-Project` README was read through the GitHub connector during
handoff preparation. It documents bounded evidence for a hybrid Phys-JEPA trajectory model
and also rejected ETA/delay heads [S17]. Its numerical results were not reproduced here and
are not Marine results. The point to reuse is a fair hybrid comparison, not a percentage target.

Candidate modules to inspect, verifying existence and exact revision before use:
```
Kripta-Studios/Kaleido-Project:
  src/flowtwin/models/ais_phys_jepa.py
  docs/decisions/0007-phys-jepa-clean-holdout-result.md
Kripta-Studios/industrial_jepa_mvp:
  src/sensor_jepa/models/dense_sensor_jepa.py
  STATUS.md
Kripta-Studios/predictiveops-worldmodel:
  src/forgeworld/world_models/sensor_jepa.py
lucas-maes/le-wm:
  jepa.py
  train.py
```

This is an inspection list, not an assertion that each current file has the desired semantics.
The previous industrial audit invalidated historical headline results. Verify status and actual
loss/gradient behaviour before copying. In particular, an encoder named shared may still detach
its targets, and a future-loss default may be zero. File names do not prove a LeWM implementation.

## Reproducible checkout procedure

On the owner's online machine, a shallow read-only source checkout is sufficient. Do not run
upstream install scripts or download weights without checking licenses and requirements.
Example commands are for an ignored external-reference directory:
```
git clone --depth 1 https://github.com/Kripta-Studios/Kaleido-Project.git external/Kaleido-Project
git clone --depth 1 https://github.com/lucas-maes/le-wm.git external/le-wm
git -C external/Kaleido-Project rev-parse HEAD
git -C external/le-wm rev-parse HEAD
```

Persist URL, returned commit SHA, license, inspected files and copied symbols in
`references/upstream-lock.json`. A referenced SHA from an earlier discussion is not a verified
current checkout. When a historical commit is required, fetch that exact allowed commit and
verify it, rather than assuming it is present in a shallow clone. No need to clone gigabytes
of unrelated repositories. Local network access during handoff preparation failed DNS, so no
clone or downloaded model is claimed to be bundled.

## Reuse matrix

Reuse: manifest/provenance patterns, EMA update mechanics, matched baselines, checkpoint/resume,
paired evaluation and quality-gate separation. Rewrite: acoustic input adapter, range-aware
tokenizer, causal time window construction, target definition and UI. Review: SIGReg integration,
calibration, batch dependence and any implicit context/target overlap. Never transfer molecular,
AIS or event-camera performance claims as evidence for this acoustic task.
