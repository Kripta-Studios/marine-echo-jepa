# Marine Echo JEPA: offline research engineering release

Run `Run-V2-Research.ps1` on Windows with Python 3.12 and `uv` available. The launcher
verifies every packaged file, installs application dependencies from the bundled
wheelhouse with network access disabled, and serves the app on loopback.

This distinct release presents independently reviewed TRAIN-development results
for a transformed AZFP instrument response code. Both ridge and direct neural
models produced real April development predictions. Ridge had lower daily mean
pinball loss at 1, 3, and 6 hours. The complete 212 issued-row table is available
in the app and `research/artifacts/raw-development.json`; the saved model predictions,
step-64 and step-128 direct checkpoints, result, protocol decisions, and reviews
are in `provenance/`.

These response codes are not calibrated Sv. The calibrated v2 target failed its
April support gate, so its 25-slot core model/seed/control campaign and JEPA
comparison were not run. The preserved v1 registry remains blocked with zero
updates. There is no sealed holdout or final evaluation and no fish, catch,
biomass, operational, or commercial validation claim.

`SOURCE_REVISION.json` and `SHA256SUMS` identify and verify this package. The
historical r2 release is separate and unchanged.
