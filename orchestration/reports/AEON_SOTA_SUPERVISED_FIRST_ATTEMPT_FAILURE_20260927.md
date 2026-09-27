# First post-hoc LightGBM invocation: prefit row-probe failure

Date: 2026-09-27. Classification: failed execution attempt, no model fit or validation prediction.

The independently approved initial runner (`aeon_sota_supervised.py` SHA-256 `922d7cbd842d25b17f0892249ba6cbb7e9c799dcb00cfdb64deb931cfce3122f`) exited 1 after reconstructing the reviewed TRAIN/validation cohort but before fitting any of its 15 quantile heads. It called the repository's exclusive-write `_save_predictions` on a path already created by `NamedTemporaryFile`, raising `FileExistsError` in `path.open("xb")`. The requested final output directory was not created. Its redirected stdout log at `outputs/aeon3_geb_2024_hourly_sv_v1/sota_supervised_lightgbm_cli.log` is empty; the process traceback and exit status were inspected in the execution session.

The focused regression test first failed at import of the absent row-probe helper. The repair at commit `974b349` uses a new filename inside a temporary directory, which satisfies exclusive creation. All five focused tests, Ruff and mypy pass. This changes the runner hash and requires a refreshed distinct prefit review before a new invocation. The fixed LightGBM recipe, source, split, metric and model selection were not changed. CAL/TEST values remain unopened.
