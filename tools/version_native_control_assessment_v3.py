"""Create additive assessment sources without editing fitted historical modules."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAIRS = {
    "src/marine_echo/evaluation/native_assessment_replication.py": "src/marine_echo/evaluation/native_assessment_controls_v3.py",
    "tools/execute_bounded_native_replication_assessment.py": "tools/execute_bounded_native_control_assessment_v3.py",
    "tools/execute_native_replication_assessment_worker.py": "tools/execute_native_control_assessment_worker_v3.py",
}

VALIDATOR = '''
def _typed_control_config(kind, config, spec, evidence):
    """Stdlib metadata checks before tensor/array decode; no method relabelling."""
    control_kind = "native_cf_control_weights_only_inference_v1"
    methods = {"cf_random_frozen", "cf_direct_supervised"}
    if kind != control_kind and spec.get("method") not in methods:
        return
    if kind != control_kind or spec.get("method") not in methods:
        raise ValueError("Distinct CF control kind and method required")
    frozen = spec["method"] == "cf_random_frozen"
    expected_mode = "frozen_readout" if frozen else "direct_end_to_end"
    fixed = {
        "method": spec["method"], "seed": spec.get("seed"), "history": 96,
        "heads": 4, "lr": 0.0003, "weight_decay": 0.0001,
        "gradient_clip": 1.0, "patience": 4, "min_daily_anchors": 18,
        "selection_policy": "scheduled_native_daily_pinball_earliest_strict_improvement_v1",
    }
    dimensions = {"width": 256, "latent": 128, "blocks": 5, "batch_size": 64,
                  "updates": 2000 if frozen else 3000, "cadence": 500 if frozen else 750}
    if (spec.get("mode") != expected_mode or type(spec.get("seed")) is not int
            or spec["seed"] not in (7, 13, 23) or set(config) != set(fixed) | set(dimensions)
            or any(config.get(k) != v or type(config.get(k)) is not type(v)
                   for k, v in fixed.items())):
        raise ValueError("Exact CF control recipe, mode, seed and fields required")
    if evidence == "SYNTHETIC_CORRECTNESS_ONLY":
        if (any(type(config[k]) is not int or not 1 <= config[k] <= ceiling
                for k, ceiling in dimensions.items())
                or config["cadence"] > config["updates"]):
            raise ValueError("Bounded synthetic control dimensions required")
    elif any(config[k] != v or type(config[k]) is not int for k, v in dimensions.items()):
        raise ValueError("Exact real CF control dimensions and supervision required")

'''


def create():
    if any((ROOT / dest).exists() for dest in PAIRS.values()):
        raise FileExistsError("Preserve existing versioned assessment files")
    names = {
        "native_assessment_replication": "native_assessment_controls_v3",
        "execute_bounded_native_replication_assessment": "execute_bounded_native_control_assessment_v3",
        "execute_native_replication_assessment_worker": "execute_native_control_assessment_worker_v3",
        "native_replication_frozen_assessment_completion_v2": "native_control_frozen_assessment_completion_v3",
    }
    for source, destination in PAIRS.items():
        text = (ROOT / source).read_text(encoding="utf-8")
        for old, new in names.items():
            text = text.replace(old, new)
        if "worker" not in source:
            text = text.replace(
                "NEURAL_KINDS = {",
                'NEURAL_KINDS = {\n    "native_cf_control_weights_only_inference_v1",',
                1,
            )
            text = text.replace(
                'METHODS = {"shared_ssl",',
                'METHODS = {"cf_random_frozen", "cf_direct_supervised", "shared_ssl",',
                1,
            )
            text = text.replace(
                '        _typed_band_config(kind, config, spec, m["evidence_kind"])',
                '        _typed_control_config(kind, config, spec, m["evidence_kind"])\n        _typed_band_config(kind, config, spec, m["evidence_kind"])',
                1,
            )
            text = text.replace("def _typed_band_config(", VALIDATOR + "def _typed_band_config(", 1)
            prefix = "ROOT_PACKAGE" if source.startswith("src/") else "package"
            text = text.replace(
                f'        {prefix} / "inference/native_acoustic.py",',
                f'        {prefix} / "inference/native_cf_controls.py",\n        {prefix} / "inference/native_acoustic.py",',
                1,
            )
        if source.startswith("src/"):
            text = text.replace(
                "*, ancestor_artifact_hashes=None):",
                "*, ancestor_artifact_hashes=None, expected_evidence=None):",
                1,
            )
            needle = '        if spec["mode"] == "core_frozen_readout":'
            addition = """        if kind == "native_cf_control_weights_only_inference_v1":
            from marine_echo.inference.native_cf_controls import NativeCFControlPredictor

            evidence = artifact.get("evidence_kind")
            if expected_evidence is not None and evidence != (
                "SYNTHETIC_CORRECTNESS_ONLY"
                if expected_evidence == "SYNTHETIC_CORRECTNESS_ONLY"
                else "REAL_TRAIN_DEVELOPMENT_FIT"
            ):
                raise ValueError("Control artifact evidence differs from admitted assessment")
            _typed_control_config(kind, config, spec, evidence)
            predictor = NativeCFControlPredictor(io.BytesIO(payloads[0]), device=device)
            if predictor.config.mode != spec["mode"]:
                raise ValueError("Control inference mode differs from frozen manifest")
            return predictor.forecast
"""
            text = text.replace(needle, addition + needle, 1)
            text = text.replace(
                '                ancestor_artifact_hashes=parent_hashes(_path(spec["ancestry_path"], base)),',
                '                ancestor_artifact_hashes=parent_hashes(_path(spec["ancestry_path"], base)),\n                expected_evidence=m["evidence_kind"],',
                1,
            )
            text = text.replace(
                '            "random_frozen": "untrained_control",',
                '            "random_frozen": "untrained_control",\n            "cf_random_frozen": "untrained_control",\n            "cf_direct_supervised": "supervised_feature_encoder",',
                1,
            )
        with (ROOT / destination).open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)


if __name__ == "__main__":
    create()
