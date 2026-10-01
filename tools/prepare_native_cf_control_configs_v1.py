"""Register six fixed extension cells. No fit, decode or approval is performed."""

import json
from pathlib import Path

from execute_native_cf_control_job_v1 import ROOT, catalog, digest, source_paths, write_new


def prepare(root=ROOT):
    root = Path(root).resolve()
    if root != ROOT:
        raise ValueError("Registration belongs to the integrated research checkout")
    destination = root / "orchestration/native_cf_matched_controls_v1.json"
    folder = root / "configs/native_cf_controls_v1"
    recipes = catalog()
    config_paths = {k: folder / (k + ".json") for k in recipes}
    if destination.exists() or any(p.exists() for p in config_paths.values()):
        raise FileExistsError("Preserve previous control registrations")
    from marine_echo.training.native_cf_controls import Config

    for value in recipes.values():
        config = Config(**value)
        config.validate()
        if config.to_dict() != value:
            raise ValueError("Control factory differs from fixed executable catalog")
    required = [
        *source_paths(root),
        root / "docs/adr/0022-cf-backbone-matched-controls.md",
        root / "docs/adr/0025-cf-matched-controls-continuation.md",
        root / "docs/adr/0016-native-ssl-selection-and-source-baseline.md",
        root / "orchestration/ssl_vnext_cf_matched_controls_builder_contract.txt",
        root / "orchestration/native_cf_control_budget_owner_resolution_v1.json",
        root / "orchestration/native_cf_control_owner_scope_v1.json",
        root / "configs/native_ssl_split_v1.json",
        root / "data/processed/native_ssl_v1/train.npz",
        root / "data/processed/native_ssl_v1/development.npz",
        root / "data/processed/native_ssl_v1/train.json",
        root / "data/processed/native_ssl_v1/development.json",
        root / ".venv/Scripts/python.exe",
        root / "pyproject.toml",
        root / "uv.lock",
        Path(__file__).resolve(),
    ]
    bindings = {str(p.resolve()): digest(p) for p in required}
    original = root / "orchestration/native_development_comparison_v3.json"
    previous = json.loads(original.read_bytes())
    comparisons = {
        name: path
        for name, path in previous["methods"].items()
        if name.startswith(("cf_jepa_frozen_readout", "cf_jepa_full_finetune"))
        or name == "lightgbm"
    }
    if len(comparisons) != 7:
        raise ValueError("Exactly six existing strong CF endpoints and LightGBM required")
    bindings[str(original)] = digest(original)
    rows = []
    for identifier, value in recipes.items():
        output = root / "outputs/native_cf_matched_controls_v1" / identifier
        receipt = root / "evidence/ssl-research-v1" / (identifier + "-attempt-01")
        review = root / "evidence/ssl-research-v1" / (identifier + "-prefit-review-final.json")
        if any(p.exists() for p in (output, receipt, review)):
            raise FileExistsError("Prospective control output/receipt/review must be absent")
        rows.append(
            {
                "id": identifier,
                "config": str(config_paths[identifier]),
                "method": value["method"],
                "seed": value["seed"],
                "output": str(output),
                "receipt": str(receipt),
                "review": str(review),
                "resume": None,
                "parent": None,
                "ssl_updates": 0,
                "status": "PREFIT_REQUIRED_NOT_RUN",
            }
        )
    folder.mkdir(parents=True, exist_ok=True)
    for identifier, value in recipes.items():
        write_new(config_paths[identifier], value)
        bindings[str(config_paths[identifier])] = digest(config_paths[identifier])
    result = {
        "study": "CF_MATCHED_CONTROLS",
        "status": "REGISTERED_EXACT_RECIPES_NOT_PREFIT_APPROVAL",
        "default_fits": 6,
        "planned_supervised_updates": 15000,
        "jobs": rows,
        "bindings": bindings,
        "existing_comparison_predictions": comparisons,
        "original_campaign": {"neural_methods": 43, "total_methods": 47, "modified": False},
        "feature_semantics": "Original CF forecasting encoder branch, full-H96 observed context; unchanged sampled-pretraining-crop mismatch disclosed in all arms",
        "initialization": "Original core.initialize_model(cf_jepa,seed), fresh QueryHead seed+100000",
        "numeric_decoding": False,
        "scientific_fitting": "NOT_RUN",
        "independent_prefit": "REQUIRED",
        "final_numeric_access": False,
        "extension_comparison": "SEPARATE_FROM_ORIGINAL_47",
    }
    write_new(destination, result)
    return result


if __name__ == "__main__":
    value = prepare()
    print(json.dumps({"status": value["status"], "fits": 6, "planned_supervised_updates": 15000}))
