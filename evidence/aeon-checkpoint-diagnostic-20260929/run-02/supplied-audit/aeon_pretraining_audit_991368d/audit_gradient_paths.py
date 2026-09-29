"""Bounded synthetic autograd probe, NOT an AEON training experiment.

Runs the two exact GitHub source blobs at commit 991368d with random inputs.
No acoustic data, existing checkpoints, or user workspace files are accessed.
Assertions check gradient support, not forecasting quality or cause of collapse.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPECTED = {
    "aeon_ssl.py": "7bd8a94cde303d968df1e541af47b50ce969d342",
    "sigreg.py": "ef6f77225d03c32daa8d6509401e998658f2b198",
}


def verify_sources(source_root: Path) -> dict[str, dict[str, str]]:
    result = {}
    for name, expected in EXPECTED.items():
        path = source_root / "marine_echo" / "models" / name
        content = path.read_bytes()
        digest = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        if digest != expected:
            raise ValueError(f"Source identity mismatch: {name}")
        result[name] = {"git_blob_sha1": digest, "sha256": hashlib.sha256(content).hexdigest()}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path.cwd() / "src",
                        help="Existing repository src directory; exact source hashes are required.")
    parser.add_argument("--output", type=Path, default=Path.cwd() / "audit_results_local.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite prior evidence; choose a new --output path.")
    sources = verify_sources(args.source_root.resolve())
    # Isolate these reviewed modules from unrelated package initialization.
    import types
    package = types.ModuleType("marine_echo")
    package.__path__ = [str(args.source_root.resolve() / "marine_echo")]
    subpackage = types.ModuleType("marine_echo.models")
    subpackage.__path__ = [str(args.source_root.resolve() / "marine_echo" / "models")]
    sys.modules["marine_echo"] = package
    sys.modules["marine_echo.models"] = subpackage
    import torch
    from torch.nn import functional as F
    from marine_echo.models.aeon_ssl import AeonDirect, AeonTemporalSSL

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rows = []
    initialization = []
    for seed in (7, 13, 23):
        generator = torch.Generator().manual_seed(seed + 1000)
        values = torch.randn(64, 24, 4, generator=generator)
        truth = torch.randn(64, 3, generator=generator)
        for pattern in ("all_observed", "ten_percent_missing"):
            mask = (torch.ones_like(values, dtype=torch.bool) if pattern == "all_observed"
                    else torch.rand(values.shape, generator=generator) > 0.1)
            for mode in ("ema", "shared_sigreg", "direct"):
                torch.manual_seed(seed)
                model = AeonDirect() if mode == "direct" else AeonTemporalSSL(mode=mode)
                if mode == "direct":
                    predicted = model(values, mask)
                    quantiles = torch.tensor([.05, .25, .5, .75, .95])
                    errors = truth[:, :, None] - predicted
                    loss = torch.maximum(quantiles * errors, (quantiles - 1.) * errors).mean()
                else:
                    loss = model.pretrain_loss(values, mask)
                assert torch.isfinite(loss)
                loss.backward()
                gradient = model.encoder.network[0].weight.grad
                assert gradient is not None
                prefix, suffix = gradient[:, :144], gradient[:, 144:]
                assert prefix.norm().item() > 0
                suffix_nonzero = torch.count_nonzero(suffix).item()
                if mode == "ema":
                    assert suffix_nonzero == 0
                    assert all(parameter.grad is None for parameter in model.teacher.parameters())
                    assert model.head.weight.grad is None
                else:
                    assert suffix_nonzero > 0
                rows.append({
                    "seed": seed, "mask_pattern": pattern, "mode": mode,
                    "objective": "supervised_pinball" if mode == "direct" else "pretrain_loss",
                    "first_linear_weight_shape": list(gradient.shape),
                    "prefix_columns": 144, "suffix_columns": 48,
                    "suffix_parameter_count": suffix.numel(),
                    "suffix_nonzero_gradients": suffix_nonzero,
                    "prefix_gradient_l2": prefix.norm().item(),
                    "suffix_gradient_l2": suffix.norm().item(),
                    "teacher_has_parameter_gradients": (any(p.grad is not None for p in model.teacher.parameters())
                        if mode == "ema" else None),
                })
        torch.manual_seed(seed)
        direct = AeonDirect()
        torch.manual_seed(seed)
        ema = AeonTemporalSSL(mode="ema")
        encoder_equal = all(torch.equal(v, ema.encoder.state_dict()[k])
                            for k, v in direct.encoder.state_dict().items())
        head_equal = all(torch.equal(v, ema.head.state_dict()[k])
                         for k, v in direct.head.state_dict().items())
        assert encoder_equal and not head_equal
        initialization.append({"seed": seed, "encoder_initialization_equal": encoder_equal,
                               "forecast_head_initialization_equal": head_equal})

    # Report how many statistical draws correspond to one average full-window pass.
    exposure = {"original_direct": 3000*64/4965, "expanded_direct": 3000*64/13472,
                "original_ema_supervised": 1500*64/4965,
                "expanded_ema_supervised": 1500*64/13472}
    payload = {
        "status": "ALL_SYNTHETIC_ASSERTIONS_PASSED",
        "scope": "SOURCE_GRADIENT_AND_INITIALIZATION_AUDIT_NOT_AN_AEON_BENCHMARK",
        "commit": "991368d", "source_identity": sources,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu"},
        "gradient_probes": rows, "initialization_probes": initialization,
        "approximate_nominal_window_exposure": exposure,
        "limitations": [
            "Synthetic random inputs and initialized weights only; no trained AEON checkpoint was inspected.",
            "Zero suffix data-gradient follows from fixed-position masking and detached EMA targets.",
            "AdamW may still decay those columns; later layers change and supervised fine-tuning can learn them.",
            "These findings do not establish the cause of the reported rank/scale or generalization failures.",
            "Averages of sample presentations are not independent epochs or effective sample sizes.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(payload["status"])
    print("gradient probes:", len(rows), "; initialization probes:", len(initialization))
    for mode in ("ema", "shared_sigreg", "direct"):
        items = [row for row in rows if row["mode"] == mode]
        print(mode, "suffix_nonzero_gradient_counts", sorted(set(row["suffix_nonzero_gradients"] for row in items)),
              "prefix_gradient_range", min(row["prefix_gradient_l2"] for row in items), max(row["prefix_gradient_l2"] for row in items))
    print("initialization:", initialization)
    print("nominal average presentations:", exposure)


if __name__ == "__main__":
    main()
