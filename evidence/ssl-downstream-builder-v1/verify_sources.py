"""CPU initialization/source inspection only; never load a corpus or optimize.

With --runner, dispatch the new builder CLI against main's read-only helpers.
This path bootstrap is unnecessary after the parent integrates the new module.
"""

import json
import sys
from pathlib import Path

BUILDER = Path(__file__).resolve().parents[2]
MAIN = BUILDER.parent / "marine-echo-jepa"
sys.path.insert(0, str(MAIN / "src"))
import marine_echo.training

marine_echo.training.__path__.append(str(BUILDER / "src/marine_echo/training"))
from marine_echo.training import native_downstream as ds

if __name__ == "__main__":
    if sys.argv[1:2] == ["--runner"]:
        ds.main(sys.argv[2:])
    else:
        records = []
        sources = {
            Path(ds.__file__).resolve(),
            Path(__file__).resolve(),
            BUILDER / "tests/integration/test_native_downstream.py",
        }
        for method in ("shared_ssl", "cf_jepa"):
            config = ds.DownstreamConfig(method=method)
            core_config = ds.core.Config(method=method)
            model = ds.prepare_model(config, core_config, None)
            sources.update(ds.core.required_sources(core_config))
            package = Path(ds.core.__file__).resolve().parents[1]
            sources.update(
                [
                    package / "__init__.py",
                    package / "models/__init__.py",
                    package / "training/__init__.py",
                ]
            )
            records.append(
                {
                    "method": method,
                    "dimensions": ds.core.model_dimensions(core_config),
                    "total_parameters": sum(p.numel() for p in model.parameters()),
                    "encoder_parameters": sum(p.numel() for p in model.encoder.parameters()),
                    "head_parameters": sum(p.numel() for p in model.readout.parameters()),
                    "frozen_optimized_parameters": sum(
                        p.numel() for p in model.parameters() if p.requires_grad
                    ),
                }
            )
        print(
            json.dumps(
                {
                    "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY",
                    "operation": "CPU_INITIALIZATION_AND_SOURCE_HASHING_NO_FIT",
                    "python": sys.version,
                    "torch": ds.torch.__version__,
                    "cuda_initialized": ds.torch.cuda.is_initialized(),
                    "parameters": records,
                    "source_sha256": {str(p.resolve()): ds.sha256(p) for p in sorted(sources)},
                },
                indent=2,
            )
        )
