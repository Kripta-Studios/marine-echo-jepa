"""ROOT-only SYNTHETIC_CORRECTNESS_ONLY physical CPU fixture, never a fit approval.

Run only after integration and after every scientific tree/journal has closed.
The parent checks genuine ROOT ownership state before creating any output or child.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check_sources():
    baseline = json.loads((HERE / "protected-baseline-v3.json").read_bytes())
    for relative in (
        "tools/native_reference_supervisor.py",
        "tools/native_band_budget_history_v2.py",
        "src/marine_echo/training/native_prefix_transfer.py",
        "src/marine_echo/training/native_ssl.py",
        "src/marine_echo/training/native_desktop_resources_v2.py",
        "orchestration/native_vlc_desktop_owner_resolution_v1.json",
    ):
        path = ROOT / relative
        expected = baseline[str(path)]["sha256"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Changed ROOT fixture dependency: {path}")
    for relative, expected in json.loads(
        (HERE / "root-fixture-dependencies-v3.json").read_bytes()
    ).items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Changed private synthetic fixture dependency: {relative}")


def child_checks(output):
    import io
    from unittest.mock import patch

    import numpy as np
    import torch

    from marine_echo.training import native_prefix_desktop_transfer_v3 as prefix

    def forbidden(*args, **kwargs):
        raise AssertionError("SYNTHETIC_CORRECTNESS_ONLY CPU fixture attempted CUDA initialization")

    # These are private test controls, never production admission hooks or factory swaps.
    with (
        patch.object(torch.cuda, "_lazy_init", forbidden),
        patch.object(torch.cuda, "is_available", return_value=False),
        patch.object(torch.cuda, "is_initialized", return_value=False),
        patch.object(torch.cuda, "manual_seed_all", return_value=None),
        prefix.desktop_runtime.Resources("cpu", output) as resources,
    ):
        fixture = load_module(
            "desktop_root_private_prefix_fixtures",
            ROOT / "tests/integration/test_native_prefix_transfer.py",
        )
        # Reuse immutable synthetic inputs/assertions with the additive module under test.
        # This changes only the newly loaded test module, never original production globals.
        fixture.prefix = prefix
        executed = []
        for mode in ("scratch_direct", "frozen_readout"):
            fixture.test_actual_optimizer_changes_only_declared_encoder_and_four_dev_choices(mode)
            fixture.test_exact_resume_cpu_with_optimizer_scheduler_rng_samples_and_portable_replay(
                mode
            )
            args = fixture.trajectory_inputs(mode)
            trajectory = prefix._Trajectory(*args)
            trajectory.advance()
            checkpoint = output / f"{mode}-latest.pt"
            inference = output / f"{mode}-inference.pt"
            with checkpoint.open("xb") as stream:
                stream.write(prefix.encode_checkpoint(trajectory.checkpoint()))
            with inference.open("xb") as stream:
                stream.write(prefix.encode_checkpoint(trajectory.inference_artifact()))
            with checkpoint.open("rb") as stream:
                saved = torch.load(stream, weights_only=True, map_location="cpu")
            restored = prefix._Trajectory(*args)
            restored.restore(saved)
            assert restored.samples == trajectory.samples
            assert all(
                torch.equal(value, restored.model.state_dict()[key])
                for key, value in trajectory.model.state_dict().items()
            )
            with inference.open("rb") as stream:
                portable = prefix.PrefixPredictor(io.BytesIO(stream.read()))
            data = args[4]
            np.testing.assert_array_equal(
                portable.forecast(
                    data["x"], data["context_observed"], data["metadata"], data["query"]
                ),
                prefix.predict(portable.model, data, portable.scalers, args[0]),
            )
            executed.append(
                {
                    "mode": mode,
                    "updates": trajectory.step,
                    "selected_step": trajectory.selected_step,
                }
            )
        receipt = {
            "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY",
            "device": "cpu",
            "scientific_fitting": False,
            "private_synthetic_optimizer_execution": True,
            "checks": executed,
            "resources": resources.snapshot(),
        }
    with (output / "synthetic-completion.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if ROOT.name != "marine-echo-jepa":
        raise RuntimeError(
            "ROOT-only NOT_RUN in builder; do not retry denied optimizer/cache operations"
        )
    output = args.output.resolve()
    if not output.is_relative_to(HERE) or "SYNTHETIC_CORRECTNESS_ONLY" not in output.name:
        raise ValueError("New visibly synthetic output beneath this evidence folder required")
    check_sources()
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "tools"))
    if args.child:
        child_checks(output)
        return 0
    wrapper = load_module(
        "desktop_root_fixture_policy", ROOT / "tools/execute_native_prefix_desktop_job_v3.py"
    )
    ledger_path = ROOT / "orchestration/native_ssl_run_ledger_v1.json"
    wrapper.validate_idle_desktop_ledger(
        json.loads(ledger_path.read_bytes()),
        ledger_path,
        ROOT / "evidence/ssl-builder-v1/gpu-owner.lock",
    )
    if output.exists() or output.is_symlink():
        raise FileExistsError("Never overwrite or retry a prior synthetic output")
    output.mkdir()
    started = time.monotonic()
    command = [
        sys.executable,
        "-B",
        str(Path(__file__).resolve()),
        "--child",
        "--output",
        str(output),
    ]
    with (output / "console.log").open("x", encoding="utf-8") as stream:
        child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        receipt = wrapper.supervise_owned(
            child, started=started, deadline_seconds=600, rss_limit_bytes=22 * 2**30
        )
    receipt.update(
        {"evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY", "command": command, "child_pid": child.pid}
    )
    with (output / "owned-process.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return (
        0
        if (
            receipt["exit_code"] == 0
            and receipt["stopped_for"] is None
            and receipt["owned_tree_cleanup_verified"] is True
            and (output / "synthetic-completion.json").is_file()
        )
        else 1
    )


if __name__ == "__main__":
    sys.exit(main())
