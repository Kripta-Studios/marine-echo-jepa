"""Read synthetic fixture source only, not public artifacts or numerical data."""
import ast
from pathlib import Path

MAIN = Path(__file__).resolve().parents[3] / "marine-echo-jepa"
raw = (MAIN / "tests/integration/test_native_prefix_transfer.py").read_text(encoding="utf-8")
names = {"trajectory_inputs", "test_actual_optimizer_changes_only_declared_encoder_and_four_dev_choices", "test_exact_resume_cpu_with_optimizer_scheduler_rng_samples_and_portable_replay"}
for node in ast.parse(raw).body:
    if isinstance(node, ast.FunctionDef) and node.name in names:
        print(ast.get_source_segment(raw, node))
