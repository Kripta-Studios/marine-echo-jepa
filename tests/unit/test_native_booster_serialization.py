"""SYNTHETIC_CORRECTNESS_ONLY tiny real CPU boosters; in-memory text codecs."""

import ast
import hashlib
import importlib.util
import io
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pytest

BUILDER = Path(__file__).resolve().parents[2]
MAIN = BUILDER.parent / "marine-echo-jepa"
sys.path.insert(0, str(MAIN / "src"))
CANDIDATE = BUILDER / "evidence/ssl-platform-builder-v1/native_references.after.py"
SOURCE = CANDIDATE if CANDIDATE.exists() else MAIN / "src/marine_echo/training/native_references.py"
spec = importlib.util.spec_from_file_location("native_reference_platform_candidate", SOURCE)
references = importlib.util.module_from_spec(spec)
spec.loader.exec_module(references)


def fit_tiny_booster(horizon=1, quantile=0.5):
    """Generic polynomial/sinusoidal fixture, never real acoustic observations."""
    rng = np.random.default_rng(7)
    x = rng.normal(size=(128, 4))
    y = horizon * x[:, 0] + np.sin(x[:, 1]) + 0.2 * x[:, 2] ** 2
    regressor = lgb.LGBMRegressor(
        objective="quantile",
        alpha=quantile,
        n_estimators=12,
        min_child_samples=5,
        num_leaves=7,
        n_jobs=1,
        verbosity=-1,
        random_state=7,
        deterministic=True,
        force_col_wise=True,
    )
    regressor.fit(x, y)
    assessment = rng.normal(size=(29, 4))
    assessment[0, 1] = np.nan
    return regressor.booster_, assessment


@pytest.fixture
def text_transport(monkeypatch):
    """Exercise actual helper Path IO contracts without a durable write retry."""
    files, calls = {}, []
    original = Path.open
    prefix = BUILDER / "evidence/ssl-platform-builder-v1/SYNTHETIC_CORRECTNESS_ONLY_MEMORY"

    class TextFile(io.StringIO):
        def __init__(self, path, initial="", writable=False):
            super().__init__(initial)
            self.path, self.writable = path, writable

        def close(self):
            if self.writable:
                files[self.path] = self.getvalue()
            super().close()

    def opened(path, mode="r", *a, **k):
        if not path.is_relative_to(prefix):
            return original(path, mode, *a, **k)
        calls.append((mode, k.get("encoding"), k.get("newline")))
        if mode == "x":
            if path in files:
                raise FileExistsError(path)
            return TextFile(path, writable=True)
        assert mode == "r"
        return TextFile(path, files[path])

    monkeypatch.setattr(Path, "open", opened)
    monkeypatch.setattr(
        lgb.Booster,
        "save_model",
        lambda *a, **k: pytest.fail("Native filename writer was reached."),
    )
    init = lgb.Booster.__init__

    def init_without_filename(instance, *a, **k):
        assert not k.get("model_file"), "Native filename reader was reached."
        return init(instance, *a, **k)

    monkeypatch.setattr(lgb.Booster, "__init__", init_without_filename)
    return prefix, files, calls


@pytest.mark.parametrize("horizon", [1, 3, 6])
@pytest.mark.parametrize("quantile", [0.05, 0.25, 0.5, 0.75, 0.95])
def test_fifteen_fresh_real_tiny_boosters_text_and_prediction_round_trip(
    text_transport, horizon, quantile
):
    prefix, files, calls = text_transport
    booster, assessment = fit_tiny_booster(horizon, quantile)
    expected = booster.predict(assessment, num_threads=1)
    assert np.ptp(expected) > 0.01
    path = prefix / f"synthetic_Unicode_Á_é_h{horizon}_q{quantile:.2f}.txt"
    references.save_booster_text(booster, path)
    text = files[path]
    assert text == booster.model_to_string()
    assert "Tree=11" in text and "pandas_categorical:" in text
    replay = references.load_booster_text(path)
    np.testing.assert_array_equal(replay.predict(assessment, num_threads=1), expected)
    assert replay.num_trees() == booster.num_trees()
    assert replay.feature_name() == booster.feature_name()
    assert calls == [("x", "utf-8", ""), ("r", "utf-8", None)]
    original = hashlib.sha256(text.encode("utf-8")).hexdigest()
    with pytest.raises(FileExistsError):
        references.save_booster_text(booster, path)
    assert hashlib.sha256(files[path].encode("utf-8")).hexdigest() == original


def test_permission_error_propagates_once_without_native_or_alternate_retry(monkeypatch):
    booster, _ = fit_tiny_booster()
    calls = []

    def denied(path, *a, **k):
        calls.append((path, a, k))
        raise PermissionError("SYNTHETIC_CORRECTNESS_ONLY simulated permission denial")

    monkeypatch.setattr(Path, "open", denied)
    monkeypatch.setattr(
        lgb.Booster,
        "save_model",
        lambda *a, **k: pytest.fail("Denied Python write fell back to native IO."),
    )
    path = BUILDER / "evidence/ssl-platform-builder-v1/not_written_permission_fixture.txt"
    with pytest.raises(PermissionError):
        references.save_booster_text(booster, path)
    assert len(calls) == 1 and calls[0][0] == path


def test_fixed_reference_recipe_is_unchanged_and_uses_text_writer():
    before = ast.parse(
        (MAIN / "src/marine_echo/training/native_references.py").read_text(encoding="utf-8")
    )
    after = ast.parse(SOURCE.read_text(encoding="utf-8"))
    original = next(n for n in before.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    candidate = next(n for n in after.body if isinstance(n, ast.FunctionDef) and n.name == "run")

    class NormalizeWriter(ast.NodeTransformer):
        def visit_Expr(self, node):
            call = node.value
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "save_booster_text"
            ):
                return ast.parse("model.booster_.save_model(str(path))").body[0]
            return self.generic_visit(node)

    writers = [
        n
        for n in ast.walk(candidate)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "save_booster_text"
    ]
    assert len(writers) == 1
    assert ast.dump(NormalizeWriter().visit(original), include_attributes=False) == ast.dump(
        NormalizeWriter().visit(candidate), include_attributes=False
    )
