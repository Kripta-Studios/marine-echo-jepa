"""Guarded reconstruction from reviewed saved native prediction artifacts.

No inference, model selection, acoustic corpus or model weights are consumed.
Source-calendar dates are proxies; no UTC interpretation is introduced.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from datetime import date
from pathlib import Path

import numpy as np

import marine_echo
import marine_echo.evaluation
from marine_echo.evaluation import native_product

# Owner amendment and the runtime CODEX_THREAD_ID both identify this builder.
# Freeze identity in source; never replace it with the assessment caller's ID.
IMPLEMENTER_SESSION_ID = "01a0ef20-7397-7ba1-a98c-f59bc38ddcc1"
HORIZONS = (1, 3, 6)
RECIPE = {"bootstrap_seed": 1729, "bootstrap_replicates": 2000, "block_days": 7, "floor": 18}
REQUIRED = {"predictions", "targets", "observed", "target_dates", "deployment", "row_id", "query"}
STATUSES = {
    "development": "APPROVED_COMPARISON_RECONSTRUCTION",
    "final_test": "APPROVED_FINAL_ASSESSMENT",
}
EVIDENCE_KINDS = {"SYNTHETIC_CORRECTNESS_ONLY", "REVIEWED_SAVED_PREDICTIONS"}
SOURCE_PATHS = tuple(
    Path(p).resolve()
    for p in (
        __file__,
        native_product.__file__,
        marine_echo.__file__,
        marine_echo.evaluation.__file__,
    )
)
IMPORTED_SOURCE_HASHES = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE_PATHS}


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path):
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs)
    if not isinstance(value, dict):
        raise TypeError("Manifest/review must be JSON objects.")
    return value, raw


def _identity(value, name):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"Explicit {name} identity is required.")
    return value


def _resolve(value, base):
    if not isinstance(value, str) or not value:
        raise ValueError("An exact artifact/protocol path is required.")
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def _admit(manifest_path, review_path, output_path):
    # Both policy documents precede hash inspection and any numeric parsing.
    manifest, manifest_bytes = _read_json(manifest_path)
    review, review_bytes = _read_json(review_path)
    role = manifest.get("role")
    if role not in STATUSES or review.get("status") != STATUSES[role]:
        raise ValueError("Role requires its exact distinct comparison/assessment review status.")
    admitted = review.get("allowed_roles")
    if (
        not isinstance(admitted, list)
        or not admitted
        or any(v not in STATUSES for v in admitted)
        or len(set(admitted)) != len(admitted)
        or role not in admitted
    ):
        raise ValueError("Review must explicitly admit the requested role.")
    implementer = _identity(manifest.get("implementer_session_id"), "declared implementer")
    coordinator = _identity(manifest.get("root_coordinator_session_id"), "root coordinator")
    reviewer = _identity(review.get("reviewer_session_id"), "reviewer")
    excluded = {IMPLEMENTER_SESSION_ID, implementer, coordinator}
    for field in ("implementer_session_id", "root_coordinator_session_id"):
        if field in review:
            if review[field] != manifest[field]:
                raise ValueError("Review and manifest session identities differ.")
            excluded.add(review[field])
    if reviewer.casefold() in {value.casefold() for value in excluded}:
        raise ValueError(
            "Reviewer must be distinct from builder, coordinator and declared implementer."
        )
    kind = manifest.get("evidence_kind")
    if kind not in EVIDENCE_KINDS or review.get("evidence_kind") != kind:
        raise ValueError("Manifest/review require identical explicit evidence kinds.")
    # Synthetic labeling changes no gate, role, hashing or numerical algorithm.
    for key, fixed in RECIPE.items():
        if type(manifest.get(key)) is not int or manifest[key] != fixed:
            raise ValueError(f"Prespecified comparison recipe requires {key}={fixed}.")
    methods = manifest.get("methods")
    if (
        not isinstance(methods, dict)
        or len(methods) < 2
        or any(not isinstance(k, str) or not k.strip() for k in methods)
    ):
        raise ValueError("At least two named, prespecified methods are required.")
    if not isinstance(manifest.get("reference"), str) or manifest["reference"] not in methods:
        raise ValueError("One exact prespecified reference name must be listed.")
    paths = {name: _resolve(value, manifest_path.parent) for name, value in methods.items()}
    if any(path.suffix.lower() != ".npz" for path in paths.values()):
        raise ValueError("Only saved prediction NPZ artifacts are accepted.")
    required = [*SOURCE_PATHS, manifest_path, *paths.values()]
    if "scientific_protocol_path" in manifest:
        required.append(_resolve(manifest["scientific_protocol_path"], manifest_path.parent))
    bindings = review.get("bindings")
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError("Exact immutable source/manifest/prediction bindings are required.")
    for key, value in bindings.items():
        if (
            not isinstance(key, str)
            or not Path(key).is_absolute()
            or str(Path(key).resolve()) != key
            or not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdefABCDEF" for c in value)
        ):
            raise ValueError("Bindings must map exact absolute paths to SHA256 digests.")
    if output_path in set(required) | {review_path} or output_path.exists():
        raise FileExistsError(
            "Output must be a new path, distinct from every admitted input/source."
        )
    snapshots, exact = {}, {}
    for path in dict.fromkeys(required):
        key = str(path)
        if key not in bindings:
            raise ValueError(f"Missing required review binding: {key}")
        payload = manifest_bytes if path == manifest_path else path.read_bytes()
        actual = _sha(payload)
        if actual != bindings[key].lower() or (
            key in IMPORTED_SOURCE_HASHES and actual != IMPORTED_SOURCE_HASHES[key]
        ):
            raise ValueError(f"Stale immutable review/source binding: {key}")
        exact[key] = actual
        if path in paths.values():
            snapshots[path] = payload
    provenance = {
        "bindings": exact,
        "prediction_artifacts": {
            name: {"path": str(path), "sha256": exact[str(path)]} for name, path in paths.items()
        },
        "manifest_sha256": _sha(manifest_bytes),
        "review_sha256": _sha(review_bytes),
        "reviewer_session_id": reviewer,
        "builder_session_id": IMPLEMENTER_SESSION_ID,
        "implementer_session_id": implementer,
        "root_coordinator_session_id": coordinator,
    }
    return manifest, paths, snapshots, provenance


def _query(query, n):
    if query.shape != (n, 3, 10) or query.dtype.kind != "f" or not np.isfinite(query).all():
        raise ValueError("Complete finite native queries [N,3,10] are required.")
    fixed = {0: 38000 / 455000, 1: 1, 2: 1, 5: 0}
    if any(
        not np.allclose(query[..., key], value, rtol=0, atol=1e-6) for key, value in fixed.items()
    ):
        raise ValueError("Native query frequency/interval/integrated-product/orientation differs.")
    if (
        (query[..., 3] < 0).any()
        or (query[..., 4] <= query[..., 3]).any()
        or not np.isin(query[..., 6:9], [0, 1]).all()
        or not np.allclose(query[..., 9], HORIZONS, rtol=0, atol=1e-6)
    ):
        raise ValueError("Native query bounds, known flags or horizons are invalid.")


def _load(payload, role, evidence_kind):
    with np.load(io.BytesIO(payload), allow_pickle=False) as saved:
        if not REQUIRED.issubset(saved.files) or {
            "x",
            "future",
            "future_observed",
            "encoder",
            "weights",
        }.intersection(saved.files):
            raise ValueError("Expected saved forecasts/support, never corpus or model inputs.")
        a = {key: saved[key].copy() for key in saved.files}
    target, mask, prediction = a["targets"], a["observed"], a["predictions"]
    if (
        target.ndim != 2
        or target.shape[1] != 3
        or not len(target)
        or target.dtype.kind not in "fi"
        or mask.shape != target.shape
        or mask.dtype != np.bool_
        or prediction.shape != (*target.shape, 5)
        or prediction.dtype.kind not in "fi"
    ):
        raise ValueError("Saved numeric prediction/target/Boolean-mask shapes are invalid.")
    if (
        not np.isfinite(prediction).all()
        or not np.isfinite(target[mask]).all()
        or np.any(np.diff(prediction, axis=-1) < 0)
    ):
        raise ValueError(
            "All forecasts and observed targets must be finite; quantiles must be ordered."
        )
    n = len(target)
    for key in ("row_id", "deployment"):
        if a[key].shape != (n,) or a[key].dtype.kind != "U" or any(not v for v in a[key]):
            raise ValueError("Nonempty Unicode row/deployment identities are required.")
    pairs = list(zip(a["deployment"].tolist(), a["row_id"].tolist(), strict=True))
    if len(set(pairs)) != n:
        raise ValueError("Duplicate row_id+deployment identity.")
    dates = a["target_dates"]
    if dates.shape != (n, 3) or dates.dtype.kind != "U":
        raise ValueError("Source-calendar dates must be Unicode [N,3].")
    for value in set(dates.ravel().tolist()) - {""}:
        if date.fromisoformat(value).isoformat() != value:
            raise ValueError("Target dates must be exact source-calendar YYYY-MM-DD proxies.")
    _query(a["query"], n)
    for key, expected in (("role", role), ("corpus_role", role), ("evidence_kind", evidence_kind)):
        if key in a and (a[key].shape != () or a[key].item() != expected):
            raise ValueError(f"Saved artifact {key} differs from admitted manifest.")
    order = np.array(sorted(range(n), key=pairs.__getitem__))
    return {
        key: value[order].copy() if value.ndim and len(value) == n else value
        for key, value in a.items()
    }


def _common(a, b):
    for key in ("row_id", "deployment", "observed", "target_dates", "query"):
        if not np.array_equal(a[key], b[key]):
            raise ValueError(f"Common support differs: {key}; intersections are forbidden.")
    if not np.array_equal(a["targets"][a["observed"]], b["targets"][b["observed"]]):
        raise ValueError("Observed target values differ on the exact common support.")


def _support(a):
    expected = {}
    for dep in sorted(set(a["deployment"].tolist())):
        for h, horizon in enumerate(HORIZONS):
            valid = (a["deployment"] == dep) & a["observed"][:, h]
            dates, counts = np.unique(a["target_dates"][valid, h], return_counts=True)
            for value, count in zip(dates, counts, strict=True):
                if value and count >= RECIPE["floor"]:
                    expected[(dep, horizon, str(value))] = int(count)
    return expected


def _variation(prediction, deployments):
    ranges = np.ptp(prediction, axis=0)
    return {
        "diagnostic_only": True,
        "constant_across_issuance": bool((ranges == 0).all()),
        "median_range_db_per_horizon": ranges[:, 2].tolist(),
        "median_std_db_per_horizon": prediction[:, :, 2].std(axis=0).tolist(),
        "quantile_range_db_per_horizon": ranges.tolist(),
        "per_deployment": {
            dep: {
                "issued_rows": int((deployments == dep).sum()),
                "quantile_range_db_per_horizon": np.ptp(
                    prediction[deployments == dep], axis=0
                ).tolist(),
                "constant_across_issuance": bool(
                    (np.ptp(prediction[deployments == dep], axis=0) == 0).all()
                ),
            }
            for dep in sorted(set(deployments.tolist()))
        },
    }


def _array_json(a):
    def safe(value):
        if isinstance(value, list):
            return [safe(v) for v in value]
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, float) and not np.isfinite(value):
            return "NaN" if np.isnan(value) else "Infinity" if value > 0 else "-Infinity"
        return value

    if a.dtype.kind not in "biufUS":
        raise ValueError("Optional provenance must be safe numeric/string arrays.")
    return {"dtype": a.dtype.str, "shape": list(a.shape), "values": safe(a.tolist())}


def _geometry(query):
    products = np.unique(query[..., :9].reshape(-1, 9), axis=0)
    return {
        "horizon_intervals": list(HORIZONS),
        "dB_rescaling": False,
        "unique_products": [
            {
                "frequency_hz": float(v[0] * 455000),
                "interval_seconds": float(v[1] * 3600),
                "geometry_kind": float(v[2]),
                "upper_m": float(v[3] * 250),
                "lower_m": float(v[4] * 250),
                "orientation": float(v[5]),
                "processing_known": float(v[6]),
                "instrument_known": float(v[7]),
                "clock_known": float(v[8]),
                "original_encoded_measurement_fields": v.tolist(),
            }
            for v in products
        ],
    }


def _blocks(a, names, methods):
    deployments = sorted(set(a["deployment"].tolist()))
    records, tensors = [], []
    for dep in deployments:
        selected = a["deployment"] == dep
        all_dates = set(a["target_dates"][selected].ravel().tolist()) - {""}
        observed_dates = set(a["target_dates"][selected][a["observed"][selected]].tolist()) - {""}
        anchor = min(all_dates) if all_dates else None

        def block(value, anchor=anchor):
            return (date.fromisoformat(value) - date.fromisoformat(anchor)).days // 7

        ids = sorted({block(v) for v in observed_dates})
        sums = np.zeros((len(ids), len(names), 3), np.float64)
        counts = np.zeros((len(ids), 3), np.int64)
        lookup = {value: i for i, value in enumerate(ids)}
        eligible_dates = {h: [] for h in HORIZONS}
        for m, name in enumerate(names):
            for row in methods[name]["metrics"]["daily_rows"]:
                if row["deployment"] == dep:
                    i, h = (
                        lookup[block(row["target_source_date"])],
                        HORIZONS.index(row["horizon_intervals"]),
                    )
                    sums[i, m, h] += row["pinball_db"]
                    if m == 0:
                        counts[i, h] += 1
                        eligible_dates[HORIZONS[h]].append(row["target_source_date"])
        span = (
            (date.fromisoformat(max(observed_dates)) - date.fromisoformat(min(observed_dates))).days
            if observed_dates
            else None
        )
        records.append(
            {
                "deployment": dep,
                "anchor_source_date": anchor,
                "observed_calendar_block_ids": ids,
                "observed_source_date_count": len(observed_dates),
                "source_calendar_span_days": span,
                "zero_span": span == 0,
                "single_observed_block": len(ids) == 1,
                "eligible_source_dates_per_horizon": [len(eligible_dates[h]) for h in HORIZONS],
                "eligible_day_counts_per_block_horizon": counts.tolist(),
            }
        )
        tensors.append((sums, counts))
    return records, tensors


def _bootstrap(a, names, methods):
    records, tensors = _blocks(a, names, methods)
    result = {
        "seed": 1729,
        "replicates": 2000,
        "block_days": 7,
        "deployment_count": len(records),
        "site_count": None,
        "site_count_reason": "Deployment groups are not proven independent sites.",
        "blocks": records,
        "valid_replicates": 0,
        "generated_replicates": 0,
        "not_run_replicates": 2000,
        "unsupported_replicates": 0,
        "sequence_sha256": None,
        "eligible_day_count_ranges_per_horizon": None,
        "algorithm": "resample_deployments_then_observed_nonoverlapping_calendar_blocks_with_replacement",
        "date_basis": "target_source_calendar_date_proxy_not_claimed_UTC",
        "percentiles": [2.5, 97.5],
        "percentile_method": "linear",
        "limitations": [
            "Few deployment groups limit generalization; deployments may share sites.",
            "Seven-day blocks assume local dependence within blocks and weaker dependence across blocks.",
            "Repeated blocks/dates retain multiplicity; eligible day counts can differ by horizon and replicate.",
            "Intervals are conditional on the reviewed issued rows, floor and calendar proxies.",
        ],
    }
    if any(methods[name]["metrics"]["primary_pinball_db"] is None for name in names):
        result["interval_status"] = "NOT_ASSESSABLE"
        result["reason"] = "required_daily_horizon_deployment_floor_support_incomplete"
        return result, None
    rng, sequence = np.random.default_rng(1729), hashlib.sha256()
    draws, count_ranges = [], []
    for _ in range(2000):
        sampled = rng.integers(0, len(records), size=len(records))
        sequence.update(sampled.astype("<i8").tobytes())
        scores, eligible = [], np.zeros(3, np.int64)
        supported = True
        for index in sampled:
            sums, counts = tensors[index]
            positions = rng.integers(0, len(counts), size=len(counts))
            sequence.update(positions.astype("<i8").tobytes())
            weights = np.bincount(positions, minlength=len(counts))
            n = weights @ counts
            eligible += n
            if (n == 0).any():
                supported = False
            else:
                scores.append(
                    ((weights @ sums.reshape(len(counts), -1)).reshape(len(names), 3) / n).mean(
                        axis=1
                    )
                )
        count_ranges.append(eligible)
        if supported:
            draws.append(np.mean(scores, axis=0))
    result.update(
        generated_replicates=2000,
        not_run_replicates=0,
        valid_replicates=len(draws),
        unsupported_replicates=2000 - len(draws),
        sequence_sha256=sequence.hexdigest(),
        eligible_day_count_ranges_per_horizon=np.stack(
            [np.min(count_ranges, axis=0), np.max(count_ranges, axis=0)], axis=1
        ).tolist(),
    )
    if len(draws) != 2000 or any(row["zero_span"] for row in records):
        result["interval_status"] = "NOT_ASSESSABLE"
        result["reason"] = "unsupported_resampled_horizon_or_zero_calendar_span"
        return result, None
    result["interval_status"] = "ASSESSABLE"
    return result, np.asarray(draws)


def compare_saved_predictions(manifest_path, review_path, output_path):
    """Compare exact reviewed NPZ support and write one new strict JSON report.

    The caller supplies genuine review provenance. Session-ID and binding checks
    cannot cryptographically authenticate the author of an externally supplied
    review document. Synthetic evidence has all the same admission checks.
    """
    manifest_path, review_path, output_path = (
        Path(v).resolve() for v in (manifest_path, review_path, output_path)
    )
    manifest, paths, snapshots, provenance = _admit(manifest_path, review_path, output_path)
    names = sorted(paths)
    loaded = {
        name: _load(snapshots[paths[name]], manifest["role"], manifest["evidence_kind"])
        for name in names
    }
    base = loaded[names[0]]
    for a in loaded.values():
        _common(base, a)
    support = _support(base)
    methods = {}
    for name, a in loaded.items():
        metrics = native_product.native_scores(
            a["predictions"],
            a["targets"],
            a["observed"],
            a["target_dates"],
            a["deployment"],
            minimum_daily_rows=18,
        )
        reconstructed = {
            (row["deployment"], row["horizon_intervals"], row["target_source_date"]): row["rows"]
            for row in metrics["daily_rows"]
        }
        if reconstructed != support or len(reconstructed) != len(metrics["daily_rows"]):
            raise ValueError(
                "Imported scorer daily support differs from independently reconstructed common support."
            )
        methods[name] = {
            "metrics": metrics,
            "forecast_variation": _variation(a["predictions"], a["deployment"]),
            "provenance": {
                key: _array_json(value) for key, value in a.items() if key not in REQUIRED
            },
        }
    bootstrap, draws = _bootstrap(base, names, methods)
    reference = manifest["reference"]
    ref = methods[reference]["metrics"]["primary_pinball_db"]
    comparisons = []
    for name in names:
        if name == reference:
            continue
        point = methods[name]["metrics"]["primary_pinball_db"]
        differences = (
            draws[:, names.index(name)] - draws[:, names.index(reference)]
            if draws is not None
            else None
        )
        comparisons.append(
            {
                "method": name,
                "reference": reference,
                "direction": "method_minus_reference",
                "point_difference_db": point - ref
                if point is not None and ref is not None
                else None,
                "ci95_db": np.percentile(differences, [2.5, 97.5], method="linear").tolist()
                if differences is not None
                else None,
                "status": bootstrap["interval_status"],
                "zero_resampling_variation": bool(np.ptp(differences) == 0)
                if differences is not None
                else None,
            }
        )
    result = {
        "kind": "native_saved_prediction_comparison_v1",
        "role": manifest["role"],
        "evidence_kind": manifest["evidence_kind"],
        "recipe": dict(RECIPE),
        "reference": reference,
        "methods": methods,
        "comparisons": comparisons,
        "bootstrap": bootstrap,
        "native_geometry": _geometry(base["query"]),
        "provenance": provenance,
        "common_support": {
            "canonical_order": "deployment_then_row_id",
            "issued_rows": len(base["row_id"]),
            "row_id": base["row_id"].tolist(),
            "deployment": base["deployment"].tolist(),
            "target_dates": base["target_dates"].tolist(),
            "observed": base["observed"].tolist(),
            "targets_observed_db": np.where(base["observed"], base["targets"], None).tolist(),
            "query": _array_json(base["query"]),
            "observed_rows_per_horizon": base["observed"].sum(axis=0).tolist(),
            "observed_fraction_per_horizon": base["observed"].mean(axis=0).tolist(),
            "eligible_daily_record_count": len(support),
        },
    }
    serialized = json.dumps(result, indent=2, allow_nan=False)
    with output_path.open("x", encoding="utf-8") as stream:
        stream.write(serialized + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    compare_saved_predictions(args.manifest, args.review, args.output)


if __name__ == "__main__":
    main()
