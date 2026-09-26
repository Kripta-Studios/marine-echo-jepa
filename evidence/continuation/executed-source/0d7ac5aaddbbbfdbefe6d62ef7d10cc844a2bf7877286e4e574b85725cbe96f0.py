"""Execute prospective slant-path bounds using the preserved regional depth envelope."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from continuation_records import write_report
from echopype.utils import uwa

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started = time.monotonic()
    contract = ROOT / "evidence/continuation/slant_environment_contract.json"
    expected = (
        (ROOT / "evidence/continuation/slant_environment_contract.sha256").read_text().strip()
    )
    if sha(contract) != expected:
        raise ValueError("Slant contract differs from prospective digest")
    source = ROOT / "evidence/continuation/depth_environment_sensitivity.json"
    if sha(source) != json.loads(contract.read_text())["depth_report_sha256"]:
        raise ValueError("Reviewed depth report changed")
    report = json.loads(source.read_text())
    nominal = report["nominal"]
    c0 = nominal["sound_speed"]
    alpha0 = uwa.calc_absorption(
        frequency=np.array([38000, 125000, 200000, 455000]), **nominal, formula_source="AZFP"
    )
    rows = report["depth_bin_coverage"]
    speed = np.array([row["c_min_max_m_s"] for row in rows])
    alpha = [np.array([row[key] for row in rows]) for key in ("alpha_min_db_m", "alpha_max_db_m")]
    depths = np.arange(0.0, 132.0, 2.0)
    nominal_range = np.arange(10.0, 100.01, 0.5)
    results, zero_errors = [], []
    for angle in range(46):
        cosine = np.cos(np.deg2rad(angle))
        ray_edges = depths / cosine
        widths = np.diff(ray_edges)
        times = [np.concatenate([[0.0], np.cumsum(widths / speed[:, i])]) for i in range(2)]
        integrated_alpha = [
            np.vstack([np.zeros(4), np.cumsum(widths[:, None] * a, axis=0)]) for a in alpha
        ]
        for depth in np.arange(0.0, 10.01, 0.25):
            origin = depth / cosine
            endpoints, attenuation = [], []
            for i in range(2):
                base = np.interp(origin, ray_edges, times[i])
                query = base + nominal_range / c0
                if query.max() > times[i][-1]:
                    raise ValueError("Ray leaves reviewed source depth interval")
                endpoints.append(np.interp(query, times[i], ray_edges))
                attenuation.append(
                    np.stack(
                        [
                            np.interp(endpoints[i], ray_edges, integrated_alpha[i][:, f])
                            - np.interp(origin, ray_edges, integrated_alpha[i][:, f])
                            for f in range(4)
                        ],
                        axis=1,
                    )
                )
            actual_range = [endpoint - origin for endpoint in endpoints]
            local_min, local_max = [], []
            for lower, upper in zip(*endpoints):
                left = max(0, int(np.searchsorted(ray_edges, lower, side="left")) - 1)
                right = min(len(speed), int(np.searchsorted(ray_edges, upper, side="right")))
                local_min.append(speed[left:right, 0].min())
                local_max.append(speed[left:right, 1].max())
            low = (
                20 * np.log10(actual_range[0] / nominal_range)[:, None]
                - 10 * np.log10(np.array(local_max) / c0)[:, None]
                + 2 * attenuation[0]
                - 2 * nominal_range[:, None] * alpha0
            )
            high = (
                20 * np.log10(actual_range[1] / nominal_range)[:, None]
                - 10 * np.log10(np.array(local_min) / c0)[:, None]
                + 2 * attenuation[1]
                - 2 * nominal_range[:, None] * alpha0
            )
            maximum = np.maximum(np.abs(low), np.abs(high)).max(axis=0)
            displacement = max(
                float(np.max(np.abs(value - nominal_range))) for value in actual_range
            )
            if angle == 0:
                prior = next(
                    row for row in report["scenarios"] if row["sampled_transducer_depth_m"] == depth
                )
                zero_errors.append(float(np.max(np.abs(maximum - prior["max_abs_Sv_delta_db"]))))
                if not np.allclose(
                    maximum, prior["max_abs_Sv_delta_db"], rtol=1e-10, atol=1e-10
                ) or not np.isclose(
                    displacement, prior["max_abs_range_displacement_m"], rtol=1e-10, atol=1e-10
                ):
                    raise ValueError(
                        "Vector prefix integrals disagree with independent vertical segment integrator"
                    )
            results.append(
                {
                    "start_depth_m": float(depth),
                    "ray_angle_degrees": angle,
                    "max_abs_Sv_delta_db": maximum.tolist(),
                    "max_range_displacement_m": displacement,
                    "primary_limits_passed": bool(maximum[0] <= 1 and displacement <= 2),
                }
            )
    result = {
        "status": "SLANT_SENSITIVITY_REQUIRES_REVIEW",
        "benchmark_eligible": False,
        "contract_sha256": sha(contract),
        "source_report_sha256": sha(source),
        "code_sha256": sha(Path(__file__)),
        "all_sampled_scenarios_passed": all(row["primary_limits_passed"] for row in results),
        "scenario_count": len(results),
        "max_abs_Sv_delta_db": np.max(
            [row["max_abs_Sv_delta_db"] for row in results], axis=0
        ).tolist(),
        "max_range_displacement_m": max(row["max_range_displacement_m"] for row in results),
        "independent_vertical_integrator_max_difference_db": max(zero_errors),
        "scenarios": results,
        "elapsed_seconds": time.monotonic() - started,
        "limitations": json.loads(contract.read_text())["limits"],
    }
    write_report(ROOT / "evidence/continuation/slant_environment_sensitivity.json", result)
    print(json.dumps({key: value for key, value in result.items() if key != "scenarios"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
