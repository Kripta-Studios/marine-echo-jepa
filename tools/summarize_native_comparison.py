"""Print bounded metadata and scores from a completed comparison JSON."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    summary = {key: result[key] for key in ("role", "reference", "native_geometry", "comparisons")}
    summary["bootstrap"] = {key: value for key, value in result["bootstrap"].items()
                            if key != "blocks"}
    summary["methods"] = {
        name: {"primary_pinball_db": value["metrics"]["primary_pinball_db"],
               "scored_per_horizon": value["metrics"]["scored_per_horizon"],
               "eligible_source_dates_per_horizon": value["metrics"]["eligible_source_dates_per_horizon"],
               "constant_across_issuance": value["forecast_variation"]["constant_across_issuance"],
               "median_std_db_per_horizon": value["forecast_variation"]["median_std_db_per_horizon"]}
        for name, value in result["methods"].items()
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
