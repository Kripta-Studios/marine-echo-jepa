"""Run separately admitted suffix forecasts under one owned resource context."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "review", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    from marine_echo.evaluation import native_prefix_suffix_assessment_v1 as assessment
    from marine_echo.training.native_desktop_runtime_v3 import Resources

    admitted = assessment.admit(args.manifest, args.review, args.output)
    with Resources(admitted.manifest["device"], args.output) as resources:
        result = assessment.execute_assessment(args.manifest, args.review, args.output)
        resources.check()
    print(json.dumps({"status": result["status"], "role": "adapted_suffix", "fitting": False}))


if __name__ == "__main__":
    main()
