"""Run hard evaluation suite from recorded agent observations.

Usage:
  python scripts/run_hard_evals.py --write-sample hard_eval_input.json
  python scripts/run_hard_evals.py --input hard_eval_input.json
  python scripts/run_hard_evals.py --input hard_eval_input.json --output hard_eval_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.hard_eval_suite import HardEvalSuite, default_hard_cases, load_hard_eval_input


def write_sample(path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    cases = default_hard_cases()
    observations: dict[str, list[dict]] = {}
    for case in cases:
        observations[case.case_id] = [
            {
                "case_id": case.case_id,
                "variant_id": "baseline",
                "prompt": case.prompt,
                "response": "",
                "tool_calls": [],
                "changed_files": [],
                "evidence": [],
            }
        ]

    payload = {
        "cases": [asdict(case) for case in cases],
        "observations": observations,
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run(
    input_path: str,
    output_path: str | None,
    min_consistency: float,
    min_variants: int,
) -> None:
    cases, obs_by_case = load_hard_eval_input(input_path)
    suite = HardEvalSuite(
        min_consistency=min_consistency,
        min_variant_count=min_variants,
    )

    case_results = []
    for case in cases:
        observations = obs_by_case.get(case.case_id, [])
        case_results.append(suite.evaluate_case(case, observations))
    summary = suite.summary(case_results, cases)

    text = json.dumps(summary.to_dict(), indent=2)
    print(text)
    if output_path:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hard AI-resistance eval suite.")
    parser.add_argument("--input", help="Path to hard eval input JSON.")
    parser.add_argument("--output", help="Path to write summary JSON.")
    parser.add_argument(
        "--min-consistency",
        type=float,
        default=0.45,
        help="Minimum cross-variant consistency threshold (0-1).",
    )
    parser.add_argument(
        "--min-variants",
        type=int,
        default=3,
        help="Minimum variant observations required per case.",
    )
    parser.add_argument(
        "--write-sample",
        help="Write a sample hard-eval input JSON file to this path.",
    )
    args = parser.parse_args()

    if args.write_sample:
        write_sample(args.write_sample)
        print(f"Wrote sample hard eval input to {args.write_sample}")
        return

    if not args.input:
        raise SystemExit("Provide --input, or use --write-sample to bootstrap.")

    run(args.input, args.output, args.min_consistency, args.min_variants)


if __name__ == "__main__":
    main()
