"""Run hard evals by executing adversarial variants through Orchestrator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.hard_eval_runtime import OrchestratorHardEvalRunner
from evals.hard_eval_suite import HardEvalCase, HardEvalSuite, default_hard_cases


def _load_cases(path: str | None) -> list[HardEvalCase]:
    if path is None:
        return default_hard_cases()

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    case_rows = raw.get("cases", raw)
    if not isinstance(case_rows, list):
        raise ValueError("Case file must contain a list or an object with a 'cases' list.")
    return [HardEvalCase.from_dict(row) for row in case_rows]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run live hard AI-resistance evals via Orchestrator.",
    )
    parser.add_argument("--cases", help="Optional path to JSON file with case definitions.")
    parser.add_argument(
        "--cwd",
        default=".",
        help="Working directory passed into orchestrator runner.",
    )
    parser.add_argument(
        "--mode",
        choices=["orchestrator_default", "policy_stub"],
        default="orchestrator_default",
        help="Execution mode: real orchestrator default callbacks or stub policy model.",
    )
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
    parser.add_argument("--output", help="Optional path to write summary JSON.")
    args = parser.parse_args()

    cases = _load_cases(args.cases)
    suite = HardEvalSuite(
        min_consistency=args.min_consistency,
        min_variant_count=args.min_variants,
    )
    runner = OrchestratorHardEvalRunner(cwd=args.cwd, mode=args.mode)
    summary = suite.run_with_runner(cases, runner.run_variant)

    text = json.dumps(summary.to_dict(), indent=2)
    print(text)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
