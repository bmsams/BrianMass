"""Monitor tasks.md and run QA checks on newly completed tasks.

Usage:
  python scripts/task_qa_monitor.py
  python scripts/task_qa_monitor.py --watch --interval-seconds 180
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
TASKS_PATH = ROOT / ".kiro" / "specs" / "claude-code-v3-enterprise" / "tasks.md"
STATE_PATH = ROOT / ".kiro" / "specs" / "claude-code-v3-enterprise" / ".qa_state.json"
REPORT_PATH = ROOT / ".kiro" / "specs" / "claude-code-v3-enterprise" / "qa_findings.md"

TASK_RE = re.compile(
    r"^\s*-\s+\[(?P<status>[ xX~])\](?P<optional>\*)?\s+(?P<id>\d+(?:\.\d+)?)\s+(?P<title>.+)$"
)


@dataclass
class TaskRow:
    task_id: str
    title: str
    status: str
    optional: bool

    @property
    def is_completed(self) -> bool:
        return self.status in {"x", "X"}


TASK_QA_MAP: dict[str, dict[str, list[str]]] = {
    "1.4": {
        "requirements": ["19.1"],
        "sources": ["src/config/config_loader.py"],
        "tests": ["tests/unit/test_config_loader.py"],
    },
    "2.2": {
        "requirements": ["2.1"],
        "sources": ["src/context/triage.py"],
        "tests": ["tests/property/test_context_properties.py"],
    },
    "2.4": {
        "requirements": ["2.4", "2.5", "2.7", "2.8", "2.11"],
        "sources": ["src/context/context_manager.py"],
        "tests": ["tests/property/test_context_properties.py"],
    },
    "3.6": {
        "requirements": ["3.4", "3.7", "3.8"],
        "sources": ["src/hooks/hook_engine.py"],
        "tests": ["tests/property/test_hook_properties.py"],
    },
    "5.2": {
        "requirements": ["4.2", "4.3", "4.5", "4.6"],
        "sources": ["src/cost/cost_governor.py"],
        "tests": ["tests/property/test_cost_properties.py"],
    },
    "6.3": {
        "requirements": ["12.1", "12.2"],
        "sources": ["src/orchestrator/effort_controller.py"],
        "tests": ["tests/property/test_effort_properties.py"],
    },
    "9.4": {
        "requirements": ["6.4"],
        "sources": ["src/agents/file_lock.py"],
        "tests": ["tests/property/test_team_properties.py"],
    },
    "10.4": {
        "requirements": ["7.1", "7.2"],
        "sources": ["src/agents/loop_runner.py", "src/agents/context_file.py"],
        "tests": ["tests/property/test_loop_properties.py"],
    },
    "12.2": {
        "requirements": ["11.1"],
        "sources": ["src/cache/cache_manager.py"],
        "tests": ["tests/property/test_cache_properties.py"],
    },
    "16.1": {
        "requirements": ["9.1", "9.2", "9.3", "9.4", "9.5"],
        "sources": ["src/plugins/plugin_registry.py"],
        "tests": ["tests/unit/test_plugin_registry.py"],
    },
}


def parse_tasks(tasks_path: Path) -> list[TaskRow]:
    rows: list[TaskRow] = []
    for line in tasks_path.read_text(encoding="utf-8").splitlines():
        m = TASK_RE.match(line)
        if not m:
            continue
        rows.append(
            TaskRow(
                task_id=m.group("id"),
                title=m.group("title").strip(),
                status=m.group("status"),
                optional=bool(m.group("optional")),
            )
        )
    return rows


def load_state(state_path: Path) -> dict:
    if not state_path.exists():
        return {"processed_completed": []}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"processed_completed": []}


def save_state(state_path: Path, state: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def run_pytest(test_paths: list[str]) -> tuple[bool, str]:
    cmd = [sys.executable, "-m", "pytest", "-q", *test_paths]
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    output = (result.stdout or "") + (result.stderr or "")
    summary = "\n".join(output.strip().splitlines()[-12:])
    return result.returncode == 0, summary


def qa_task(task: TaskRow) -> dict:
    mapping = TASK_QA_MAP.get(task.task_id)
    report = {
        "task_id": task.task_id,
        "title": task.title,
        "requirements": [],
        "checks": [],
        "findings": [],
    }

    if mapping is None:
        report["findings"].append(
            "No automated QA mapping exists for this task yet. Add it to TASK_QA_MAP."
        )
        return report

    report["requirements"] = mapping.get("requirements", [])

    for source in mapping.get("sources", []):
        exists = (ROOT / source).exists()
        report["checks"].append(
            f"{'PASS' if exists else 'FAIL'} source exists: `{source}`"
        )
        if not exists:
            report["findings"].append(f"Missing expected source file: `{source}`")

    tests = mapping.get("tests", [])
    if tests:
        ok, summary = run_pytest(tests)
        report["checks"].append(
            f"{'PASS' if ok else 'FAIL'} tests: `{' '.join(tests)}`"
        )
        if not ok:
            report["findings"].append("Targeted tests failed.")
            report["findings"].append(f"Pytest summary:\n```\n{summary}\n```")
    else:
        report["checks"].append("SKIP tests: no tests configured")

    return report


def append_report(report_path: Path, run_data: dict) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = run_data["timestamp"]
    new_tasks = run_data["new_completed"]
    qa_results = run_data["qa_results"]

    lines: list[str] = []
    lines.append(f"## QA Run {timestamp}")
    lines.append("")
    if not new_tasks:
        lines.append("- No newly completed tasks detected.")
        lines.append("")
    else:
        lines.append(f"- Newly completed tasks: {', '.join(new_tasks)}")
        lines.append("")
        for item in qa_results:
            lines.append(f"### Task {item['task_id']}: {item['title']}")
            if item["requirements"]:
                lines.append(f"- Requirements: {', '.join(item['requirements'])}")
            else:
                lines.append("- Requirements: not mapped")
            lines.append("- Checks:")
            for check in item["checks"]:
                lines.append(f"  - {check}")
            if item["findings"]:
                lines.append("- Findings to fix:")
                for finding in item["findings"]:
                    lines.append(f"  - {finding}")
            else:
                lines.append("- Findings to fix: none")
            lines.append("")

    previous = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    report_path.write_text(
        previous + ("\n" if previous and not previous.endswith("\n") else "") + "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def run_once(verbose: bool = True, recheck_all: bool = False) -> None:
    rows = parse_tasks(TASKS_PATH)
    completed_ids = [row.task_id for row in rows if row.is_completed]
    state = load_state(STATE_PATH)
    processed = set(state.get("processed_completed", []))
    if recheck_all:
        new_completed = list(completed_ids)
    else:
        new_completed = [tid for tid in completed_ids if tid not in processed]

    qa_results = []
    for row in rows:
        if row.task_id in new_completed:
            qa_results.append(qa_task(row))

    run_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "new_completed": new_completed,
        "qa_results": qa_results,
    }
    append_report(REPORT_PATH, run_data)

    state["processed_completed"] = sorted(set(processed).union(completed_ids))
    state["last_run"] = run_data["timestamp"]
    save_state(STATE_PATH, state)

    if verbose:
        print(f"QA monitor run complete at {run_data['timestamp']}")
        if new_completed:
            print("New completed tasks:", ", ".join(new_completed))
        else:
            print("No newly completed tasks.")
        print(f"Report: {REPORT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor tasks.md for QA checks.")
    parser.add_argument("--watch", action="store_true", help="Run continuously.")
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=180,
        help="Polling interval in seconds when --watch is set.",
    )
    parser.add_argument(
        "--recheck-all",
        action="store_true",
        help="Run QA against all currently completed tasks.",
    )
    args = parser.parse_args()

    if not args.watch:
        run_once(verbose=True, recheck_all=args.recheck_all)
        return

    while True:
        try:
            run_once(verbose=True, recheck_all=args.recheck_all)
        except Exception as exc:  # noqa: BLE001
            print(f"QA monitor error: {exc}", file=sys.stderr)
        time.sleep(max(args.interval_seconds, 10))


if __name__ == "__main__":
    main()
