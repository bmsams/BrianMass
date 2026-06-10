"""Unit tests for the brainmass CLI entrypoint."""

from __future__ import annotations

import json

import pytest

from src.cli import build_orchestrator, build_parser, main


class TestParser:
    def test_request_positional(self):
        args = build_parser().parse_args(["do something"])
        assert args.request == "do something"
        assert args.production is False
        assert args.json is False

    def test_no_request_means_interactive(self):
        args = build_parser().parse_args([])
        assert args.request is None

    def test_flags(self):
        args = build_parser().parse_args([
            "task", "--production", "--json", "--session-id", "s1",
            "--cwd", "/tmp", "--window-size", "50000",
        ])
        assert args.production is True
        assert args.json is True
        assert args.session_id == "s1"
        assert args.cwd == "/tmp"
        assert args.window_size == 50_000


class TestBuildOrchestrator:
    def test_local_mode_by_default(self):
        orch = build_orchestrator(session_id="t1")
        assert orch._use_production_agent is False

    def test_production_flag_enables_strands_path(self):
        orch = build_orchestrator(session_id="t2", production=True)
        assert orch._use_production_agent is True


class TestMain:
    def test_single_request_exit_zero(self, capsys):
        exit_code = main(["review the codebase", "--session-id", "test-cli"])
        assert exit_code == 0
        out = capsys.readouterr().out
        assert out.strip()

    def test_json_output_is_valid(self, capsys):
        exit_code = main(["fix the tests", "--json", "--session-id", "test-cli-json"])
        assert exit_code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["request_id"]
        assert payload["response"]
        assert payload["model_tier"] in {"haiku", "sonnet", "opus"}
        assert "usage" in payload
        assert payload["usage"]["cost_usd"] >= 0

    def test_version_flag(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            main(["--version"])
        assert excinfo.value.code == 0
        assert "brainmass" in capsys.readouterr().out

    def test_repl_exits_on_eof(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: (_ for _ in ()).throw(EOFError))
        assert main([]) == 0

    def test_repl_processes_then_exits(self, monkeypatch, capsys):
        lines = iter(["check the build", "exit"])
        monkeypatch.setattr("builtins.input", lambda _: next(lines))
        assert main([]) == 0
        assert capsys.readouterr().out.strip()
