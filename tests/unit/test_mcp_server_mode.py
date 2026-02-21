"""Unit tests for MCP server mode."""

from __future__ import annotations

from pathlib import Path

from src.mcp.server_mode import BrainmassMCPServer


def test_tools_exposed() -> None:
    server = BrainmassMCPServer(cwd=".")
    names = {t["name"] for t in server.list_tools()}
    assert names == {"Bash", "Read", "Write", "Edit", "LS", "GrepTool", "GlobTool", "Replace"}


def test_write_read_edit_replace(tmp_path: Path) -> None:
    server = BrainmassMCPServer(cwd=str(tmp_path))
    path = "a.txt"

    w = server.call_tool("Write", {"path": path, "content": "hello world"})
    assert not w.is_error

    r = server.call_tool("Read", {"path": path})
    assert "hello world" in r.content

    e = server.call_tool("Edit", {"path": path, "old": "hello", "new": "goodbye"})
    assert not e.is_error

    rep = server.call_tool("Replace", {"path": path, "old": "world", "new": "earth"})
    assert not rep.is_error

    r2 = server.call_tool("Read", {"path": path})
    assert "goodbye earth" in r2.content


def test_grep_and_glob(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir(parents=True, exist_ok=True)
    (tmp_path / "src" / "one.py").write_text("alpha\\nbeta\\n", encoding="utf-8")
    (tmp_path / "src" / "two.py").write_text("beta\\ngamma\\n", encoding="utf-8")
    server = BrainmassMCPServer(cwd=str(tmp_path))

    glob_res = server.call_tool("GlobTool", {"root": ".", "pattern": "src/*.py"})
    assert "src/one.py" in glob_res.content
    assert "src/two.py" in glob_res.content

    grep_res = server.call_tool("GrepTool", {"root": ".", "pattern": "beta"})
    assert "src/one.py" in grep_res.content
    assert "src/two.py" in grep_res.content


def test_glob_blocks_path_escape(tmp_path: Path) -> None:
    server = BrainmassMCPServer(cwd=str(tmp_path))
    result = server.call_tool("GlobTool", {"root": ".", "pattern": "../*"})
    assert result.is_error
    assert "escapes server working directory" in result.content


def test_permission_callback_blocks_tool(tmp_path: Path) -> None:
    def deny_bash(tool_name: str, args: dict) -> tuple[bool, str]:
        if tool_name == "Bash":
            return False, "policy"
        return True, "ok"

    server = BrainmassMCPServer(cwd=str(tmp_path), permission_callback=deny_bash)
    result = server.call_tool("Bash", {"command": "echo hello"})
    assert result.is_error
    assert "Permission denied" in result.content
