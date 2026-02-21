"""Brainmass MCP server mode.

Implements MCP stdio server behavior with required tools:
Bash, Read, Write, Edit, LS, GrepTool, GlobTool, Replace.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

PermissionCallback = Callable[[str, dict], tuple[bool, str]]


@dataclass
class ToolResult:
    is_error: bool
    content: str

    def to_mcp(self) -> dict:
        return {
            "isError": self.is_error,
            "content": [
                {"type": "text", "text": self.content},
            ],
        }


class BrainmassMCPServer:
    """A minimal MCP stdio server with filesystem and shell tools."""

    def __init__(
        self,
        cwd: str = ".",
        permission_callback: PermissionCallback | None = None,
    ) -> None:
        self.cwd = Path(cwd).resolve()
        self._permission_callback = permission_callback
        self._tool_handlers: dict[str, Callable[[dict], ToolResult]] = {
            "Bash": self._tool_bash,
            "Read": self._tool_read,
            "Write": self._tool_write,
            "Edit": self._tool_edit,
            "LS": self._tool_ls,
            "GrepTool": self._tool_grep,
            "GlobTool": self._tool_glob,
            "Replace": self._tool_replace,
        }

    def capabilities(self) -> dict:
        return {
            "server": "brainmass-mcp",
            "protocol": "stdio",
            "tools": list(self._tool_handlers.keys()),
            "no_mcp_passthrough": True,
            "fresh_instance_per_client": True,
        }

    def list_tools(self) -> list[dict]:
        return [
            {"name": name, "description": f"{name} tool", "inputSchema": {"type": "object"}}
            for name in self._tool_handlers
        ]

    def call_tool(self, name: str, arguments: dict) -> ToolResult:
        if name not in self._tool_handlers:
            return ToolResult(True, f"Unknown tool: {name}")
        allowed, reason = self._check_permission(name, arguments)
        if not allowed:
            return ToolResult(True, f"Permission denied for {name}: {reason}")
        try:
            return self._tool_handlers[name](arguments)
        except Exception as exc:
            return ToolResult(True, f"{name} failed: {exc}")

    def _check_permission(self, tool_name: str, args: dict) -> tuple[bool, str]:
        if self._permission_callback is None:
            return True, "no policy callback configured"
        return self._permission_callback(tool_name, args)

    def _safe_path(self, raw: str) -> Path:
        target = (self.cwd / raw).resolve() if not Path(raw).is_absolute() else Path(raw).resolve()
        if not self._is_within_cwd(target):
            raise ValueError("Path escapes server working directory")
        return target

    def _is_within_cwd(self, path: Path) -> bool:
        return self.cwd == path or self.cwd in path.parents

    def _tool_bash(self, arguments: dict) -> ToolResult:
        command = str(arguments.get("command", "")).strip()
        if not command:
            return ToolResult(True, "command is required")
        proc = subprocess.run(
            command,
            cwd=str(self.cwd),
            shell=True,
            capture_output=True,
            text=True,
            timeout=int(arguments.get("timeout", 60)),
        )
        output = proc.stdout or ""
        err = proc.stderr or ""
        text = output + (("\n" + err) if err else "")
        return ToolResult(proc.returncode != 0, text.strip())

    def _tool_read(self, arguments: dict) -> ToolResult:
        path = self._safe_path(str(arguments.get("path", "")))
        if not path.exists():
            return ToolResult(True, f"File not found: {path}")
        return ToolResult(False, path.read_text(encoding="utf-8"))

    def _tool_write(self, arguments: dict) -> ToolResult:
        path = self._safe_path(str(arguments.get("path", "")))
        content = str(arguments.get("content", ""))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return ToolResult(False, f"Wrote {len(content)} chars to {path}")

    def _tool_edit(self, arguments: dict) -> ToolResult:
        path = self._safe_path(str(arguments.get("path", "")))
        old = str(arguments.get("old", ""))
        new = str(arguments.get("new", ""))
        if not path.exists():
            return ToolResult(True, f"File not found: {path}")
        text = path.read_text(encoding="utf-8")
        if old not in text:
            return ToolResult(True, "old text not found")
        updated = text.replace(old, new, 1)
        path.write_text(updated, encoding="utf-8")
        return ToolResult(False, "Applied one edit")

    def _tool_replace(self, arguments: dict) -> ToolResult:
        path = self._safe_path(str(arguments.get("path", "")))
        old = str(arguments.get("old", ""))
        new = str(arguments.get("new", ""))
        if not path.exists():
            return ToolResult(True, f"File not found: {path}")
        text = path.read_text(encoding="utf-8")
        count = text.count(old) if old else 0
        if count == 0:
            return ToolResult(True, "old text not found")
        path.write_text(text.replace(old, new), encoding="utf-8")
        return ToolResult(False, f"Replaced {count} occurrence(s)")

    def _tool_ls(self, arguments: dict) -> ToolResult:
        path = self._safe_path(str(arguments.get("path", ".")))
        if not path.exists():
            return ToolResult(True, f"Path not found: {path}")
        entries = sorted(p.name for p in path.iterdir())
        return ToolResult(False, "\n".join(entries))

    def _tool_glob(self, arguments: dict) -> ToolResult:
        pattern = str(arguments.get("pattern", "**/*"))
        root = self._safe_path(str(arguments.get("root", ".")))
        matches: list[str] = []
        for match in root.glob(pattern):
            resolved = match.resolve()
            if not self._is_within_cwd(resolved):
                raise ValueError("Glob pattern escapes server working directory")
            matches.append(resolved.relative_to(self.cwd).as_posix())
        return ToolResult(False, "\n".join(sorted(matches)))

    def _tool_grep(self, arguments: dict) -> ToolResult:
        pattern = str(arguments.get("pattern", ""))
        root = self._safe_path(str(arguments.get("root", ".")))
        if not pattern:
            return ToolResult(True, "pattern is required")
        hits: list[str] = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except Exception:
                continue
            for idx, line in enumerate(text.splitlines(), start=1):
                if pattern in line:
                    rel = path.relative_to(self.cwd).as_posix()
                    hits.append(f"{rel}:{idx}:{line}")
        return ToolResult(False, "\n".join(hits))


def _jsonrpc_result(request_id, result) -> str:
    return json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result}, ensure_ascii=True)


def _jsonrpc_error(request_id, message: str, code: int = -32000) -> str:
    return json.dumps(
        {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}},
        ensure_ascii=True,
    )


def run_stdio_server(cwd: str = ".", permission_callback: PermissionCallback | None = None) -> None:
    """Run MCP server over stdio transport.

    A fresh server instance is created for each process invocation.
    """
    import sys

    server = BrainmassMCPServer(cwd=cwd, permission_callback=permission_callback)
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            request_id = req.get("id")
            method = req.get("method")
            params = req.get("params", {}) or {}

            if method == "initialize":
                print(_jsonrpc_result(request_id, {"capabilities": server.capabilities()}), flush=True)
            elif method == "capabilities/get":
                print(_jsonrpc_result(request_id, server.capabilities()), flush=True)
            elif method == "tools/list":
                print(_jsonrpc_result(request_id, {"tools": server.list_tools()}), flush=True)
            elif method == "tools/call":
                name = params.get("name")
                arguments = params.get("arguments", {}) or {}
                result = server.call_tool(str(name), arguments)
                print(_jsonrpc_result(request_id, result.to_mcp()), flush=True)
            elif method == "shutdown":
                print(_jsonrpc_result(request_id, {"ok": True}), flush=True)
                break
            else:
                print(_jsonrpc_error(request_id, f"Unknown method: {method}", code=-32601), flush=True)
        except Exception as exc:
            print(_jsonrpc_error(None, str(exc)), flush=True)
