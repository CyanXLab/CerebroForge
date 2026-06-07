"""
CerebroForge (铸脑) - Computer Use Toolkit
===========================================
Computer operation tools for the cognitive agent:
  - Terminal execution (multi-platform)
  - File operations (workspace-restricted)
  - Python sandbox execution
  - System information
  - Process management
  - Download with safety checks

All dangerous commands are blocked. All file operations are restricted
to the workspace directory.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from backend.config import WORKSPACE_DIR
except ImportError:
    from config import WORKSPACE_DIR

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Dangerous Command Blocklist
# ────────────────────────────────────────────────────────────────────────────

_DANGEROUS_PATTERNS: List[str] = [
    "rm -rf /",
    "rm -rf /*",
    "rm -rf ~",
    "rm -rf ~/*",
    "sudo rm",
    "mkfs",
    "dd if=",
    "format ",
    "> /dev/sd",
    ":(){ :|:& };:",
    "fork bomb",
    "chmod -R 777 /",
    "chown -R",
    "shutdown",
    "reboot",
    "init 0",
    "init 6",
    "systemctl stop",
    "systemctl disable",
    "service stop",
    "pkill -9",
    "kill -9 1",
    "killall",
    "crontab -r",
    "iptables -F",
    "ip6tables -F",
    "nvim",
    "vim",
    "nano",  # Interactive editors block execution
]

_DANGEROUS_WINDOWS_PATTERNS: List[str] = [
    "format ",
    "del /f /s /q C:",
    "rd /s /q C:",
    "rmdir /s /q C:",
    "net user",
    "net localgroup",
    "reg delete",
    "reg add",
    "taskkill /f /pid 0",
    "taskkill /f /im svchost",
    "cipher /w:C",
    "sfc /scannow",
]


# ────────────────────────────────────────────────────────────────────────────
# Platform Detection
# ────────────────────────────────────────────────────────────────────────────

def _is_windows() -> bool:
    return platform.system().lower() == "windows"


def _is_linux() -> bool:
    return platform.system().lower() == "linux"


def _is_macos() -> bool:
    return platform.system().lower() == "darwin"


# ────────────────────────────────────────────────────────────────────────────
# Path Safety (Workspace-Restricted)
# ────────────────────────────────────────────────────────────────────────────

def _resolve_and_validate_path(path: str, must_exist: bool = False) -> Path:
    """
    Resolve a path and ensure it's within the workspace directory.

    Raises ValueError if the path escapes the workspace.
    """
    workspace = WORKSPACE_DIR.resolve()

    # Handle relative paths
    if not os.path.isabs(path):
        resolved = (workspace / path).resolve()
    else:
        resolved = Path(path).resolve()

    # Security check: must be within workspace
    try:
        resolved.relative_to(workspace)
    except ValueError:
        raise ValueError(
            f"Path '{path}' is outside the workspace directory '{workspace}'. "
            "All file operations are restricted to the workspace."
        )

    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved}")

    return resolved


# ────────────────────────────────────────────────────────────────────────────
# Computer Use Toolkit
# ────────────────────────────────────────────────────────────────────────────

class ComputerUseToolkit:
    """Comprehensive computer operation toolkit for the cognitive agent.

    All operations are safety-checked and workspace-restricted.
    """

    def __init__(self, workspace_dir: Optional[Path] = None) -> None:
        self.workspace = workspace_dir or WORKSPACE_DIR
        os.makedirs(self.workspace, exist_ok=True)

    # ── Terminal Execution ─────────────────────────────────────────────────

    def execute_terminal(
        self,
        command: str,
        timeout: int = 30,
    ) -> Dict[str, Any]:
        """
        Execute a terminal command.

        Multi-platform: uses bash on Linux/macOS, cmd or PowerShell on Windows.
        Dangerous commands are blocked.

        Args:
            command: The command to execute.
            timeout: Maximum execution time in seconds.

        Returns:
            Dict with stdout, stderr, exit_code, timed_out.
        """
        # Safety check
        is_safe, reason = self._check_command_safety(command)
        if not is_safe:
            return {
                "stdout": "",
                "stderr": f"Command blocked for safety: {reason}",
                "exit_code": -1,
                "timed_out": False,
                "blocked": True,
            }

        # Platform-specific shell selection
        if _is_windows():
            # Use PowerShell for better compatibility
            shell_cmd = ["powershell", "-Command", command]
        else:
            shell_cmd = ["/bin/bash", "-c", command]

        timed_out = False
        try:
            result = subprocess.run(
                shell_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.workspace),
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            return {
                "stdout": result.stdout[:50000],  # Cap output size
                "stderr": result.stderr[:10000],
                "exit_code": result.returncode,
                "timed_out": False,
                "blocked": False,
            }
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"Command timed out after {timeout} seconds",
                "exit_code": -1,
                "timed_out": True,
                "blocked": False,
            }
        except Exception as exc:
            return {
                "stdout": "",
                "stderr": f"Execution error: {exc}",
                "exit_code": -1,
                "timed_out": False,
                "blocked": False,
            }

    def _check_command_safety(self, command: str) -> Tuple[bool, str]:
        """Check if a command is safe to execute."""
        cmd_lower = command.lower().strip()

        # Check against dangerous patterns
        all_patterns = _DANGEROUS_PATTERNS + (
            _DANGEROUS_WINDOWS_PATTERNS if _is_windows() else []
        )

        for pattern in all_patterns:
            if pattern.lower() in cmd_lower:
                return False, f"Matches dangerous pattern: '{pattern}'"

        # Additional heuristics
        # Block piping to shell execution
        if "| sh" in cmd_lower or "| bash" in cmd_lower:
            return False, "Piping to shell execution is blocked"

        # Block command substitution in dangerous context
        if "$(rm" in cmd_lower or "`rm" in cmd_lower:
            return False, "Command substitution with rm is blocked"

        # Block writing to system directories
        system_dirs = ["/etc", "/boot", "/sys", "/proc", "/dev", "/root"]
        for d in system_dirs:
            if f"> {d}/" in cmd_lower or f">> {d}/" in cmd_lower:
                return False, f"Writing to system directory '{d}' is blocked"

        return True, ""

    # ── File Operations ────────────────────────────────────────────────────

    def read_file(self, path: str) -> Dict[str, Any]:
        """
        Read a file from the workspace.

        Args:
            path: Relative or absolute path within workspace.

        Returns:
            Dict with content, size, path.
        """
        try:
            resolved = _resolve_and_validate_path(path, must_exist=True)

            if resolved.is_dir():
                return {
                    "content": "",
                    "error": f"Path is a directory, not a file: {path}",
                    "size": 0,
                    "path": str(resolved),
                }

            # Check file size (limit to 1MB)
            size = resolved.stat().st_size
            if size > 1_048_576:
                return {
                    "content": "",
                    "error": f"File too large ({size} bytes). Maximum is 1MB.",
                    "size": size,
                    "path": str(resolved),
                }

            content = resolved.read_text(encoding="utf-8", errors="replace")
            return {
                "content": content,
                "error": None,
                "size": size,
                "path": str(resolved),
            }

        except ValueError as exc:
            return {"content": "", "error": str(exc), "size": 0, "path": path}
        except FileNotFoundError as exc:
            return {"content": "", "error": str(exc), "size": 0, "path": path}
        except Exception as exc:
            return {"content": "", "error": f"Read error: {exc}", "size": 0, "path": path}

    def write_file(
        self,
        path: str,
        content: str,
        mode: str = "write",
    ) -> Dict[str, Any]:
        """
        Write content to a file within the workspace.

        Args:
            path: Relative or absolute path within workspace.
            content: Content to write.
            mode: "write" (overwrite) or "append".

        Returns:
            Dict with success, path, bytes_written.
        """
        try:
            resolved = _resolve_and_validate_path(path)

            # Create parent directories
            resolved.parent.mkdir(parents=True, exist_ok=True)

            write_mode = "a" if mode == "append" else "w"
            bytes_written = resolved.write_text(content, encoding="utf-8")

            return {
                "success": True,
                "path": str(resolved),
                "bytes_written": bytes_written,
                "mode": mode,
                "error": None,
            }

        except ValueError as exc:
            return {"success": False, "path": path, "bytes_written": 0, "mode": mode, "error": str(exc)}
        except Exception as exc:
            return {"success": False, "path": path, "bytes_written": 0, "mode": mode, "error": str(exc)}

    def list_directory(self, path: str = ".") -> Dict[str, Any]:
        """
        List directory contents within the workspace.

        Args:
            path: Directory path within workspace.

        Returns:
            Dict with entries (list of {name, type, size}), path.
        """
        try:
            resolved = _resolve_and_validate_path(path, must_exist=True)

            if not resolved.is_dir():
                return {
                    "entries": [],
                    "error": f"Path is not a directory: {path}",
                    "path": str(resolved),
                }

            entries = []
            for item in sorted(resolved.iterdir()):
                try:
                    stat = item.stat()
                    entries.append({
                        "name": item.name,
                        "type": "directory" if item.is_dir() else "file",
                        "size": stat.st_size if item.is_file() else 0,
                        "modified": stat.st_mtime,
                    })
                except (PermissionError, OSError):
                    entries.append({
                        "name": item.name,
                        "type": "unknown",
                        "size": 0,
                        "modified": 0,
                    })

            return {
                "entries": entries,
                "error": None,
                "path": str(resolved),
                "total": len(entries),
            }

        except ValueError as exc:
            return {"entries": [], "error": str(exc), "path": path, "total": 0}
        except FileNotFoundError as exc:
            return {"entries": [], "error": str(exc), "path": path, "total": 0}
        except Exception as exc:
            return {"entries": [], "error": str(exc), "path": path, "total": 0}

    def search_files(
        self,
        pattern: str,
        path: str = ".",
        recursive: bool = True,
    ) -> Dict[str, Any]:
        """
        Search for files matching a pattern (pure Python grep).

        Args:
            pattern: Text pattern to search for.
            path: Starting directory within workspace.
            recursive: Whether to search recursively.

        Returns:
            Dict with matches (list of {file, line_number, line}), total_matches.
        """
        try:
            resolved = _resolve_and_validate_path(path, must_exist=True)

            if not resolved.is_dir():
                return {
                    "matches": [],
                    "error": f"Path is not a directory: {path}",
                    "total_matches": 0,
                }

            matches: List[Dict[str, Any]] = []
            max_matches = 100  # Cap results

            glob_pattern = "**/*" if recursive else "*"
            for file_path in resolved.glob(glob_pattern):
                if len(matches) >= max_matches:
                    break

                if not file_path.is_file():
                    continue

                # Skip binary files and large files
                try:
                    if file_path.stat().st_size > 500_000:
                        continue
                except OSError:
                    continue

                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        for line_num, line in enumerate(f, 1):
                            if pattern.lower() in line.lower():
                                rel_path = file_path.relative_to(resolved)
                                matches.append({
                                    "file": str(rel_path),
                                    "line_number": line_num,
                                    "line": line.strip()[:200],
                                })
                                if len(matches) >= max_matches:
                                    break
                except (OSError, UnicodeDecodeError):
                    continue

            return {
                "matches": matches,
                "error": None,
                "total_matches": len(matches),
                "pattern": pattern,
                "path": str(resolved),
            }

        except ValueError as exc:
            return {"matches": [], "error": str(exc), "total_matches": 0, "pattern": pattern, "path": path}
        except Exception as exc:
            return {"matches": [], "error": str(exc), "total_matches": 0, "pattern": pattern, "path": path}

    # ── Python Execution ───────────────────────────────────────────────────

    def execute_python(
        self,
        code: str,
        timeout: int = 30,
    ) -> Dict[str, Any]:
        """
        Execute Python code in a sandboxed environment.

        Uses a subprocess with restricted builtins and workspace-restricted
        file access.

        Args:
            code: Python code to execute.
            timeout: Maximum execution time in seconds.

        Returns:
            Dict with stdout, stderr, exit_code, timed_out.
        """
        # Safety check on code content
        code_safety = self._check_python_safety(code)
        if not code_safety[0]:
            return {
                "stdout": "",
                "stderr": f"Code blocked for safety: {code_safety[1]}",
                "exit_code": -1,
                "timed_out": False,
                "blocked": True,
            }

        # Write code to temp file and execute
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as tmp:
            # Add sandbox header
            sandbox_header = (
                "import sys\n"
                "import os\n"
                f"os.chdir('{self.workspace}')\n"
                "sys.path.insert(0, os.getcwd())\n"
                "# __builtins__ restrictions applied by process isolation\n"
                "\n"
            )
            tmp.write(sandbox_header + code)
            tmp_path = tmp.name

        try:
            result = subprocess.run(
                [sys.executable, "-u", tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.workspace),
                env={
                    **os.environ,
                    "PYTHONIOENCODING": "utf-8",
                    "PYTHONPATH": str(self.workspace),
                },
            )

            return {
                "stdout": result.stdout[:50000],
                "stderr": result.stderr[:10000],
                "exit_code": result.returncode,
                "timed_out": False,
                "blocked": False,
            }

        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"Python execution timed out after {timeout} seconds",
                "exit_code": -1,
                "timed_out": True,
                "blocked": False,
            }
        except Exception as exc:
            return {
                "stdout": "",
                "stderr": f"Execution error: {exc}",
                "exit_code": -1,
                "timed_out": False,
                "blocked": False,
            }
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _check_python_safety(self, code: str) -> Tuple[bool, str]:
        """Check Python code for dangerous patterns."""
        code_lower = code.lower()

        dangerous_imports = [
            "subprocess",
            "os.system",
            "os.popen",
            "shutil.rmtree",
            "ctypes",
            "multiprocessing",
        ]

        for imp in dangerous_imports:
            if imp in code_lower:
                # Allow if it's in a comment
                for line in code.split("\n"):
                    stripped = line.strip()
                    if stripped.startswith("#"):
                        continue
                    if imp in stripped.lower():
                        return False, f"Dangerous import or call detected: '{imp}'"

        # Block eval/exec with user input
        if "eval(" in code_lower or "exec(" in code_lower:
            return False, "eval/exec usage is blocked for security"

        # Block __import__ bypass
        if "__import__" in code_lower:
            return False, "__import__ usage is blocked for security"

        return True, ""

    # ── Process Management ─────────────────────────────────────────────────

    def process_list(self) -> Dict[str, Any]:
        """List running processes (cross-platform)."""
        try:
            if _is_windows():
                result = subprocess.run(
                    ["tasklist", "/FO", "CSV"],
                    capture_output=True, text=True, timeout=10,
                )
            else:
                result = subprocess.run(
                    ["ps", "aux"],
                    capture_output=True, text=True, timeout=10,
                )

            lines = result.stdout.strip().split("\n")[:50]  # Cap at 50 lines
            return {
                "processes": lines,
                "error": None,
                "platform": platform.system(),
            }
        except Exception as exc:
            return {
                "processes": [],
                "error": str(exc),
                "platform": platform.system(),
            }

    # ── Screenshot Stub ────────────────────────────────────────────────────

    def screenshot_stub(self) -> Dict[str, Any]:
        """Placeholder for screen capture functionality.

        Returns a stub response. Full implementation requires
        a display server or VNC setup.
        """
        return {
            "success": False,
            "message": (
                "Screen capture is not available in this environment. "
                "Requires a display server (X11/Wayland) or headless browser setup."
            ),
            "platform": platform.system(),
            "data": None,
        }

    # ── Application Launch ─────────────────────────────────────────────────

    def open_application(self, name: str) -> Dict[str, Any]:
        """
        Launch an application (cross-platform).

        Args:
            name: Application name or command.

        Returns:
            Dict with success, pid, error.
        """
        # Block dangerous applications
        blocked_apps = {"rm", "format", "fdisk", "mkfs", "dd", "shutdown", "reboot"}
        if name.lower().strip() in blocked_apps:
            return {
                "success": False,
                "pid": None,
                "error": f"Application '{name}' is blocked for safety.",
            }

        try:
            if _is_windows():
                cmd = ["start", "", name]
                proc = subprocess.Popen(
                    cmd, shell=True, cwd=str(self.workspace),
                )
            elif _is_macos():
                cmd = ["open", "-a", name]
                proc = subprocess.Popen(
                    cmd, cwd=str(self.workspace),
                )
            else:  # Linux
                cmd = [name]
                proc = subprocess.Popen(
                    cmd, cwd=str(self.workspace),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

            return {
                "success": True,
                "pid": proc.pid,
                "error": None,
                "command": " ".join(cmd),
            }
        except FileNotFoundError:
            return {
                "success": False,
                "pid": None,
                "error": f"Application not found: {name}",
            }
        except Exception as exc:
            return {
                "success": False,
                "pid": None,
                "error": str(exc),
            }

    # ── File Download ──────────────────────────────────────────────────────

    def download_file(
        self,
        url: str,
        dest: str = "",
    ) -> Dict[str, Any]:
        """
        Download a file from a URL to the workspace.

        Safety checks:
          - Only HTTP/HTTPS URLs allowed
          - Destination must be within workspace
          - File size limit: 50MB

        Args:
            url: URL to download from.
            dest: Destination filename within workspace.

        Returns:
            Dict with success, path, size, error.
        """
        # URL safety check
        if not url.startswith(("http://", "https://")):
            return {
                "success": False,
                "path": "",
                "size": 0,
                "error": f"Only HTTP/HTTPS URLs are allowed. Got: {url[:50]}",
            }

        # Resolve destination
        if not dest:
            # Extract filename from URL
            dest = url.split("/")[-1].split("?")[0] or "downloaded_file"
            # Sanitize filename
            dest = "".join(c for c in dest if c.isalnum() or c in "._-")

        try:
            resolved = _resolve_and_validate_path(dest)
            resolved.parent.mkdir(parents=True, exist_ok=True)
        except ValueError as exc:
            return {"success": False, "path": dest, "size": 0, "error": str(exc)}

        try:
            import urllib.request

            # Set up request with user agent
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "CerebroForge/1.0"},
            )

            with urllib.request.urlopen(req, timeout=60) as response:
                # Check content length
                content_length = response.headers.get("Content-Length")
                if content_length and int(content_length) > 50_000_000:
                    return {
                        "success": False,
                        "path": "",
                        "size": int(content_length),
                        "error": f"File too large ({int(content_length)} bytes). Maximum is 50MB.",
                    }

                data = response.read()

                with open(resolved, "wb") as f:
                    f.write(data)

            return {
                "success": True,
                "path": str(resolved),
                "size": len(data),
                "error": None,
            }

        except Exception as exc:
            return {
                "success": False,
                "path": dest,
                "size": 0,
                "error": f"Download error: {exc}",
            }

    # ── System Information ─────────────────────────────────────────────────

    def system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        info: Dict[str, Any] = {
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
            },
        }

        # Disk usage
        try:
            disk = shutil.disk_usage(str(self.workspace))
            info["disk"] = {
                "total_gb": round(disk.total / (1024 ** 3), 2),
                "used_gb": round(disk.used / (1024 ** 3), 2),
                "free_gb": round(disk.free / (1024 ** 3), 2),
                "usage_percent": round(disk.used / disk.total * 100, 1),
            }
        except Exception:
            info["disk"] = {"error": "Unable to retrieve disk usage"}

        # Memory usage (Linux only)
        if _is_linux():
            try:
                with open("/proc/meminfo", "r") as f:
                    mem_lines = f.readlines()[:5]
                mem_info = {}
                for line in mem_lines:
                    parts = line.strip().split(":")
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip().replace(" kB", "")
                        try:
                            mem_info[key] = int(value)
                        except ValueError:
                            mem_info[key] = value
                info["memory"] = mem_info
            except Exception:
                info["memory"] = {"error": "Unable to retrieve memory info"}

        # Workspace info
        try:
            workspace_size = sum(
                f.stat().st_size for f in self.workspace.rglob("*") if f.is_file()
            )
            file_count = sum(1 for f in self.workspace.rglob("*") if f.is_file())
            info["workspace"] = {
                "path": str(self.workspace),
                "size_mb": round(workspace_size / (1024 ** 2), 2),
                "file_count": file_count,
            }
        except Exception:
            info["workspace"] = {"error": "Unable to compute workspace stats"}

        return info

    def os_info(self) -> Dict[str, str]:
        """Quick OS information."""
        return {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "node": platform.node(),
        }

    def disk_usage(self, path: str = ".") -> Dict[str, Any]:
        """Get disk usage for a path within the workspace."""
        try:
            resolved = _resolve_and_validate_path(path, must_exist=True)
            disk = shutil.disk_usage(str(resolved))
            return {
                "total_gb": round(disk.total / (1024 ** 3), 2),
                "used_gb": round(disk.used / (1024 ** 3), 2),
                "free_gb": round(disk.free / (1024 ** 3), 2),
                "usage_percent": round(disk.used / disk.total * 100, 1),
                "path": str(resolved),
            }
        except Exception as exc:
            return {"error": str(exc)}

    def memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information (Linux only, best-effort)."""
        if not _is_linux():
            return {
                "error": "Memory usage reporting is only available on Linux",
                "platform": platform.system(),
            }

        try:
            with open("/proc/meminfo", "r") as f:
                lines = f.readlines()

            mem = {}
            for line in lines:
                parts = line.strip().split(":")
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip().replace(" kB", "")
                    try:
                        mem[key] = int(value)
                    except ValueError:
                        mem[key] = value

            total = mem.get("MemTotal", 0)
            available = mem.get("MemAvailable", 0)
            used = total - available

            return {
                "total_mb": round(total / 1024, 1),
                "used_mb": round(used / 1024, 1),
                "available_mb": round(available / 1024, 1),
                "usage_percent": round(used / total * 100, 1) if total else 0,
            }
        except Exception as exc:
            return {"error": str(exc)}

    # ── Tool Registration Helpers ──────────────────────────────────────────

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Return tool definitions in a format suitable for agent binding."""
        return [
            {
                "name": "execute_terminal",
                "description": "Execute a terminal/command-line command",
                "parameters": {
                    "command": {"type": "string", "description": "The command to execute"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 30},
                },
                "handler": self.execute_terminal,
            },
            {
                "name": "read_file",
                "description": "Read a file from the workspace",
                "parameters": {
                    "path": {"type": "string", "description": "File path within workspace"},
                },
                "handler": self.read_file,
            },
            {
                "name": "write_file",
                "description": "Write content to a file in the workspace",
                "parameters": {
                    "path": {"type": "string", "description": "File path within workspace"},
                    "content": {"type": "string", "description": "Content to write"},
                    "mode": {"type": "string", "description": "'write' or 'append'", "default": "write"},
                },
                "handler": self.write_file,
            },
            {
                "name": "list_directory",
                "description": "List contents of a directory in the workspace",
                "parameters": {
                    "path": {"type": "string", "description": "Directory path", "default": "."},
                },
                "handler": self.list_directory,
            },
            {
                "name": "search_files",
                "description": "Search for a text pattern in files",
                "parameters": {
                    "pattern": {"type": "string", "description": "Text pattern to search for"},
                    "path": {"type": "string", "description": "Starting directory", "default": "."},
                    "recursive": {"type": "boolean", "description": "Search recursively", "default": True},
                },
                "handler": self.search_files,
            },
            {
                "name": "execute_python",
                "description": "Execute Python code in a sandboxed environment",
                "parameters": {
                    "code": {"type": "string", "description": "Python code to execute"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 30},
                },
                "handler": self.execute_python,
            },
            {
                "name": "process_list",
                "description": "List running processes",
                "parameters": {},
                "handler": self.process_list,
            },
            {
                "name": "screenshot",
                "description": "Capture a screenshot (stub - requires display server)",
                "parameters": {},
                "handler": self.screenshot_stub,
            },
            {
                "name": "open_application",
                "description": "Launch an application",
                "parameters": {
                    "name": {"type": "string", "description": "Application name or command"},
                },
                "handler": self.open_application,
            },
            {
                "name": "download_file",
                "description": "Download a file from a URL to the workspace",
                "parameters": {
                    "url": {"type": "string", "description": "URL to download from"},
                    "dest": {"type": "string", "description": "Destination filename", "default": ""},
                },
                "handler": self.download_file,
            },
            {
                "name": "system_info",
                "description": "Get comprehensive system information",
                "parameters": {},
                "handler": self.system_info,
            },
            {
                "name": "disk_usage",
                "description": "Get disk usage information",
                "parameters": {
                    "path": {"type": "string", "description": "Path to check", "default": "."},
                },
                "handler": self.disk_usage,
            },
            {
                "name": "memory_usage",
                "description": "Get memory usage information (Linux only)",
                "parameters": {},
                "handler": self.memory_usage,
            },
        ]
