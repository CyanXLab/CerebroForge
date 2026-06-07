"""
CerebroForge (铸脑) — Tool Registry & Base Tools
===================================================
Complete tool system with sandboxed execution, dynamic skill loading,
and evolutionary tool forging capabilities.

Imports config, memory, llm_client from the same directory.
"""

from __future__ import annotations

import ast
import io
import json
import logging
import math
import os
import platform
import re
import subprocess
import sys
import textwrap
import threading
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Imports from sibling modules
# ---------------------------------------------------------------------------
try:
    from backend.config import (
        WORKSPACE_DIR,
        PROJECT_ROOT,
        NVIDIA_BASE_URL,
        NVIDIA_API_KEY,
        DEFAULT_MODEL,
        FALLBACK_MODEL,
        DEFAULT_TEMPERATURE,
        DEFAULT_TOP_P,
        DEFAULT_MAX_TOKENS,
        MAX_RETRIES as CONFIG_MAX_RETRIES,
        MAX_TASK_EXECUTION_CNT,
        MAX_TOOL_FORGE_PER_TASK,
        BASE_TOOLS,
    )
except ImportError:
    from config import (
        WORKSPACE_DIR,
        PROJECT_ROOT,
        NVIDIA_BASE_URL,
        NVIDIA_API_KEY,
        DEFAULT_MODEL,
        FALLBACK_MODEL,
        DEFAULT_TEMPERATURE,
        DEFAULT_TOP_P,
        DEFAULT_MAX_TOKENS,
        MAX_RETRIES as CONFIG_MAX_RETRIES,
        MAX_TASK_EXECUTION_CNT,
        MAX_TOOL_FORGE_PER_TASK,
        BASE_TOOLS,
    )

try:
    from backend.memory import MemorySystem
except ImportError:
    from memory import MemorySystem

try:
    from backend.llm_client import LLMClient
except ImportError:
    from llm_client import LLMClient

# ---------------------------------------------------------------------------
# Derived configuration
# ---------------------------------------------------------------------------
DYNAMIC_SKILLS_DIR: Path = PROJECT_ROOT / "dynamic_skills"
DYNAMIC_SKILLS_PUBLIC_DIR: Path = PROJECT_ROOT / "dynamic_skills_public"
EGL_THRESHOLD: float = 0.6  # EGL > threshold → system in exploration phase

# Ensure dynamic skill directories exist
DYNAMIC_SKILLS_DIR.mkdir(parents=True, exist_ok=True)
DYNAMIC_SKILLS_PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("cerebroforge.tools")

# ---------------------------------------------------------------------------
# Sandbox: safe builtins & modules
# ---------------------------------------------------------------------------

_SAFE_BUILTINS: Dict[str, Any] = {
    # Core types
    "bool": bool,
    "int": int,
    "float": float,
    "str": str,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "frozenset": frozenset,
    "bytes": bytes,
    "bytearray": bytearray,
    "complex": complex,
    "range": range,
    "slice": slice,
    "type": type,
    "object": object,
    # Conversion / introspection
    "abs": abs,
    "all": all,
    "any": any,
    "bin": bin,
    "chr": chr,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "format": format,
    "getattr": getattr,
    "hasattr": hasattr,
    "hash": hash,
    "hex": hex,
    "id": id,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "iter": iter,
    "len": len,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "oct": oct,
    "ord": ord,
    "pow": pow,
    "print": print,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "setattr": setattr,
    "sorted": sorted,
    "sum": sum,
    "zip": zip,
    # Constants
    "True": True,
    "False": False,
    "None": None,
    "NotImplemented": NotImplemented,
    "Ellipsis": Ellipsis,
    # Exception types (for catching)
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "AttributeError": AttributeError,
    "RuntimeError": RuntimeError,
    "StopIteration": StopIteration,
    "ZeroDivisionError": ZeroDivisionError,
    "OverflowError": OverflowError,
    "FileNotFoundError": FileNotFoundError,
    # Misc
    "staticmethod": staticmethod,
    "classmethod": classmethod,
    "property": property,
    "super": super,
}

_SAFE_MODULES: Dict[str, Any] = {
    "math": math,
    "json": json,
    "re": re,
    "collections": __import__("collections"),
    "itertools": __import__("itertools"),
    "functools": __import__("functools"),
    "datetime": __import__("datetime"),
    "decimal": __import__("decimal"),
    "fractions": __import__("fractions"),
    "string": __import__("string"),
    "hashlib": __import__("hashlib"),
    "base64": __import__("base64"),
    "urllib.parse": __import__("urllib.parse", fromlist=["parse"]),
    "textwrap": textwrap,
    "io": io,
    "copy": __import__("copy"),
    "operator": __import__("operator"),
    "random": __import__("random"),
    "statistics": __import__("statistics"),
    "typing": __import__("typing"),
    "pydantic": __import__("pydantic"),
}


# ---------------------------------------------------------------------------
# Helper: resolve workspace-relative path safely
# ---------------------------------------------------------------------------

def _resolve_workspace_path(path: str) -> Path:
    """Resolve a path, ensuring it stays within the workspace directory."""
    workspace = WORKSPACE_DIR.resolve()
    resolved = (workspace / path).resolve() if not os.path.isabs(path) else Path(path).resolve()
    # Security check: prevent path traversal outside workspace
    try:
        resolved.relative_to(workspace)
    except ValueError:
        raise PermissionError(f"Path '{path}' is outside the workspace directory.")
    return resolved


# ---------------------------------------------------------------------------
# Base Tool Implementations
# ---------------------------------------------------------------------------

def tool_web_search(query: str, num_results: int = 5) -> str:
    """Search the web using DuckDuckGo HTML and return structured results.

    Args:
        query: Search query string.
        num_results: Maximum number of results to return (default 5).

    Returns:
        JSON string with list of {title, href, body} dicts.
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        params = {"q": query, "kl": "us-en"}
        resp = requests.get(
            "https://html.duckduckgo.com/html/",
            params=params,
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        results: List[Dict[str, str]] = []
        for item in soup.select(".result"):
            if len(results) >= num_results:
                break
            title_el = item.select_one(".result__title a")
            snippet_el = item.select_one(".result__snippet")
            if title_el is None:
                continue
            href = title_el.get("href", "")
            # DuckDuckGo redirects through a prefix
            if href.startswith("//duckduckgo.com/l/?uddg="):
                from urllib.parse import unquote
                href = unquote(href.split("uddg=")[1].split("&")[0])
            results.append({
                "title": title_el.get_text(strip=True),
                "href": href,
                "body": snippet_el.get_text(strip=True) if snippet_el else "",
            })

        return json.dumps(results, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning("web_search failed for '%s': %s", query, e)
        return json.dumps({"error": str(e)})


def tool_web_fetch(url: str) -> str:
    """Fetch a URL and extract readable text content with BeautifulSoup.

    Args:
        url: The URL to fetch.

    Returns:
        Extracted text content from the page.
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")

        if "application/json" in content_type:
            try:
                return json.dumps(resp.json(), ensure_ascii=False, indent=2)
            except json.JSONDecodeError:
                return resp.text[:8000]

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove script, style, nav, footer, header elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        # Try to find main content area first
        main = soup.find("main") or soup.find("article") or soup.find(class_="content") or soup
        text = main.get_text(separator="\n", strip=True)

        # Truncate to avoid excessive output
        if len(text) > 12000:
            text = text[:12000] + "\n... [truncated]"

        return text
    except Exception as e:
        logger.warning("web_fetch failed for '%s': %s", url, e)
        return json.dumps({"error": str(e)})


def tool_python_exec(code: str, timeout: int = 8) -> str:
    """Execute Python code in a sandboxed environment with restricted builtins.

    Args:
        code: Python source code to execute.
        timeout: Maximum execution time in seconds (default 8).

    Returns:
        Captured stdout output or error message as JSON.
    """
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    # Build sandbox globals
    sandbox_globals: Dict[str, Any] = {"__builtins__": _SAFE_BUILTINS.copy()}
    # Inject safe modules
    for mod_name, mod in _SAFE_MODULES.items():
        sandbox_globals[mod_name.split(".")[-1]] = mod

    exec_error: Optional[Exception] = None

    def _run() -> None:
        nonlocal exec_error
        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(compile(code, "<sandbox>", "exec"), sandbox_globals)  # noqa: S102
        except Exception as exc:
            exec_error = exc

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        return json.dumps({
            "status": "timeout",
            "error": f"Execution timed out after {timeout}s",
            "output": "",
        })

    if exec_error is not None:
        return json.dumps({
            "status": "error",
            "error": f"{type(exec_error).__name__}: {exec_error}",
            "output": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue(),
        })

    output = stdout_buf.getvalue()
    stderr_out = stderr_buf.getvalue()
    return json.dumps({
        "status": "success",
        "output": output,
        "stderr": stderr_out,
    }, ensure_ascii=False)


def tool_file_read(path: str) -> str:
    """Read a file from the workspace directory.

    Args:
        path: Relative path within the workspace.

    Returns:
        File contents as a string.
    """
    try:
        resolved = _resolve_workspace_path(path)
        if not resolved.is_file():
            return json.dumps({"error": f"File not found: {path}"})
        content = resolved.read_text(encoding="utf-8", errors="replace")
        if len(content) > 100000:
            content = content[:100000] + "\n... [truncated]"
        return content
    except PermissionError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        return json.dumps({"error": str(e)})


def tool_file_write(path: str, content: str, mode: str = "w") -> str:
    """Write content to a file in the workspace directory.

    Args:
        path: Relative path within the workspace.
        content: Content to write.
        mode: Write mode — 'w' for overwrite, 'a' for append (default 'w').

    Returns:
        Confirmation message or error as JSON.
    """
    try:
        if mode not in ("w", "a"):
            return json.dumps({"error": f"Invalid mode '{mode}'. Use 'w' or 'a'."})
        resolved = _resolve_workspace_path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return json.dumps({"status": "ok", "path": str(resolved), "bytes_written": len(content.encode("utf-8"))})
    except PermissionError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        return json.dumps({"error": str(e)})


def tool_text_extract(path: str) -> str:
    """Extract text content from PDF/TXT files.

    Args:
        path: Relative path to file within workspace.

    Returns:
        Extracted text content.
    """
    try:
        resolved = _resolve_workspace_path(path)
        if not resolved.is_file():
            return json.dumps({"error": f"File not found: {path}"})

        suffix = resolved.suffix.lower()

        if suffix == ".txt":
            text = resolved.read_text(encoding="utf-8", errors="replace")
        elif suffix == ".pdf":
            try:
                import pymupdf  # type: ignore
                doc = pymupdf.open(str(resolved))
                pages: List[str] = []
                for page in doc:
                    pages.append(page.get_text())
                text = "\n\n".join(pages)
                doc.close()
            except ImportError:
                try:
                    from pypdf import PdfReader  # type: ignore
                    reader = PdfReader(str(resolved))
                    pages = [page.extract_text() or "" for page in reader.pages]
                    text = "\n\n".join(pages)
                except ImportError:
                    return json.dumps({
                        "error": "No PDF reader available. Install pymupdf or pypdf."
                    })
        elif suffix in (".md", ".csv", ".json", ".xml", ".html", ".htm", ".log"):
            text = resolved.read_text(encoding="utf-8", errors="replace")
        else:
            # Try reading as text
            try:
                text = resolved.read_text(encoding="utf-8", errors="replace")
            except Exception:
                return json.dumps({"error": f"Unsupported file type: {suffix}"})

        if len(text) > 100000:
            text = text[:100000] + "\n... [truncated]"

        return text
    except Exception as e:
        return json.dumps({"error": str(e)})


def tool_image_query(path_or_desc: str) -> str:
    """Query a vision model about an image (stub for future integration).

    Args:
        path_or_desc: Path to image in workspace, or a text description.

    Returns:
        Description or analysis result as JSON.
    """
    # Check if the input is a file path
    try:
        resolved = _resolve_workspace_path(path_or_desc)
        if resolved.is_file():
            # Future: send image to vision model API
            return json.dumps({
                "status": "stub",
                "message": (
                    f"Image file found at '{path_or_desc}'. "
                    "Vision model integration pending. "
                    f"File size: {resolved.stat().st_size} bytes."
                ),
            })
    except (PermissionError, ValueError):
        pass

    # Treat as a text description
    return json.dumps({
        "status": "stub",
        "message": f"Image query received for description: '{path_or_desc}'. Vision model integration pending.",
    })


def tool_calculate(expression: str) -> str:
    """Safely evaluate a mathematical expression.

    Args:
        expression: Mathematical expression string (e.g., '2**10 + sqrt(144)').

    Returns:
        The computed result as JSON.
    """
    # Allow math functions by name
    safe_names: Dict[str, Any] = {}
    for name in dir(math):
        if not name.startswith("_"):
            safe_names[name] = getattr(math, name)
    # Add common aliases
    safe_names["sqrt"] = math.sqrt
    safe_names["log2"] = math.log2
    safe_names["log10"] = math.log10
    safe_names["pi"] = math.pi
    safe_names["e"] = math.e
    safe_names["inf"] = math.inf

    try:
        # Parse the expression as an AST to ensure it's a single expression
        tree = ast.parse(expression, mode="eval")
        # Validate: only allow safe node types
        for node in ast.walk(tree):
            if isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
                                ast.Name, ast.Call, ast.Attribute, ast.Load,
                                ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
                                ast.Mod, ast.FloorDiv, ast.USub, ast.UAdd,
                                ast.BitAnd, ast.BitOr, ast.BitXor, ast.Invert,
                                ast.LShift, ast.RShift)):
                continue
            if isinstance(node, (ast.Compare, ast.BoolOp, ast.And, ast.Or, ast.Not,
                                 ast.Lt, ast.Gt, ast.LtE, ast.GtE, ast.Eq, ast.NotEq)):
                continue
            return json.dumps({"error": f"Disallowed syntax element: {type(node).__name__}"})

        result = eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, safe_names)  # noqa: S307
        return json.dumps({"result": result, "expression": expression})
    except Exception as e:
        return json.dumps({"error": f"Calculation error: {type(e).__name__}: {e}"})


def tool_run_terminal(command: str, timeout: int = 15) -> str:
    """Execute a terminal command (multi-platform: bash on Linux, cmd/PowerShell on Windows).

    Args:
        command: Shell command to execute.
        timeout: Timeout in seconds (default 15).

    Returns:
        JSON with stdout, stderr, and return code.
    """
    try:
        is_windows = platform.system() == "Windows"

        if is_windows:
            # Try PowerShell first, fall back to cmd
            try:
                shell_args = ["powershell", "-NoProfile", "-Command", command]
                result = subprocess.run(
                    shell_args,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=str(WORKSPACE_DIR),
                )
            except FileNotFoundError:
                shell_args = ["cmd", "/c", command]
                result = subprocess.run(
                    shell_args,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=str(WORKSPACE_DIR),
                )
        else:
            result = subprocess.run(
                ["bash", "-c", command],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(WORKSPACE_DIR),
            )

        stdout = result.stdout
        stderr = result.stderr
        # Truncate if too large
        if len(stdout) > 50000:
            stdout = stdout[:50000] + "\n... [truncated]"
        if len(stderr) > 10000:
            stderr = stderr[:10000] + "\n... [truncated]"

        return json.dumps({
            "returncode": result.returncode,
            "stdout": stdout,
            "stderr": stderr,
        }, ensure_ascii=False)
    except subprocess.TimeoutExpired:
        return json.dumps({
            "returncode": -1,
            "stdout": "",
            "stderr": f"Command timed out after {timeout}s",
        })
    except Exception as e:
        return json.dumps({"error": str(e), "returncode": -1})


def tool_list_files(path: str = ".") -> str:
    """List files in a directory within the workspace (cross-platform).

    Args:
        path: Relative directory path within workspace (default '.').

    Returns:
        JSON list of files with metadata.
    """
    try:
        resolved = _resolve_workspace_path(path)
        if not resolved.is_dir():
            return json.dumps({"error": f"Directory not found: {path}"})

        entries: List[Dict[str, Any]] = []
        for entry in sorted(resolved.iterdir()):
            try:
                stat = entry.stat()
                entries.append({
                    "name": entry.name,
                    "type": "dir" if entry.is_dir() else "file",
                    "size": stat.st_size if entry.is_file() else None,
                    "modified": stat.st_mtime,
                })
            except (PermissionError, OSError):
                entries.append({
                    "name": entry.name,
                    "type": "unknown",
                    "size": None,
                    "modified": None,
                })

        return json.dumps(entries, ensure_ascii=False, indent=2)
    except PermissionError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        return json.dumps({"error": str(e)})


def tool_grep_files(pattern: str, path: str = ".") -> str:
    """Search for a text pattern in workspace files (recursive).

    Args:
        pattern: Regular expression pattern to search for.
        path: Relative directory path within workspace (default '.').

    Returns:
        JSON list of matches with file, line number, and content.
    """
    try:
        resolved = _resolve_workspace_path(path)
        if not resolved.exists():
            return json.dumps({"error": f"Path not found: {path}"})

        regex = re.compile(pattern, re.IGNORECASE)
        matches: List[Dict[str, Any]] = []
        max_matches = 100
        max_file_size = 5 * 1024 * 1024  # 5MB per file

        if resolved.is_file():
            files_to_search = [resolved]
        else:
            files_to_search = []
            for root, _dirs, filenames in os.walk(resolved):
                for fname in filenames:
                    fpath = Path(root) / fname
                    # Skip hidden dirs and binary-like files
                    if any(part.startswith(".") for part in fpath.relative_to(resolved).parts):
                        continue
                    if fpath.suffix.lower() in (
                        ".pyc", ".pyo", ".so", ".dll", ".exe", ".bin",
                        ".png", ".jpg", ".jpeg", ".gif", ".zip", ".tar", ".gz",
                    ):
                        continue
                    try:
                        if fpath.stat().st_size <= max_file_size:
                            files_to_search.append(fpath)
                    except OSError:
                        continue

        for fpath in files_to_search:
            if len(matches) >= max_matches:
                break
            try:
                text = fpath.read_text(encoding="utf-8", errors="ignore")
                for line_no, line in enumerate(text.splitlines(), start=1):
                    if regex.search(line):
                        rel_path = str(fpath.relative_to(WORKSPACE_DIR.resolve()))
                        matches.append({
                            "file": rel_path,
                            "line": line_no,
                            "content": line.strip()[:300],
                        })
                        if len(matches) >= max_matches:
                            break
            except (OSError, UnicodeDecodeError):
                continue

        return json.dumps(matches, ensure_ascii=False, indent=2)
    except re.error as e:
        return json.dumps({"error": f"Invalid regex pattern: {e}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def tool_computer_use(action: str, **kwargs: Any) -> str:
    """Dispatch to the computer_use.py toolkit for GUI/desktop automation.

    Args:
        action: The computer use action (e.g., 'screenshot', 'execute_terminal',
                'read_file', 'write_file', 'list_directory', 'search_files',
                'execute_python', 'process_list', 'open_application',
                'download_file', 'system_info', 'disk_usage', 'memory_usage').
        **kwargs: Action-specific parameters.

    Returns:
        JSON result from the computer_use module.
    """
    try:
        from backend.computer_use import ComputerUseToolkit
    except ImportError:
        from computer_use import ComputerUseToolkit

        toolkit = ComputerUseToolkit()

        # Map actions to toolkit methods
        action_map: Dict[str, Callable[..., Dict[str, Any]]] = {
            "screenshot": toolkit.screenshot_stub,
            "execute_terminal": toolkit.execute_terminal,
            "read_file": toolkit.read_file,
            "write_file": toolkit.write_file,
            "list_directory": toolkit.list_directory,
            "search_files": toolkit.search_files,
            "execute_python": toolkit.execute_python,
            "process_list": toolkit.process_list,
            "open_application": toolkit.open_application,
            "download_file": toolkit.download_file,
            "system_info": toolkit.system_info,
            "disk_usage": toolkit.disk_usage,
            "memory_usage": toolkit.memory_usage,
            "os_info": toolkit.os_info,
        }

        handler = action_map.get(action)
        if handler is None:
            available = ", ".join(sorted(action_map.keys()))
            return json.dumps({
                "error": f"Unknown computer_use action: '{action}'. Available: {available}",
            })

        result = handler(**kwargs)
        return json.dumps(result, ensure_ascii=False, default=str)

    except Exception as e:
        return json.dumps({"error": f"computer_use error: {type(e).__name__}: {e}"})


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """Central registry for all tools — base, dynamic, and forged.

    Supports registration, execution with telemetry, dynamic skill loading,
    and evolutionary tool forging via LLM.
    """

    def __init__(
        self,
        memory: Optional[MemorySystem] = None,
        llm_client: Optional[LLMClient] = None,
    ) -> None:
        self._tools: Dict[str, Dict[str, Any]] = {}
        self._memory = memory or MemorySystem()
        self._llm = llm_client or LLMClient()

        # Register all base tools
        self._register_base_tools()

    # -----------------------------------------------------------------------
    # Registration
    # -----------------------------------------------------------------------

    def register(
        self,
        name: str,
        func: Callable[..., Any],
        doc: str = "",
        schema: Optional[Dict[str, Any]] = None,
        is_dynamic: bool = False,
        code: Optional[str] = None,
    ) -> None:
        """Register a tool in the registry.

        Args:
            name: Unique tool name.
            func: Callable implementing the tool.
            doc: Human-readable documentation string.
            schema: Optional JSON schema for input validation.
            is_dynamic: Whether this is a dynamically forged tool.
            code: Source code for dynamic tools (used in re-generation).
        """
        if name in self._tools:
            logger.warning("Overwriting existing tool: %s", name)

        self._tools[name] = {
            "func": func,
            "doc": doc or (func.__doc__ or ""),
            "schema": schema or {},
            "is_dynamic": is_dynamic,
            "code": code,
        }
        # Also register in the memory system's tools table
        try:
            self._memory.register_tool(
                name=name,
                description=doc or "",
                code=code or "",
                tool_type="dynamic" if is_dynamic else "base",
            )
        except Exception:
            pass  # Memory registration is best-effort
        logger.info("Registered tool: %s (dynamic=%s)", name, is_dynamic)

    def _register_base_tools(self) -> None:
        """Register all built-in base tools."""
        base_tools: List[Tuple[str, Callable[..., Any], str, Dict[str, Any]]] = [
            (
                "web_search",
                tool_web_search,
                "Search the web using DuckDuckGo. Args: query (str), num_results (int, default 5).",
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "num_results": {"type": "integer", "description": "Max results", "default": 5},
                    },
                    "required": ["query"],
                },
            ),
            (
                "web_fetch",
                tool_web_fetch,
                "Fetch a URL and extract readable text with BeautifulSoup. Args: url (str).",
                {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to fetch"},
                    },
                    "required": ["url"],
                },
            ),
            (
                "python_exec",
                tool_python_exec,
                "Execute Python code in sandbox with restricted builtins. Args: code (str), timeout (int, default 8).",
                {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Python source code"},
                        "timeout": {"type": "integer", "description": "Timeout seconds", "default": 8},
                    },
                    "required": ["code"],
                },
            ),
            (
                "file_read",
                tool_file_read,
                "Read a file from the workspace. Args: path (str).",
                {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path in workspace"},
                    },
                    "required": ["path"],
                },
            ),
            (
                "file_write",
                tool_file_write,
                "Write content to a workspace file. Args: path (str), content (str), mode (str, default 'w').",
                {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path in workspace"},
                        "content": {"type": "string", "description": "Content to write"},
                        "mode": {"type": "string", "enum": ["w", "a"], "default": "w"},
                    },
                    "required": ["path", "content"],
                },
            ),
            (
                "text_extract",
                tool_text_extract,
                "Extract text from PDF/TXT files. Args: path (str).",
                {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path in workspace"},
                    },
                    "required": ["path"],
                },
            ),
            (
                "image_query",
                tool_image_query,
                "Query a vision model about an image. Args: path_or_desc (str).",
                {
                    "type": "object",
                    "properties": {
                        "path_or_desc": {"type": "string", "description": "Image path or description"},
                    },
                    "required": ["path_or_desc"],
                },
            ),
            (
                "calculate",
                tool_calculate,
                "Safely evaluate a math expression. Args: expression (str).",
                {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Math expression"},
                    },
                    "required": ["expression"],
                },
            ),
            (
                "run_terminal",
                tool_run_terminal,
                "Execute terminal command (Linux bash / Windows cmd/PowerShell). Args: command (str), timeout (int, default 15).",
                {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Shell command"},
                        "timeout": {"type": "integer", "description": "Timeout seconds", "default": 15},
                    },
                    "required": ["command"],
                },
            ),
            (
                "list_files",
                tool_list_files,
                "List files in a workspace directory. Args: path (str, default '.').",
                {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Directory path", "default": "."},
                    },
                    "required": [],
                },
            ),
            (
                "grep_files",
                tool_grep_files,
                "Search for text pattern in workspace files. Args: pattern (str), path (str, default '.').",
                {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Regex pattern"},
                        "path": {"type": "string", "description": "Search directory", "default": "."},
                    },
                    "required": ["pattern"],
                },
            ),
            (
                "computer_use",
                tool_computer_use,
                "Dispatch GUI/desktop automation via computer_use toolkit. Args: action (str), **kwargs.",
                {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": "Action: screenshot, execute_terminal, read_file, etc.",
                            "enum": [
                                "screenshot", "execute_terminal", "read_file", "write_file",
                                "list_directory", "search_files", "execute_python",
                                "process_list", "open_application", "download_file",
                                "system_info", "disk_usage", "memory_usage", "os_info",
                            ],
                        },
                    },
                    "required": ["action"],
                },
            ),
        ]

        for name, func, doc, schema in base_tools:
            self.register(name, func, doc=doc, schema=schema, is_dynamic=False)

    # -----------------------------------------------------------------------
    # Query
    # -----------------------------------------------------------------------

    def get_tool_names(self) -> List[str]:
        """Return a list of all registered tool names."""
        return list(self._tools.keys())

    def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
        """Return tool metadata by name, or None."""
        return self._tools.get(name)

    def get_dynamic_tools(self) -> Dict[str, Dict[str, Any]]:
        """Return all dynamic (forged) tools."""
        return {n: t for n, t in self._tools.items() if t["is_dynamic"]}

    # -----------------------------------------------------------------------
    # Execution
    # -----------------------------------------------------------------------

    def execute(self, name: str, args: Dict[str, Any]) -> str:
        """Execute a tool by name with given arguments.

        Tracks success/failure and latency in memory for EGL computation.

        Args:
            name: Tool name.
            args: Keyword arguments to pass to the tool function.

        Returns:
            Tool output as a string.
        """
        tool_info = self._tools.get(name)
        if tool_info is None:
            logger.error("Tool not found: %s", name)
            return json.dumps({"error": f"Tool '{name}' not found in registry."})

        func = tool_info["func"]
        start = time.monotonic()
        success = False

        try:
            # Validate against schema if present
            schema = tool_info.get("schema", {})
            if schema and "required" in schema:
                for req_field in schema["required"]:
                    if req_field not in args:
                        return json.dumps({
                            "error": f"Missing required argument '{req_field}' for tool '{name}'.",
                            "schema": schema,
                        })

            result = func(**args)
            success = True
            return result
        except Exception as e:
            logger.exception("Tool '%s' execution failed", name)
            return json.dumps({"error": f"Tool '{name}' failed: {type(e).__name__}: {e}"})
        finally:
            elapsed = time.monotonic() - start
            latency_ms = elapsed * 1000
            try:
                self._memory.update_tool_stats(
                    name=name,
                    success=success,
                    execution_time=elapsed,
                )
            except Exception:
                pass  # Stats update is best-effort

    # -----------------------------------------------------------------------
    # Dynamic Tool Forging (single tool)
    # -----------------------------------------------------------------------

    def forge_new_tool(
        self,
        capability_desc: str,
        example_usage: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Generate a new atomic Python tool via LLM, validate in sandbox, register.

        Args:
            capability_desc: Natural language description of the tool's capability.
            example_usage: Example of how the tool should be called.

        Returns:
            Dict with tool info if successful, None otherwise.
        """
        logger.info("Forging new tool for capability: %s", capability_desc[:80])

        prompt = textwrap.dedent(f"""\
            You are a tool forge engine for CerebroForge (铸脑).
            Generate a COMPLETE, self-contained Python tool that implements the requested capability.

            REQUIREMENTS:
            1. The code MUST define __TOOL_META__ = {{"name": "...", "description": "...", "dependencies": [...]}}
            2. The code MUST define a pydantic InputModel and OutputModel
            3. The code MUST define a run(input: InputModel) -> OutputModel function
            4. Only use standard library modules + pydantic. If external deps needed, list in __TOOL_META__.dependencies
            5. The code MUST be safe: no file system writes outside designated areas, no network attacks, no eval/exec
            6. Return ONLY the Python code, no markdown fences, no explanation

            Capability requested: {capability_desc}
            Example usage: {example_usage or 'N/A'}

            Generate the tool code now:
        """)

        messages = [
            {"role": "system", "content": "You are a Python tool code generator. Return ONLY valid Python code."},
            {"role": "user", "content": prompt},
        ]

        try:
            raw_code = self._llm.chat(
                messages,
                temperature=0.2,
                max_tokens=4096,
            )
        except Exception as e:
            logger.error("LLM call failed during tool forging: %s", e)
            return None

        # Clean up markdown fences if present
        code = _clean_llm_code(raw_code)

        # Validate in sandbox
        validation = self._validate_tool_code(code)
        if not validation["valid"]:
            logger.warning("Sandbox validation failed for forged tool: %s", validation["error"])
            return None

        # Extract tool metadata
        meta = validation.get("meta", {})
        tool_name = meta.get("name", f"dynamic_{len(self._tools)}")
        tool_doc = meta.get("description", capability_desc)

        # Build the executable function
        tool_func = self._build_tool_func(code, tool_name)

        # Register
        self.register(
            name=tool_name,
            func=tool_func,
            doc=tool_doc,
            schema=validation.get("schema", {}),
            is_dynamic=True,
            code=code,
        )

        # Persist to dynamic_skills/
        self._persist_skill(tool_name, code, meta)

        # Record in memory evolution log
        try:
            self._memory.log_evolution(
                event_type="tool_forged",
                description=f"Forged tool '{tool_name}': {capability_desc[:200]}",
                after_state={"name": tool_name, "meta": meta},
            )
        except Exception:
            pass

        return {
            "name": tool_name,
            "description": tool_doc,
            "meta": meta,
            "code": code,
        }

    def _validate_tool_code(self, code: str) -> Dict[str, Any]:
        """Validate tool code in a sandbox environment.

        Returns dict with keys: valid, error, meta, schema.
        """
        result: Dict[str, Any] = {"valid": False, "error": None, "meta": {}, "schema": {}}

        # 1. Syntax check
        try:
            ast.parse(code)
        except SyntaxError as e:
            result["error"] = f"SyntaxError: {e}"
            return result

        # 2. Sandbox execution
        sandbox_globals: Dict[str, Any] = {"__builtins__": _SAFE_BUILTINS.copy()}
        for mod_name, mod in _SAFE_MODULES.items():
            sandbox_globals[mod_name.split(".")[-1]] = mod

        try:
            exec(compile(code, "<tool_validation>", "exec"), sandbox_globals)  # noqa: S102
        except Exception as e:
            result["error"] = f"Sandbox execution error: {type(e).__name__}: {e}"
            return result

        # 3. Check for required components
        if "__TOOL_META__" not in sandbox_globals:
            result["error"] = "Missing __TOOL_META__ definition"
            return result

        meta = sandbox_globals["__TOOL_META__"]
        result["meta"] = meta

        if "InputModel" not in sandbox_globals:
            result["error"] = "Missing InputModel class"
            return result

        if "OutputModel" not in sandbox_globals:
            result["error"] = "Missing OutputModel class"
            return result

        if "run" not in sandbox_globals:
            result["error"] = "Missing run() function"
            return result

        # 4. Extract schema from InputModel
        try:
            input_model = sandbox_globals["InputModel"]
            if hasattr(input_model, "model_json_schema"):
                result["schema"] = input_model.model_json_schema()
            elif hasattr(input_model, "schema"):
                result["schema"] = input_model.schema()
        except Exception:
            pass

        result["valid"] = True
        return result

    def _build_tool_func(self, code: str, tool_name: str) -> Callable[..., str]:
        """Build a callable function from tool code.

        The returned function accepts keyword arguments, constructs an InputModel,
        calls run(), and returns the OutputModel as a JSON string.
        """
        def dynamic_tool(**kwargs: Any) -> str:
            # Re-exec the code in a fresh sandbox
            sandbox_globals: Dict[str, Any] = {"__builtins__": _SAFE_BUILTINS.copy()}
            for mod_name, mod in _SAFE_MODULES.items():
                sandbox_globals[mod_name.split(".")[-1]] = mod

            try:
                exec(compile(code, f"<dynamic_tool_{tool_name}>", "exec"), sandbox_globals)  # noqa: S102
            except Exception as e:
                return json.dumps({"error": f"Failed to load tool code: {e}"})

            InputModel = sandbox_globals.get("InputModel")
            OutputModel = sandbox_globals.get("OutputModel")
            run_fn = sandbox_globals.get("run")

            if InputModel is None or run_fn is None:
                return json.dumps({"error": "Tool code missing InputModel or run()"})

            try:
                input_obj = InputModel(**kwargs)
                output_obj = run_fn(input_obj)
                if hasattr(output_obj, "model_dump_json"):
                    return output_obj.model_dump_json()
                if hasattr(output_obj, "json"):
                    return output_obj.json()
                return json.dumps({"result": str(output_obj)})
            except Exception as e:
                return json.dumps({"error": f"Tool execution error: {type(e).__name__}: {e}"})

        dynamic_tool.__name__ = tool_name
        dynamic_tool.__doc__ = f"Dynamic tool: {tool_name}"
        return dynamic_tool

    def _persist_skill(self, name: str, code: str, meta: Dict[str, Any]) -> None:
        """Persist a dynamic skill to the dynamic_skills/ directory."""
        skills_dir = DYNAMIC_SKILLS_DIR
        skills_dir.mkdir(parents=True, exist_ok=True)

        skill_path = skills_dir / f"{name}.py"
        skill_path.write_text(code, encoding="utf-8")

        meta_path = skills_dir / f"{name}.meta.json"
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Persisted skill '%s' to %s", name, skill_path)

    # -----------------------------------------------------------------------
    # Load dynamic skills from disk
    # -----------------------------------------------------------------------

    def load_dynamic_skills(self) -> int:
        """Load all dynamic skills from the dynamic_skills/ directory.

        Returns:
            Number of skills loaded.
        """
        skills_dir = DYNAMIC_SKILLS_DIR
        if not skills_dir.is_dir():
            logger.info("No dynamic_skills directory found.")
            return 0

        count = 0
        for skill_file in sorted(skills_dir.glob("*.py")):
            name = skill_file.stem
            try:
                code = skill_file.read_text(encoding="utf-8")

                # Validate
                validation = self._validate_tool_code(code)
                if not validation["valid"]:
                    logger.warning(
                        "Skipping dynamic skill '%s': validation failed: %s",
                        name, validation["error"],
                    )
                    continue

                meta = validation.get("meta", {})
                tool_name = meta.get("name", name)
                tool_doc = meta.get("description", f"Dynamic skill: {name}")

                tool_func = self._build_tool_func(code, tool_name)
                self.register(
                    name=tool_name,
                    func=tool_func,
                    doc=tool_doc,
                    schema=validation.get("schema", {}),
                    is_dynamic=True,
                    code=code,
                )
                count += 1
                logger.info("Loaded dynamic skill: %s", tool_name)
            except Exception as e:
                logger.warning("Failed to load dynamic skill '%s': %s", name, e)

        return count

    # -----------------------------------------------------------------------
    # Documentation
    # -----------------------------------------------------------------------

    def get_tool_documentation(self) -> str:
        """Return formatted documentation for all registered tools.

        Used for LLM context to inform tool selection.
        """
        lines: List[str] = ["# Available Tools\n"]

        for name, info in sorted(self._tools.items()):
            dynamic_marker = " [DYNAMIC]" if info["is_dynamic"] else ""
            lines.append(f"## {name}{dynamic_marker}\n")
            lines.append(f"{info['doc']}\n")

            schema = info.get("schema", {})
            if schema:
                props = schema.get("properties", {})
                if props:
                    lines.append("**Parameters:**\n")
                    for pname, pinfo in props.items():
                        ptype = pinfo.get("type", "any")
                        pdesc = pinfo.get("description", "")
                        pdefault = pinfo.get("default")
                        default_str = f" (default: {pdefault})" if pdefault is not None else ""
                        required = pname in schema.get("required", [])
                        req_marker = " (required)" if required else ""
                        lines.append(f"- `{pname}` ({ptype}): {pdesc}{default_str}{req_marker}")
                lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helper: clean LLM-generated code
# ---------------------------------------------------------------------------

def _clean_llm_code(raw_code: str) -> str:
    """Clean LLM-generated code by removing markdown fences and extra whitespace."""
    code = raw_code.strip()
    # Remove markdown fences
    if code.startswith("```python"):
        code = code[len("```python"):]
    elif code.startswith("```Python"):
        code = code[len("```Python"):]
    elif code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    return code.strip()


# ---------------------------------------------------------------------------
# Module-level singleton (lazy)
# ---------------------------------------------------------------------------

_registry_instance: Optional[ToolRegistry] = None
_registry_lock = threading.Lock()


def get_tool_registry() -> ToolRegistry:
    """Return the global ToolRegistry singleton."""
    global _registry_instance
    if _registry_instance is None:
        with _registry_lock:
            if _registry_instance is None:
                _registry_instance = ToolRegistry()
                _registry_instance.load_dynamic_skills()
    return _registry_instance
