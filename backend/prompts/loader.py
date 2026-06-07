"""
CerebroForge (铸脑) — Prompt Template Loader
===============================================
Loads and renders Jinja2-based prompt templates from the templates/ directory.
Implements singleton pattern with template caching.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from jinja2 import BaseLoader, Environment, Template, TemplateNotFound

logger = logging.getLogger("cerebroforge.prompts.loader")


# ---------------------------------------------------------------------------
# Custom Jinja2 loader that reads from the templates/ directory
# ---------------------------------------------------------------------------

class _FileSystemLoader(BaseLoader):
    """Jinja2 template loader that reads from the templates/ directory."""

    def __init__(self, templates_dir: str | Path) -> None:
        self._templates_dir = Path(templates_dir)
        if not self._templates_dir.is_dir():
            raise FileNotFoundError(f"Templates directory not found: {self._templates_dir}")

    def get_source(
        self, environment: Environment, template: str
    ) -> tuple[str, str, callable]:
        """Load template source from disk."""
        # Normalize: append .md if not present
        if not template.endswith(".md"):
            template = f"{template}.md"

        path = self._templates_dir / template

        if not path.is_file():
            raise TemplateNotFound(template)

        source = path.read_text(encoding="utf-8")
        mtime = path.stat().st_mtime

        def _uptodate() -> bool:
            try:
                return path.stat().st_mtime == mtime
            except OSError:
                return False

        return source, str(path), _uptodate


# ---------------------------------------------------------------------------
# PromptLoader — Singleton with caching
# ---------------------------------------------------------------------------

class PromptLoader:
    """Load and render prompt templates from the templates/ directory.

    Features:
    - Jinja2-based template rendering with variable substitution
    - Template caching for performance
    - Singleton pattern for global access
    - Graceful handling of missing templates
    """

    _instance: Optional[PromptLoader] = None
    _lock = threading.Lock()

    def __new__(cls, *args: Any, **kwargs: Any) -> PromptLoader:
        """Ensure singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self, templates_dir: Optional[str | Path] = None) -> None:
        """Initialize the PromptLoader.

        Args:
            templates_dir: Path to the templates directory.
                           Defaults to the templates/ directory adjacent to this file.
        """
        if self._initialized:
            return

        if templates_dir is None:
            # Default: templates/ directory next to this file
            templates_dir = Path(__file__).parent / "templates"
        else:
            templates_dir = Path(templates_dir)

        self._templates_dir = templates_dir

        # Set up Jinja2 environment
        try:
            loader = _FileSystemLoader(templates_dir)
            self._env = Environment(
                loader=loader,
                keep_trailing_newline=True,
                trim_blocks=True,
                lstrip_blocks=True,
                undefined=self._make_undefined_handler(),
            )
        except FileNotFoundError:
            logger.warning("Templates directory not found: %s. PromptLoader will use fallback.", templates_dir)
            self._env = Environment(
                loader=None,
                keep_trailing_newline=True,
                trim_blocks=True,
                lstrip_blocks=True,
            )

        # Template cache
        self._cache: Dict[str, Template] = {}

        # List of known templates for validation
        self._known_templates: set[str] = {
            "manager",
            "tool_developer",
            "worker",
            "integrator",
            "critic",
            "clarify",
            "compress",
            "tool_cluster",
            "tool_merge",
        }

        self._initialized = True
        logger.info("PromptLoader initialized with templates dir: %s", templates_dir)

    @staticmethod
    def _make_undefined_handler() -> type:
        """Create a custom undefined handler that logs missing variables."""
        from jinja2 import Undefined

        class LoggingUndefined(Undefined):
            """Undefined variables render as empty string with a warning log."""

            def _fail_with_undefined_error(self) -> Any:
                logger.warning(
                    "Undefined template variable: %s",
                    self._undefined_name,
                )
                return ""

            def __str__(self) -> str:
                if self._undefined_name:
                    logger.warning(
                        "Undefined template variable: %s",
                        self._undefined_name,
                    )
                return ""

            def __iter__(self):
                return iter([])

            def __bool__(self) -> bool:
                return False

        return LoggingUndefined

    def get_prompt(self, template_name: str, **kwargs: Any) -> str:
        """Load and render a prompt template with the given variables.

        Args:
            template_name: Name of the template (without .md extension).
                           E.g., "manager", "worker", "critic".
            **kwargs: Template variables to substitute.

        Returns:
            Rendered prompt string.

        Raises:
            TemplateNotFound: If the template doesn't exist.
            ValueError: If template_name is invalid.
        """
        if not template_name or not isinstance(template_name, str):
            raise ValueError(f"Invalid template name: {template_name!r}")

        # Sanitize template name
        template_name = template_name.strip().replace("/", "").replace("\\", "")
        if not template_name:
            raise ValueError("Template name cannot be empty after sanitization.")

        # Get or load template
        template = self._load_template(template_name)

        if template is None:
            raise TemplateNotFound(template_name)

        # Render with variables
        try:
            rendered = template.render(**kwargs)
            return rendered
        except Exception as e:
            logger.error(
                "Failed to render template '%s' with kwargs %s: %s",
                template_name,
                list(kwargs.keys()),
                e,
            )
            raise

    def _load_template(self, template_name: str) -> Optional[Template]:
        """Load a template from cache or disk.

        Args:
            template_name: Name of the template.

        Returns:
            Jinja2 Template object or None if not found.
        """
        # Check cache first
        if template_name in self._cache:
            cached_template = self._cache[template_name]
            # Check if still up-to-date
            try:
                source, filename, uptodate = self._env.loader.get_source(
                    self._env, template_name
                )
                if uptodate():
                    return cached_template
            except (TemplateNotFound, AttributeError):
                return cached_template

        # Load from disk
        try:
            template = self._env.get_template(template_name)
            self._cache[template_name] = template
            logger.debug("Loaded and cached template: %s", template_name)
            return template
        except TemplateNotFound:
            logger.warning("Template not found: %s", template_name)
            return None
        except Exception as e:
            logger.error("Error loading template '%s': %s", template_name, e)
            return None

    def list_templates(self) -> list[str]:
        """List all available template names.

        Returns:
            List of template names (without .md extension).
        """
        templates: list[str] = []

        if self._templates_dir.is_dir():
            for path in sorted(self._templates_dir.glob("*.md")):
                templates.append(path.stem)

        return templates

    def get_raw_template(self, template_name: str) -> Optional[str]:
        """Get the raw template content without rendering.

        Args:
            template_name: Name of the template.

        Returns:
            Raw template string or None if not found.
        """
        # Try with and without .md
        for name in (template_name, f"{template_name}.md"):
            path = self._templates_dir / name
            if path.is_file():
                return path.read_text(encoding="utf-8")

        return None

    def reload_template(self, template_name: str) -> bool:
        """Force reload a template from disk (bypass cache).

        Args:
            template_name: Name of the template.

        Returns:
            True if reloaded successfully, False otherwise.
        """
        # Remove from cache
        self._cache.pop(template_name, None)

        # Reload
        template = self._load_template(template_name)
        return template is not None

    def clear_cache(self) -> None:
        """Clear the entire template cache."""
        self._cache.clear()
        logger.info("Template cache cleared.")

    def validate_template(self, template_name: str) -> Dict[str, Any]:
        """Validate a template for syntax errors.

        Args:
            template_name: Name of the template.

        Returns:
            Dict with 'valid' (bool) and 'error' (str or None).
        """
        raw = self.get_raw_template(template_name)
        if raw is None:
            return {"valid": False, "error": f"Template '{template_name}' not found."}

        try:
            self._env.parse(raw)
            return {"valid": True, "error": None}
        except Exception as e:
            return {"valid": False, "error": str(e)}

    def validate_all_templates(self) -> Dict[str, Dict[str, Any]]:
        """Validate all known templates.

        Returns:
            Dict mapping template names to validation results.
        """
        results: Dict[str, Dict[str, Any]] = {}
        for name in self.list_templates():
            results[name] = self.validate_template(name)
        return results


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def get_prompt_loader() -> PromptLoader:
    """Get the global PromptLoader singleton instance."""
    return PromptLoader()
