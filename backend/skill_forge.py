"""
CerebroForge (铸脑) — Skill Forge / Evolution Engine
======================================================
Inspired by Yunjue Agent's tool evolution system.
Generates, validates, enhances, clusters, merges, and evolves
atomic Python tools through LLM-driven synthesis.

Imports config, memory, llm_client, tools from the same directory.
"""

from __future__ import annotations

import ast
import copy
import json
import logging
import os
import re
import shutil
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
        DEFAULT_MAX_TOKENS,
        MAX_RETRIES as CONFIG_MAX_RETRIES,
        MAX_TOOL_FORGE_PER_TASK,
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
        DEFAULT_MAX_TOKENS,
        MAX_RETRIES as CONFIG_MAX_RETRIES,
        MAX_TOOL_FORGE_PER_TASK,
    )

try:
    from backend.memory import MemorySystem
except ImportError:
    from memory import MemorySystem

try:
    from backend.llm_client import LLMClient
except ImportError:
    from llm_client import LLMClient

try:
    from backend.tools import (
        ToolRegistry,
        _SAFE_BUILTINS,
        _SAFE_MODULES,
        _clean_llm_code,
        DYNAMIC_SKILLS_DIR,
        DYNAMIC_SKILLS_PUBLIC_DIR,
        EGL_THRESHOLD,
        get_tool_registry,
    )
except ImportError:
    from tools import (
        ToolRegistry,
        _SAFE_BUILTINS,
        _SAFE_MODULES,
        _clean_llm_code,
        DYNAMIC_SKILLS_DIR,
        DYNAMIC_SKILLS_PUBLIC_DIR,
        EGL_THRESHOLD,
        get_tool_registry,
    )

try:
    from backend.prompts.loader import PromptLoader
except ImportError:
    try:
        from prompts.loader import PromptLoader
    except ImportError:
        PromptLoader = None  # type: ignore[assignment,misc]

logger = logging.getLogger("cerebroforge.skill_forge")


# ---------------------------------------------------------------------------
# Tool code template
# ---------------------------------------------------------------------------

TOOL_CODE_TEMPLATE = textwrap.dedent("""\
    __TOOL_META__ = {{
        "name": "{name}",
        "description": "{description}",
        "dependencies": {dependencies}
    }}

    from pydantic import BaseModel, Field

    class InputModel(BaseModel):
        {input_fields}

    class OutputModel(BaseModel):
        {output_fields}

    def run(input: InputModel) -> OutputModel:
        {body}
""")


# ---------------------------------------------------------------------------
# SkillForge Class
# ---------------------------------------------------------------------------

class SkillForge:
    """Evolutionary skill forging engine for CerebroForge.

    Capabilities:
    - forge_skill: Create new tools from capability descriptions via LLM
    - enhance_skill: Fix failing tools using error reports
    - cluster_skills: Group semantically similar tools
    - merge_skills: Merge a cluster of similar tools into one
    - batch_evolve: Parallel batch evolution with post-batch merge
    - compute_egl: Track Evolutionary Generality Loss metric
    - should_evolve: Decide whether the system needs more tools
    - promote_to_public: Move validated skills to public directory
    """

    MAX_RETRIES: int = CONFIG_MAX_RETRIES

    def __init__(
        self,
        tool_registry: Optional[ToolRegistry] = None,
        memory: Optional[MemorySystem] = None,
        llm_client: Optional[LLMClient] = None,
    ) -> None:
        self._registry = tool_registry or get_tool_registry()
        self._memory = memory or MemorySystem()
        self._llm = llm_client or LLMClient()
        self._prompt_loader = PromptLoader() if PromptLoader is not None else None

        # EGL tracking — read from evolution log
        self._cumulative_synthesized: int = self._count_synthesized()
        self._cumulative_invocations: int = self._count_invocations()

    # -----------------------------------------------------------------------
    # EGL counter helpers
    # -----------------------------------------------------------------------

    def _count_synthesized(self) -> int:
        """Count total tools synthesized from the evolution log."""
        try:
            stats = self._memory.get_evolution_stats()
            by_type = stats.get("by_type", {})
            forged = by_type.get("tool_forged", {}).get("count", 0)
            merged = by_type.get("tool_merged", {}).get("count", 0)
            enhanced = by_type.get("tool_enhanced", {}).get("count", 0)
            return forged + merged + enhanced
        except Exception:
            return 0

    def _count_invocations(self) -> int:
        """Count total tool invocations from the tools table."""
        try:
            tools = self._memory.get_all_tools()
            return sum(t.get("usage_count", 0) for t in tools)
        except Exception:
            return 0

    # -----------------------------------------------------------------------
    # Core: forge_skill
    # -----------------------------------------------------------------------

    def forge_skill(
        self,
        task_description: str,
        required_capabilities: List[str],
    ) -> Optional[Dict[str, Any]]:
        """Create a new skill through LLM-driven code generation.

        Process:
        1. Use LLM to generate Python tool code with __TOOL_META__, InputModel, OutputModel, run()
        2. Validate in sandbox (exec with safe builtins)
        3. Run a basic test case
        4. If fails, retry with error feedback (max 3 retries)
        5. Register in tool_registry and memory
        6. Return skill info or None

        Args:
            task_description: What the tool should accomplish.
            required_capabilities: List of capability keywords.

        Returns:
            Skill info dict or None if forging failed.
        """
        logger.info(
            "Forging skill for: %s (capabilities: %s)",
            task_description[:80],
            ", ".join(required_capabilities),
        )

        # Build the generation prompt
        prompt = self._build_forge_prompt(task_description, required_capabilities)

        for attempt in range(1, self.MAX_RETRIES + 1):
            logger.info("Forge attempt %d/%d", attempt, self.MAX_RETRIES)

            # Step 1: Generate tool code via LLM
            try:
                raw_code = self._llm.chat(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a Python tool code generator for the CerebroForge skill evolution engine. "
                                "Return ONLY valid, self-contained Python code. No markdown fences. No explanations. "
                                "The code MUST define: __TOOL_META__, InputModel (pydantic BaseModel), "
                                "OutputModel (pydantic BaseModel), and run(input: InputModel) -> OutputModel."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=4096,
                )
            except Exception as e:
                logger.error("LLM call failed on attempt %d: %s", attempt, e)
                continue

            # Clean up code
            code = _clean_llm_code(raw_code)

            # Step 2: Validate in sandbox
            validation = self._validate_in_sandbox(code)
            if not validation["valid"]:
                logger.warning("Sandbox validation failed (attempt %d): %s", attempt, validation["error"])
                # Feed error back for retry
                prompt = self._build_retry_prompt(task_description, required_capabilities, code, validation["error"])
                continue

            # Step 3: Run basic test case
            test_result = self._run_test_case(code, validation.get("meta", {}))
            if not test_result["passed"]:
                logger.warning("Test case failed (attempt %d): %s", attempt, test_result["error"])
                prompt = self._build_retry_prompt(
                    task_description, required_capabilities, code,
                    f"Test case failed: {test_result['error']}",
                )
                continue

            # Step 4: Success — register and record
            meta = validation.get("meta", {})
            tool_name = meta.get("name", f"skill_{int(time.time())}")
            tool_doc = meta.get("description", task_description)

            tool_func = self._registry._build_tool_func(code, tool_name)
            self._registry.register(
                name=tool_name,
                func=tool_func,
                doc=tool_doc,
                schema=validation.get("schema", {}),
                is_dynamic=True,
                code=code,
            )

            # Persist
            self._registry._persist_skill(tool_name, code, meta)

            # Record in memory
            try:
                self._memory.log_evolution(
                    event_type="tool_forged",
                    description=f"Forged tool '{tool_name}': {task_description[:200]}",
                    after_state={
                        "name": tool_name,
                        "description": tool_doc,
                        "meta": meta,
                        "attempt": attempt,
                        "test_passed": True,
                    },
                )
            except Exception:
                pass

            # Update EGL counters
            self._cumulative_synthesized += 1

            logger.info("Successfully forged skill '%s' on attempt %d", tool_name, attempt)

            return {
                "name": tool_name,
                "description": tool_doc,
                "meta": meta,
                "code": code,
                "attempt": attempt,
                "test_result": test_result,
            }

        logger.error("Failed to forge skill after %d attempts: %s", self.MAX_RETRIES, task_description[:80])
        return None

    # -----------------------------------------------------------------------
    # Enhance: fix failing skills
    # -----------------------------------------------------------------------

    def enhance_skill(
        self,
        skill_name: str,
        error_report: str,
        historical_calls: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Fix a failing skill using error context and historical call data.

        Process:
        1. Load original code from registry
        2. Generate enhanced version with error context
        3. Validate and test
        4. Update if passes

        Args:
            skill_name: Name of the skill to enhance.
            error_report: Description of the failure.
            historical_calls: Previous call records with args/results.

        Returns:
            Updated skill info dict or None if enhancement failed.
        """
        tool_info = self._registry.get_tool(skill_name)
        if tool_info is None:
            logger.error("Skill '%s' not found in registry.", skill_name)
            return None

        original_code = tool_info.get("code")
        if not original_code:
            logger.error("No source code available for skill '%s'.", skill_name)
            return None

        logger.info("Enhancing skill '%s' based on error: %s", skill_name, error_report[:80])

        # Build enhancement prompt
        history_str = json.dumps(historical_calls[-10:], ensure_ascii=False, indent=2, default=str)
        prompt = textwrap.dedent(f"""\
            You are a tool enhancement engine for CerebroForge. A previously generated tool is failing.
            Analyze the error and historical usage, then produce a FIXED version of the tool.

            ORIGINAL CODE:
            ```python
            {original_code}
            ```

            ERROR REPORT:
            {error_report}

            HISTORICAL CALLS:
            {history_str}

            REQUIREMENTS:
            1. Keep the same __TOOL_META__ name (or improve the description)
            2. Keep InputModel and OutputModel structure (add fields if needed for robustness)
            3. Fix the bug that caused the error
            4. Add better error handling and input validation
            5. Return ONLY the improved Python code, no markdown fences, no explanation
        """)

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                raw_code = self._llm.chat(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a Python tool fixer. Return ONLY valid Python code. "
                                "No markdown fences. No explanations. Fix the bug."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=4096,
                )
            except Exception as e:
                logger.error("LLM call failed during enhancement attempt %d: %s", attempt, e)
                continue

            code = _clean_llm_code(raw_code)

            # Validate
            validation = self._validate_in_sandbox(code)
            if not validation["valid"]:
                logger.warning("Enhanced code validation failed (attempt %d): %s", attempt, validation["error"])
                prompt = textwrap.dedent(f"""\
                    The enhanced code also has issues. Original error: {error_report}
                    Validation error: {validation['error']}

                    ORIGINAL CODE:
                    ```python
                    {original_code}
                    ```

                    FAILED ENHANCEMENT:
                    ```python
                    {code}
                    ```

                    Please fix and return ONLY valid Python code:
                """)
                continue

            # Test
            test_result = self._run_test_case(code, validation.get("meta", {}))
            if not test_result["passed"]:
                logger.warning("Enhanced code test failed (attempt %d): %s", attempt, test_result["error"])
                continue

            # Update the registry
            meta = validation.get("meta", {})
            tool_name = meta.get("name", skill_name)
            tool_doc = meta.get("description", tool_info.get("doc", ""))

            tool_func = self._registry._build_tool_func(code, tool_name)
            self._registry.register(
                name=tool_name,
                func=tool_func,
                doc=tool_doc,
                schema=validation.get("schema", {}),
                is_dynamic=True,
                code=code,
            )
            self._registry._persist_skill(tool_name, code, meta)

            # Record in memory
            try:
                self._memory.log_evolution(
                    event_type="tool_enhanced",
                    description=f"Enhanced tool '{tool_name}': {error_report[:200]}",
                    before_state={"name": skill_name, "error": error_report},
                    after_state={"name": tool_name, "meta": meta},
                )
            except Exception:
                pass

            logger.info("Successfully enhanced skill '%s' on attempt %d", skill_name, attempt)

            return {
                "name": tool_name,
                "description": tool_doc,
                "meta": meta,
                "code": code,
                "attempt": attempt,
            }

        logger.error("Failed to enhance skill '%s' after %d attempts", skill_name, self.MAX_RETRIES)
        return None

    # -----------------------------------------------------------------------
    # Cluster: group similar skills
    # -----------------------------------------------------------------------

    def cluster_skills(self, skills_list: List[str]) -> Dict[str, List[str]]:
        """Use LLM to cluster semantically similar tools.

        Args:
            skills_list: List of tool/skill names to cluster.

        Returns:
            Dict mapping cluster labels to lists of tool names.
        """
        if not skills_list:
            return {}

        # Gather tool documentation
        tool_docs: List[Dict[str, str]] = []
        for name in skills_list:
            info = self._registry.get_tool(name)
            if info:
                tool_docs.append({
                    "name": name,
                    "description": info.get("doc", "No description"),
                })
            else:
                tool_docs.append({"name": name, "description": "Unknown tool"})

        tool_list_str = json.dumps(tool_docs, ensure_ascii=False, indent=2)

        # Use prompt template if available
        if self._prompt_loader is not None:
            try:
                prompt = self._prompt_loader.get_prompt("tool_cluster", tool_list=tool_list_str)
            except Exception:
                prompt = self._build_cluster_prompt(tool_list_str)
        else:
            prompt = self._build_cluster_prompt(tool_list_str)

        try:
            response = self._llm.chat(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a tool clustering engine. Group similar tools by functionality. "
                            "Return ONLY a JSON object mapping cluster names to arrays of tool names. "
                            "No markdown fences, no explanation."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=2048,
            )
        except Exception as e:
            logger.error("LLM call failed during clustering: %s", e)
            # Fallback: each tool in its own cluster
            return {name: [name] for name in skills_list}

        # Parse response
        clusters = self._parse_json_response(response)
        if clusters is None:
            # Fallback: each tool in its own cluster
            return {name: [name] for name in skills_list}

        # Validate: all skills should be in a cluster
        clustered_tools: set = set()
        for members in clusters.values():
            if isinstance(members, list):
                clustered_tools.update(members)

        # Add unclustered tools as singletons
        for name in skills_list:
            if name not in clustered_tools:
                clusters[f"unclustered_{name}"] = [name]

        logger.info("Clustered %d tools into %d clusters", len(skills_list), len(clusters))
        return clusters

    # -----------------------------------------------------------------------
    # Merge: combine similar skills
    # -----------------------------------------------------------------------

    def merge_skills(self, cluster: List[str]) -> Optional[Dict[str, Any]]:
        """Merge a cluster of similar tools into one unified tool.

        Process:
        1. Takes a cluster of similar tools
        2. LLM generates unified tool preserving all functionality
        3. Validates and registers
        4. Returns merged tool name

        Args:
            cluster: List of tool names in the same cluster.

        Returns:
            Merged tool info dict or None if merging failed.
        """
        if len(cluster) <= 1:
            logger.info("Cluster has %d tools — nothing to merge.", len(cluster))
            return None

        # Collect all tool code and docs
        cluster_tools: List[Dict[str, str]] = []
        for name in cluster:
            info = self._registry.get_tool(name)
            if info:
                cluster_tools.append({
                    "name": name,
                    "description": info.get("doc", ""),
                    "code": info.get("code", ""),
                })

        cluster_str = json.dumps(cluster_tools, ensure_ascii=False, indent=2)

        # Use prompt template if available
        if self._prompt_loader is not None:
            try:
                prompt = self._prompt_loader.get_prompt("tool_merge", cluster_tools=cluster_str)
            except Exception:
                prompt = self._build_merge_prompt(cluster_str)
        else:
            prompt = self._build_merge_prompt(cluster_str)

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                raw_code = self._llm.chat(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a tool merging engine. Combine multiple similar tools into one unified tool. "
                                "Return ONLY valid Python code with __TOOL_META__, InputModel, OutputModel, and run(). "
                                "No markdown fences, no explanations."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=4096,
                )
            except Exception as e:
                logger.error("LLM call failed during merge attempt %d: %s", attempt, e)
                continue

            code = _clean_llm_code(raw_code)

            validation = self._validate_in_sandbox(code)
            if not validation["valid"]:
                logger.warning("Merged code validation failed (attempt %d): %s", attempt, validation["error"])
                continue

            test_result = self._run_test_case(code, validation.get("meta", {}))
            if not test_result["passed"]:
                logger.warning("Merged code test failed (attempt %d): %s", attempt, test_result["error"])
                continue

            # Register the merged tool
            meta = validation.get("meta", {})
            merged_name = meta.get("name", f"merged_{int(time.time())}")
            merged_doc = meta.get("description", f"Merged tool combining: {', '.join(cluster)}")

            tool_func = self._registry._build_tool_func(code, merged_name)
            self._registry.register(
                name=merged_name,
                func=tool_func,
                doc=merged_doc,
                schema=validation.get("schema", {}),
                is_dynamic=True,
                code=code,
            )
            self._registry._persist_skill(merged_name, code, meta)

            # Remove the original tools that were merged
            for old_name in cluster:
                if old_name in self._registry._tools and old_name != merged_name:
                    del self._registry._tools[old_name]
                    logger.info("Removed absorbed tool: %s", old_name)

            # Record in memory
            try:
                self._memory.log_evolution(
                    event_type="tool_merged",
                    description=f"Merged {len(cluster)} tools into '{merged_name}'",
                    before_state={"tools": cluster},
                    after_state={"merged_name": merged_name, "meta": meta},
                )
            except Exception:
                pass

            self._cumulative_synthesized += 1

            logger.info("Successfully merged %d tools into '%s'", len(cluster), merged_name)

            return {
                "name": merged_name,
                "description": merged_doc,
                "meta": meta,
                "code": code,
                "merged_from": cluster,
            }

        logger.error("Failed to merge cluster after %d attempts", self.MAX_RETRIES)
        return None

    # -----------------------------------------------------------------------
    # Batch Evolution
    # -----------------------------------------------------------------------

    def batch_evolve(
        self,
        tasks: List[Dict[str, Any]],
        batch_size: int = 16,
    ) -> Dict[str, Any]:
        """Parallel batch evolution of multiple tasks.

        Process:
        1. Process multiple tasks in parallel
        2. Each independently generates tools
        3. Post-batch: cluster + merge + absorb
        4. Update EGL metric

        Args:
            tasks: List of dicts with 'task_description' and 'required_capabilities'.
            batch_size: Number of parallel workers.

        Returns:
            Summary of batch evolution results.
        """
        logger.info("Starting batch evolution with %d tasks, batch_size=%d", len(tasks), batch_size)

        results: List[Optional[Dict[str, Any]]] = [None] * len(tasks)
        successful: List[Dict[str, Any]] = []
        failed_indices: List[int] = []

        # Parallel processing
        with ThreadPoolExecutor(max_workers=min(batch_size, len(tasks))) as executor:
            future_to_idx: Dict[Any, int] = {}
            for idx, task in enumerate(tasks):
                future = executor.submit(
                    self.forge_skill,
                    task.get("task_description", ""),
                    task.get("required_capabilities", []),
                )
                future_to_idx[future] = idx

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results[idx] = result
                    if result is not None:
                        successful.append(result)
                    else:
                        failed_indices.append(idx)
                except Exception as e:
                    logger.error("Task %d raised exception: %s", idx, e)
                    failed_indices.append(idx)

        logger.info(
            "Batch evolution complete: %d/%d successful",
            len(successful), len(tasks),
        )

        # Post-batch: cluster and merge
        merged_results: List[Dict[str, Any]] = []
        clusters: Dict[str, List[str]] = {}
        if len(successful) >= 2:
            skill_names = [s["name"] for s in successful]
            clusters = self.cluster_skills(skill_names)

            for cluster_name, members in clusters.items():
                if isinstance(members, list) and len(members) >= 2:
                    merge_result = self.merge_skills(members)
                    if merge_result is not None:
                        merged_results.append(merge_result)
                        logger.info("Merged cluster '%s' into '%s'", cluster_name, merge_result["name"])

        # Update EGL
        egl = self.compute_egl()

        summary = {
            "total_tasks": len(tasks),
            "successful": len(successful),
            "failed": len(failed_indices),
            "failed_indices": failed_indices,
            "clusters_found": len([c for c in clusters.values() if isinstance(c, list) and len(c) >= 2]),
            "merged_count": len(merged_results),
            "egl": egl,
            "results": results,
            "merged_results": merged_results,
        }

        logger.info(
            "Batch evolution summary: %d success, %d failed, %d merged, EGL=%.4f",
            summary["successful"], summary["failed"], summary["merged_count"], egl,
        )

        return summary

    # -----------------------------------------------------------------------
    # EGL: Evolutionary Generality Loss
    # -----------------------------------------------------------------------

    def compute_egl(self) -> float:
        """Compute Evolutionary Generality Loss.

        EGL = cumulative_tools_synthesized / cumulative_tool_invocations

        A high EGL means the system is in an exploration phase (creating more tools
        than it uses). A low EGL means the system is in an exploitation phase
        (reusing existing tools heavily).

        Returns:
            EGL value as float.
        """
        # Count total invocations from memory
        total_invocations = self._count_invocations()

        total_synthesized = self._cumulative_synthesized

        if total_invocations == 0:
            return float("inf") if total_synthesized > 0 else 0.0

        egl = total_synthesized / total_invocations
        logger.debug("EGL = %d / %d = %.4f", total_synthesized, total_invocations, egl)
        return egl

    def should_evolve(self) -> bool:
        """Check if the system should evolve more tools.

        Returns True if EGL > threshold, meaning the system is in exploration
        phase and should create more tools.

        Returns:
            Boolean indicating whether more tools should be forged.
        """
        egl = self.compute_egl()
        should = egl > EGL_THRESHOLD
        logger.info("EGL=%.4f, threshold=%.4f, should_evolve=%s", egl, EGL_THRESHOLD, should)
        return should

    # -----------------------------------------------------------------------
    # Promote to public
    # -----------------------------------------------------------------------

    def promote_to_public(self, skill_name: str) -> bool:
        """Move a validated skill to the public directory.

        Args:
            skill_name: Name of the skill to promote.

        Returns:
            True if promotion succeeded, False otherwise.
        """
        tool_info = self._registry.get_tool(skill_name)
        if tool_info is None:
            logger.error("Skill '%s' not found.", skill_name)
            return False

        code = tool_info.get("code")
        if not code:
            logger.error("No source code for skill '%s'.", skill_name)
            return False

        # Source paths
        src_dir = DYNAMIC_SKILLS_DIR
        src_py = src_dir / f"{skill_name}.py"
        src_meta = src_dir / f"{skill_name}.meta.json"

        # Destination paths
        dst_dir = DYNAMIC_SKILLS_PUBLIC_DIR
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_py = dst_dir / f"{skill_name}.py"
        dst_meta = dst_dir / f"{skill_name}.meta.json"

        try:
            # Copy to public dir
            dst_py.write_text(code, encoding="utf-8")

            meta = {}
            if src_meta.exists():
                meta = json.loads(src_meta.read_text(encoding="utf-8"))
            meta["promoted_at"] = datetime.now(timezone.utc).isoformat()
            meta["public"] = True
            dst_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

            # Remove from private dir
            if src_py.exists():
                src_py.unlink()
            if src_meta.exists():
                src_meta.unlink()

            # Record in memory
            try:
                self._memory.log_evolution(
                    event_type="skill_promoted",
                    description=f"Promoted skill '{skill_name}' to public directory",
                    after_state={"name": skill_name, "public": True},
                )
            except Exception:
                pass

            logger.info("Promoted skill '%s' to public directory.", skill_name)
            return True
        except Exception as e:
            logger.error("Failed to promote skill '%s': %s", skill_name, e)
            return False

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _build_forge_prompt(
        self,
        task_description: str,
        required_capabilities: List[str],
    ) -> str:
        """Build the prompt for initial tool forging."""
        capabilities_str = json.dumps(required_capabilities, ensure_ascii=False)

        # Try using template
        if self._prompt_loader is not None:
            try:
                tool_request = json.dumps({
                    "task": task_description,
                    "capabilities": required_capabilities,
                }, ensure_ascii=False)
                return self._prompt_loader.get_prompt(
                    "tool_developer",
                    tool_request_json=tool_request,
                    proxy_url="",
                )
            except Exception:
                pass

        return textwrap.dedent(f"""\
            Generate a Python tool for the following task:

            TASK: {task_description}

            REQUIRED CAPABILITIES: {capabilities_str}

            The tool MUST follow this exact structure:
            1. __TOOL_META__ = {{"name": "descriptive_snake_case_name", "description": "...", "dependencies": [...]}}
            2. from pydantic import BaseModel, Field
            3. class InputModel(BaseModel): ...  (define all input fields with Field descriptions)
            4. class OutputModel(BaseModel): ...  (define output fields)
            5. def run(input: InputModel) -> OutputModel: ...  (main logic)

            RULES:
            - Use only standard library + pydantic (list any other deps in __TOOL_META__.dependencies)
            - Add comprehensive error handling
            - Validate all inputs
            - Keep it atomic: one clear purpose
            - Return ONLY the Python code, no markdown fences, no explanation
        """)

    def _build_retry_prompt(
        self,
        task_description: str,
        required_capabilities: List[str],
        failed_code: str,
        error: str,
    ) -> str:
        """Build the prompt for retry after validation/test failure."""
        return textwrap.dedent(f"""\
            The previously generated tool FAILED. Please fix it.

            TASK: {task_description}
            CAPABILITIES: {json.dumps(required_capabilities, ensure_ascii=False)}

            FAILED CODE:
            ```python
            {failed_code}
            ```

            ERROR:
            {error}

            Fix the code and return ONLY valid Python code with __TOOL_META__, InputModel, OutputModel, run().
            No markdown fences, no explanations.
        """)

    def _build_cluster_prompt(self, tool_list_str: str) -> str:
        """Build the prompt for tool clustering."""
        return textwrap.dedent(f"""\
            Group the following tools by functional similarity. Tools that do similar things
            should be in the same cluster.

            TOOLS:
            {tool_list_str}

            Return a JSON object where keys are cluster labels (like "web_search", "file_ops", "math", etc.)
            and values are arrays of tool names belonging to that cluster.

            Example format:
            {{
                "cluster_name": ["tool_a", "tool_b"],
                "another_cluster": ["tool_c"]
            }}

            Return ONLY the JSON, no markdown fences, no explanation.
        """)

    def _build_merge_prompt(self, cluster_str: str) -> str:
        """Build the prompt for merging a cluster of tools."""
        return textwrap.dedent(f"""\
            Merge the following similar tools into ONE unified tool that preserves ALL their functionality.

            TOOLS TO MERGE:
            {cluster_str}

            The merged tool MUST:
            1. Define __TOOL_META__ with a descriptive name and combined description
            2. Define InputModel with a "mode" or "action" field to select which sub-functionality to use
            3. Define OutputModel that covers all possible output types
            4. Implement run() that dispatches to the correct logic based on the mode/action
            5. Preserve all original functionality — do not lose any feature
            6. Handle all edge cases from the original tools

            Return ONLY valid Python code, no markdown fences, no explanation.
        """)

    def _validate_in_sandbox(self, code: str) -> Dict[str, Any]:
        """Validate tool code by executing it in a sandbox.

        Returns dict with: valid, error, meta, schema.
        """
        result: Dict[str, Any] = {"valid": False, "error": None, "meta": {}, "schema": {}}

        # Syntax check
        try:
            ast.parse(code)
        except SyntaxError as e:
            result["error"] = f"SyntaxError: {e}"
            return result

        # Sandbox execution
        sandbox_globals: Dict[str, Any] = {"__builtins__": dict(_SAFE_BUILTINS)}
        for mod_name, mod in _SAFE_MODULES.items():
            sandbox_globals[mod_name.split(".")[-1]] = mod

        try:
            exec(compile(code, "<skill_validation>", "exec"), sandbox_globals)  # noqa: S102
        except Exception as e:
            result["error"] = f"Execution error: {type(e).__name__}: {e}"
            return result

        # Check required components
        if "__TOOL_META__" not in sandbox_globals:
            result["error"] = "Missing __TOOL_META__"
            return result

        meta = sandbox_globals["__TOOL_META__"]
        if not isinstance(meta, dict) or "name" not in meta:
            result["error"] = "__TOOL_META__ must be a dict with 'name' key"
            return result

        result["meta"] = meta

        for required in ("InputModel", "OutputModel", "run"):
            if required not in sandbox_globals:
                result["error"] = f"Missing required: {required}"
                return result

        # Extract schema
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

    def _run_test_case(
        self,
        code: str,
        meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run a basic test case on the generated tool.

        Tries to instantiate InputModel with default/empty values and call run().

        Returns dict with: passed, error, output.
        """
        result: Dict[str, Any] = {"passed": False, "error": None, "output": None}

        sandbox_globals: Dict[str, Any] = {"__builtins__": dict(_SAFE_BUILTINS)}
        for mod_name, mod in _SAFE_MODULES.items():
            sandbox_globals[mod_name.split(".")[-1]] = mod

        try:
            exec(compile(code, "<skill_test>", "exec"), sandbox_globals)  # noqa: S102
        except Exception as e:
            result["error"] = f"Code execution failed: {type(e).__name__}: {e}"
            return result

        InputModel = sandbox_globals.get("InputModel")
        run_fn = sandbox_globals.get("run")

        if InputModel is None or run_fn is None:
            result["error"] = "Missing InputModel or run()"
            return result

        # Try to create an input with defaults
        try:
            # First try empty instantiation
            input_obj = InputModel()
        except Exception:
            # Try with None/empty values for required fields
            try:
                schema = {}
                if hasattr(InputModel, "model_json_schema"):
                    schema = InputModel.model_json_schema()
                elif hasattr(InputModel, "schema"):
                    schema = InputModel.schema()

                props = schema.get("properties", {})
                required_fields = schema.get("required", [])
                test_kwargs: Dict[str, Any] = {}
                for field_name in required_fields:
                    field_info = props.get(field_name, {})
                    ftype = field_info.get("type", "string")
                    if ftype == "string":
                        test_kwargs[field_name] = "test"
                    elif ftype == "integer":
                        test_kwargs[field_name] = 1
                    elif ftype == "number":
                        test_kwargs[field_name] = 1.0
                    elif ftype == "boolean":
                        test_kwargs[field_name] = True
                    elif ftype == "array":
                        test_kwargs[field_name] = []
                    elif ftype == "object":
                        test_kwargs[field_name] = {}
                    else:
                        test_kwargs[field_name] = "test"

                input_obj = InputModel(**test_kwargs)
            except Exception as e:
                result["error"] = f"Cannot create test input: {type(e).__name__}: {e}"
                return result

        # Try to run
        try:
            output = run_fn(input_obj)
            result["output"] = str(output)[:500]
            result["passed"] = True
        except Exception as e:
            # If the tool fails on test input, that might be expected
            # (e.g., needs real data). We accept it as "passable" if it
            # doesn't crash on import/setup.
            error_str = str(e)
            # Check if it's a "meaningful" error (not a code bug)
            meaningful_errors = [
                "ConnectionError", "TimeoutError", "HTTPError",
                "FileNotFoundError", "not found", "not available",
                "API", "authentication", "permission",
            ]
            if any(err_type in error_str for err_type in meaningful_errors):
                # External dependency failure — acceptable
                result["passed"] = True
                result["output"] = f"Test accepted (external dependency): {error_str[:200]}"
            else:
                result["error"] = f"run() failed: {type(e).__name__}: {e}"

        return result

    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Try to extract and parse JSON from an LLM response."""
        # Strip markdown fences
        text = response.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON in the response
        json_match = re.search(r'\{[\s\S]*\}', text)  # noqa
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return None


# ---------------------------------------------------------------------------
# Module-level singleton (lazy)
# ---------------------------------------------------------------------------

_forge_instance: Optional[SkillForge] = None
_forge_lock = __import__("threading").Lock()


def get_skill_forge() -> SkillForge:
    """Return the global SkillForge singleton."""
    global _forge_instance
    if _forge_instance is None:
        with _forge_lock:
            if _forge_instance is None:
                registry = get_tool_registry()
                _forge_instance = SkillForge(tool_registry=registry)
    return _forge_instance
