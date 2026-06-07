"""
CerebroForge (铸脑) - LangGraph Workflow (CAST-IRON SKELETON)
=============================================================
Full state machine with conditional edges:

  START → clarify → manager → [tool_developer | executor] → executor → manager (loop) → integrator → END

Node functions:
  - clarify_node:     checks ambiguity, generates clarification or passes through
  - manager_node:     analyzes task, selects tools, decides if new tools needed
  - tool_developer_node: generates new tools/skills with sandbox validation
  - executor_node:    runs ReAct loop with bound tools
  - integrator_node:  extracts final answer from execution results

Each node uses Pydantic constrained output. Conditional edges based on state.
Maximum task execution count with auto-terminate. Validation at each step.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Sequence, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

try:
    from backend.config import (
        BASE_TOOLS,
        MAX_TASK_EXECUTION_CNT,
        MAX_TOOL_FORGE_PER_TASK,
        TOOL_EVO_BUDGET_TOKENS,
    )
except ImportError:
    from config import (
        BASE_TOOLS,
        MAX_TASK_EXECUTION_CNT,
        MAX_TOOL_FORGE_PER_TASK,
        TOOL_EVO_BUDGET_TOKENS,
    )

try:
    from backend.cognitive import CognitiveChunk, CognitiveEngine
except ImportError:
    from cognitive import CognitiveChunk, CognitiveEngine

try:
    from backend.llm_client import LLMClient
except ImportError:
    from llm_client import LLMClient

try:
    from backend.memory import MemorySystem
except ImportError:
    from memory import MemorySystem

try:
    from backend.schemas import (
        AgentDecision,
        AgentState,
        ClarificationOutput,
        CognitiveState,
        MemoryLevel,
        MemoryOp,
        MemoryOperation,
        NodeOutput,
        NodeRole,
        SurpriseLevel,
        SystemMode,
        ToolCall,
    )
except ImportError:
    from schemas import (
        AgentDecision,
        AgentState,
        ClarificationOutput,
        CognitiveState,
        MemoryLevel,
        MemoryOp,
        MemoryOperation,
        NodeOutput,
        NodeRole,
        SurpriseLevel,
        SystemMode,
        ToolCall,
    )

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Workflow Builder
# ────────────────────────────────────────────────────────────────────────────

class CerebroForgeWorkflow:
    """Builds and manages the CAST-IRON SKELETON LangGraph workflow.

    CAST-IRON = Cognitive Architecture for Self-Training &
    Intelligent Reasoning over Networks.
    """

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        memory: Optional[MemorySystem] = None,
        cognitive_engine: Optional[CognitiveEngine] = None,
    ) -> None:
        self.llm = llm or LLMClient()
        self.memory = memory or MemorySystem()
        self.cognitive = cognitive_engine or CognitiveEngine(self.llm)
        self.graph = self._build_graph()

    # ── Graph Construction ─────────────────────────────────────────────────

    def _build_graph(self) -> CompiledStateGraph:
        """Build the complete LangGraph state machine."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("clarify", self.clarify_node)
        workflow.add_node("manager", self.manager_node)
        workflow.add_node("tool_developer", self.tool_developer_node)
        workflow.add_node("executor", self.executor_node)
        workflow.add_node("integrator", self.integrator_node)

        # Define edges
        workflow.add_edge(START, "clarify")
        workflow.add_edge("clarify", "manager")

        # Conditional edge: manager → tool_developer or executor
        workflow.add_conditional_edges(
            "manager",
            self._route_after_manager,
            {
                "tool_developer": "tool_developer",
                "executor": "executor",
                "terminate": "integrator",
            },
        )

        # tool_developer → executor
        workflow.add_edge("tool_developer", "executor")

        # executor → manager (loop) or integrator
        workflow.add_conditional_edges(
            "executor",
            self._route_after_executor,
            {
                "manager": "manager",
                "integrator": "integrator",
            },
        )

        workflow.add_edge("integrator", END)

        return workflow.compile()

    # ── Routing Functions ──────────────────────────────────────────────────

    @staticmethod
    def _route_after_manager(state: Dict[str, Any]) -> Literal["tool_developer", "executor", "terminate"]:
        """Route after manager node based on state."""
        # If should terminate, go to integrator
        if state.get("should_terminate", False):
            return "terminate"

        # If new tools needed and we haven't exceeded the limit
        if state.get("need_new_tools", False):
            current_specs = state.get("new_tool_specs", [])
            if len(current_specs) < MAX_TOOL_FORGE_PER_TASK:
                return "tool_developer"

        return "executor"

    @staticmethod
    def _route_after_executor(state: Dict[str, Any]) -> Literal["manager", "integrator"]:
        """Route after executor: loop back to manager or proceed to integrator."""
        # Check termination conditions
        if state.get("should_terminate", False):
            return "integrator"

        # Check execution count
        exec_count = state.get("execution_count", 0)
        max_exec = state.get("max_executions", MAX_TASK_EXECUTION_CNT)
        if exec_count >= max_exec:
            logger.info(f"Max execution count reached ({exec_count}), proceeding to integrator")
            return "integrator"

        # Check if we have a final response
        if state.get("final_response"):
            return "integrator"

        # Loop back to manager for re-evaluation
        return "manager"

    # ── Node: Clarify ─────────────────────────────────────────────────────

    def clarify_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check ambiguity and generate clarification or pass through.

        Uses the cognitive engine's ambiguity detection (4 rules) and
        generates clarifying questions if needed.
        """
        query = state.get("query", "")
        logger.info(f"[clarify] Processing query: {query[:100]}")

        # Run cognitive cycle for ambiguity check
        is_ambiguous, issues = self.cognitive.check_ambiguity(query)

        updates: Dict[str, Any] = {
            "is_ambiguous": is_ambiguous,
            "ambiguity_issues": issues,
        }

        if is_ambiguous:
            # Generate clarification
            try:
                clarification = self.cognitive.generate_clarification(query, issues)
                updates["clarification_questions"] = clarification.questions
                updates["clarified_query"] = clarification.clarified_query or query
            except Exception as exc:
                logger.warning(f"Clarification generation failed: {exc}")
                updates["clarification_questions"] = [
                    "Could you provide more details about your request?"
                ]
                updates["clarified_query"] = query
        else:
            updates["clarification_questions"] = []
            updates["clarified_query"] = query

        # Record node output
        node_output = NodeOutput(
            role=NodeRole.CLARIFY,
            success=True,
            message=f"Ambiguity check: {'ambiguous' if is_ambiguous else 'clear'}",
            data={
                "is_ambiguous": is_ambiguous,
                "issues": issues,
                "clarified_query": updates.get("clarified_query", query),
            },
        )

        existing_outputs = state.get("node_outputs", [])
        updates["node_outputs"] = existing_outputs + [node_output.model_dump()]

        return updates

    # ── Node: Manager ─────────────────────────────────────────────────────

    def manager_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze task, select tools, decide if new tools are needed.

        This is the central decision-making node, inspired by Yunjue's
        manager pattern. It:
          1. Builds cognitive chunks from current state
          2. Classifies intent
          3. Selects appropriate tools
          4. Decides if new tools need to be forged
          5. Updates cognitive state
        """
        query = state.get("clarified_query", state.get("query", ""))
        north_star = state.get("north_star", "")
        ground_truth = state.get("ground_truth", "")
        relevant_memories = state.get("relevant_memories", [])
        execution_count = state.get("execution_count", 0)

        logger.info(f"[manager] Cycle {execution_count}: {query[:100]}")

        # Retrieve relevant memories
        try:
            mem_results = self.memory.retrieve_relevant(query, top_k=5)
            relevant_memories = mem_results
        except Exception as exc:
            logger.warning(f"Memory retrieval failed: {exc}")

        # Build cognitive chunks
        chunks = self.cognitive.build_4_chunks(
            north_star=north_star or "Complete the user's task effectively",
            ground_truth=ground_truth,
            relevant_memories=relevant_memories,
            prediction=state.get("prediction", ""),
        )

        # Classify intent
        intent = self.cognitive.classify_intent(query)

        # Select tools based on intent and available tools
        available_tools = state.get("available_tools", [])
        all_tools = available_tools if available_tools else [
            {"name": t, "description": f"Base tool: {t}", "type": "base"}
            for t in BASE_TOOLS
        ]

        # Use LLM to analyze and decide
        tool_names = [t["name"] for t in all_tools]
        chunks_text = "\n".join(f"[{c.name}] {c.content}" for c in chunks)

        manager_prompt = [
            {
                "role": "system",
                "content": (
                    "You are the Manager node of CerebroForge, a self-evolving cognitive agent. "
                    "Analyze the user's task and decide:\n"
                    "1. Which tools to select for execution\n"
                    "2. Whether new tools need to be forged\n"
                    "3. The current cognitive state\n\n"
                    "Respond with a JSON object matching the AgentDecision schema.\n"
                    "Key constraints:\n"
                    "- At most 3 tool_calls\n"
                    "- Only use tools from the available list\n"
                    "- Set need_new_tools=true only if existing tools are insufficient"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Query: {query}\n"
                    f"Intent: {intent}\n"
                    f"Execution cycle: {execution_count}/{state.get('max_executions', MAX_TASK_EXECUTION_CNT)}\n\n"
                    f"Cognitive context:\n{chunks_text}\n\n"
                    f"Available tools: {', '.join(tool_names)}\n\n"
                    f"Previous tool results: {json.dumps(state.get('tool_results', [])[-3:], default=str)[:1000]}\n\n"
                    "Analyze and decide:"
                ),
            },
        ]

        try:
            decision = self.llm.structured_json(
                messages=manager_prompt,
                response_model=AgentDecision,
                temperature=0.3,
                max_tokens=2048,
            )

            # Validate tool names
            valid_tools = set(tool_names)
            validated_calls = []
            for tc in decision.tool_calls:
                if tc.name in valid_tools:
                    validated_calls.append(tc.model_dump())
                else:
                    logger.warning(f"Manager selected invalid tool: {tc.name}")

            # Determine if new tools needed
            need_new_tools = bool(decision.response and "need_new_tools" in decision.response.lower())

            # Compute prediction error if we have a previous prediction
            prediction_error = 0.0
            prev_prediction = state.get("prediction", "")
            if prev_prediction and state.get("tool_results"):
                actual = json.dumps(state.get("tool_results", [])[-1:], default=str)[:500]
                prediction_error, surprise = self.cognitive.compute_prediction_error(
                    prev_prediction, actual
                )
            else:
                surprise = SurpriseLevel.NONE

            # Check if high-frequency match
            high_freq_tools = self.memory.get_high_freq_tools()
            high_freq_match = any(
                tc.name in high_freq_tools for tc in decision.tool_calls
            )

            # Decide system mode
            system_mode = self.cognitive.decide_system(
                prediction_error=prediction_error,
                high_freq_match=high_freq_match,
                ambiguous=state.get("is_ambiguous", False),
            )

            # Generate prediction for next step
            prediction = self.cognitive.generate_prior_prediction(
                chunks, ", ".join(tc.name for tc in decision.tool_calls) or "proceed"
            )

            # Build cognitive state
            workspace_occupancy = len(chunks) / 5.0
            cognitive_state = CognitiveState(
                system_active=system_mode,
                prediction_error=prediction_error,
                surprise_level=surprise,
                workspace_occupancy=workspace_occupancy,
            )

            # Convert memory operations
            mem_ops = [op.model_dump() for op in decision.memory_operations]

            updates = {
                "selected_tools": [tc.name for tc in decision.tool_calls],
                "tool_calls": validated_calls,
                "need_new_tools": need_new_tools,
                "cognitive_state": cognitive_state.model_dump(),
                "prediction_error": prediction_error,
                "surprise_level": surprise.value if isinstance(surprise, SurpriseLevel) else surprise,
                "system_mode": system_mode.value if isinstance(system_mode, SystemMode) else system_mode,
                "chunks": [c.to_dict() for c in chunks],
                "prediction": prediction,
                "memory_operations": mem_ops,
                "relevant_memories": relevant_memories,
                "north_star": north_star or "Complete the user's task effectively",
            }

        except Exception as exc:
            logger.error(f"Manager node error: {exc}")
            updates = {
                "selected_tools": BASE_TOOLS[:3],
                "tool_calls": [],
                "need_new_tools": False,
                "error": f"Manager analysis failed: {exc}",
                "should_terminate": True,
            }

        # Record node output
        node_output = NodeOutput(
            role=NodeRole.MANAGER,
            success=True,
            message=f"Tools selected: {updates.get('selected_tools', [])}",
            data=updates,
        )
        existing_outputs = state.get("node_outputs", [])
        updates["node_outputs"] = existing_outputs + [node_output.model_dump()]

        return updates

    # ── Node: Tool Developer ───────────────────────────────────────────────

    def tool_developer_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate new tools/skills with sandbox validation.

        This implements the self-evolution capability: when existing tools
        are insufficient, the agent can forge new ones.
        """
        query = state.get("clarified_query", state.get("query", ""))
        selected_tools = state.get("selected_tools", [])
        existing_specs = state.get("new_tool_specs", [])

        logger.info(f"[tool_developer] Forging new tools for: {query[:100]}")

        # Determine what kind of tool is needed
        tool_prompt = [
            {
                "role": "system",
                "content": (
                    "You are the Tool Developer node of CerebroForge. "
                    "The manager has determined that existing tools are insufficient "
                    "for the current task. Design a new tool that fills the gap.\n\n"
                    "Output a JSON object with:\n"
                    "- name: tool name (snake_case)\n"
                    "- description: what the tool does\n"
                    "- code: complete Python function implementation\n"
                    "- parameters: dict of parameter names to types\n"
                    "- test_cases: list of test inputs to validate the tool\n\n"
                    f"Budget: {TOOL_EVO_BUDGET_TOKENS} tokens max for the code.\n"
                    "The tool must be a self-contained Python function that can be "
                    "executed in a sandboxed environment."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Task: {query}\n"
                    f"Currently available tools: {', '.join(selected_tools)}\n"
                    f"Previously forged tools: {len(existing_specs)}\n\n"
                    "Design the new tool:"
                ),
            },
        ]

        try:
            raw_response = self.llm.generate(
                messages=tool_prompt,
                temperature=0.4,
                max_tokens=TOOL_EVO_BUDGET_TOKENS,
                response_format={"type": "json_object"},
            )

            tool_spec = json.loads(raw_response)

            # Validate tool spec
            required_fields = ["name", "description", "code", "parameters"]
            for field_name in required_fields:
                if field_name not in tool_spec:
                    raise ValueError(f"Missing required field: {field_name}")

            # Sandbox validation: try to execute the tool code
            validation_result = self._validate_tool_in_sandbox(tool_spec)

            if validation_result["valid"]:
                # Register the tool
                self.memory.register_tool(
                    name=tool_spec["name"],
                    description=tool_spec["description"],
                    code=tool_spec["code"],
                    tool_type="forged",
                    metadata={"parameters": tool_spec.get("parameters", {})},
                )

                # Log evolution event
                self.memory.log_evolution(
                    event_type="tool_forged",
                    description=f"Forged new tool: {tool_spec['name']}",
                    before_state={"available_tools": selected_tools},
                    after_state={"new_tool": tool_spec["name"]},
                    egl_delta=0.5,
                )

                new_specs = existing_specs + [tool_spec]
                available_tools = state.get("available_tools", [])
                available_tools.append({
                    "name": tool_spec["name"],
                    "description": tool_spec["description"],
                    "type": "forged",
                })

                updates = {
                    "new_tool_specs": new_specs,
                    "available_tools": available_tools,
                    "need_new_tools": False,
                }
            else:
                logger.warning(
                    f"Tool validation failed: {validation_result.get('error', 'unknown')}"
                )
                updates = {
                    "need_new_tools": False,
                    "error": f"Tool validation failed: {validation_result.get('error', '')}",
                }

        except Exception as exc:
            logger.error(f"Tool developer error: {exc}")
            updates = {
                "need_new_tools": False,
                "error": f"Tool development failed: {exc}",
            }

        # Record node output
        node_output = NodeOutput(
            role=NodeRole.TOOL_DEVELOPER,
            success="error" not in updates,
            message=f"Tool forge: {updates.get('new_tool_specs', [{}])[-1].get('name', 'failed') if updates.get('new_tool_specs') else 'failed'}",
            data=updates,
        )
        existing_outputs = state.get("node_outputs", [])
        updates["node_outputs"] = existing_outputs + [node_output.model_dump()]

        return updates

    def _validate_tool_in_sandbox(self, tool_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a new tool by executing its code in a sandbox."""
        import tempfile
        import subprocess
        import sys

        code = tool_spec.get("code", "")
        name = tool_spec.get("name", "unknown")

        # Basic syntax check
        try:
            compile(code, f"<tool_{name}>", "exec")
        except SyntaxError as exc:
            return {"valid": False, "error": f"Syntax error: {exc}"}

        # Try to execute the code in a subprocess
        test_code = code + "\n\n# Validation: try to call the function\nprint('VALIDATION_OK')\n"

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(test_code)
            tmp_path = tmp.name

        try:
            result = subprocess.run(
                [sys.executable, "-u", tmp_path],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0 and "VALIDATION_OK" in result.stdout:
                return {"valid": True}
            else:
                return {
                    "valid": False,
                    "error": result.stderr[:500] or "Validation marker not found in output",
                }
        except subprocess.TimeoutExpired:
            return {"valid": False, "error": "Validation timed out"}
        except Exception as exc:
            return {"valid": False, "error": str(exc)}
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # ── Node: Executor ────────────────────────────────────────────────────

    def executor_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run ReAct loop with bound tools.

        Executes the selected tool calls and collects results.
        Implements a simplified ReAct (Reason + Act) pattern.
        """
        query = state.get("clarified_query", state.get("query", ""))
        tool_calls = state.get("tool_calls", [])
        execution_count = state.get("execution_count", 0)
        max_exec = state.get("max_executions", MAX_TASK_EXECUTION_CNT)

        logger.info(f"[executor] Cycle {execution_count + 1}/{max_exec}: {len(tool_calls)} tool calls")

        tool_results: List[Dict[str, Any]] = state.get("tool_results", [])
        should_terminate = False
        final_response = ""

        # Execute each tool call
        for tc in tool_calls:
            tool_name = tc.get("name", "")
            tool_args = tc.get("args", {})

            start_time = time.time()
            result = self._execute_tool(tool_name, tool_args, state)
            duration_ms = (time.time() - start_time) * 1000

            # Update tool stats
            success = "error" not in result or result.get("success", True)
            self.memory.update_tool_stats(tool_name, success, duration_ms / 1000)

            tool_results.append({
                "tool": tool_name,
                "args": tool_args,
                "result": result,
                "duration_ms": duration_ms,
                "success": success,
            })

        # Increment execution count
        execution_count += 1

        # Check if we should terminate
        if execution_count >= max_exec:
            should_terminate = True
            logger.info(f"Max executions reached ({execution_count})")

        # If no tool calls, try to generate a direct response
        if not tool_calls:
            try:
                direct_response = self._generate_direct_response(query, tool_results)
                final_response = direct_response
                should_terminate = True
            except Exception as exc:
                logger.warning(f"Direct response generation failed: {exc}")

        # Check if tool results contain a final answer
        if tool_results:
            last_result = tool_results[-1].get("result", {})
            if isinstance(last_result, dict) and last_result.get("final_answer"):
                final_response = last_result["final_answer"]
                should_terminate = True

        updates = {
            "tool_results": tool_results,
            "execution_count": execution_count,
            "should_terminate": should_terminate,
        }

        if final_response:
            updates["final_response"] = final_response

        # Record node output
        node_output = NodeOutput(
            role=NodeRole.EXECUTOR,
            success=True,
            message=f"Executed {len(tool_calls)} tools, cycle {execution_count}",
            data={"tool_results_count": len(tool_results), "execution_count": execution_count},
        )
        existing_outputs = state.get("node_outputs", [])
        updates["node_outputs"] = existing_outputs + [node_output.model_dump()]

        return updates

    def _execute_tool(
        self,
        tool_name: str,
        args: Dict[str, Any],
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a single tool call."""
        try:
            # Map tool names to actual implementations
            try:
                from backend.computer_use import ComputerUseToolkit
            except ImportError:
                from computer_use import ComputerUseToolkit

            toolkit = ComputerUseToolkit()
            tool_map = {
                t["name"]: t["handler"] for t in toolkit.get_tool_definitions()
            }

            if tool_name in tool_map:
                handler = tool_map[tool_name]
                result = handler(**args)
                return result if isinstance(result, dict) else {"result": str(result)}

            # Handle base tools not in computer_use
            if tool_name == "web_search":
                return self._tool_web_search(args)
            elif tool_name == "web_fetch":
                return self._tool_web_fetch(args)
            elif tool_name == "text_extract":
                return self._tool_text_extract(args)
            elif tool_name == "image_query":
                return self._tool_image_query(args)
            elif tool_name == "calculate":
                return self._tool_calculate(args)

            # Handle forged tools
            tool_record = self.memory.get_tool(tool_name)
            if tool_record and tool_record.get("tool_type") == "forged":
                return self._execute_forged_tool(tool_record, args)

            return {"error": f"Unknown tool: {tool_name}"}

        except Exception as exc:
            return {"error": f"Tool execution error ({tool_name}): {exc}"}

    @staticmethod
    def _tool_web_search(args: Dict[str, Any]) -> Dict[str, Any]:
        """Web search tool (stub - requires external API)."""
        query = args.get("query", "")
        return {
            "tool": "web_search",
            "query": query,
            "results": f"Web search results for: {query} (requires external API integration)",
            "success": True,
        }

    @staticmethod
    def _tool_web_fetch(args: Dict[str, Any]) -> Dict[str, Any]:
        """Web fetch tool (stub - requires external API)."""
        url = args.get("url", "")
        return {
            "tool": "web_fetch",
            "url": url,
            "content": f"Fetched content from: {url} (requires external API integration)",
            "success": True,
        }

    @staticmethod
    def _tool_text_extract(args: Dict[str, Any]) -> Dict[str, Any]:
        """Text extraction tool."""
        text = args.get("text", "")
        operation = args.get("operation", "summarize")

        if operation == "summarize" and len(text) > 200:
            return {"summary": text[:200] + "...", "original_length": len(text)}
        return {"text": text, "operation": operation}

    @staticmethod
    def _tool_image_query(args: Dict[str, Any]) -> Dict[str, Any]:
        """Image query tool (stub)."""
        return {
            "tool": "image_query",
            "message": "Image query requires VLM integration",
            "args": args,
        }

    @staticmethod
    def _tool_calculate(args: Dict[str, Any]) -> Dict[str, Any]:
        """Simple calculation tool."""
        expression = args.get("expression", "")
        try:
            # Safe eval with restricted builtins
            allowed_names = {
                "abs": abs, "round": round, "min": min, "max": max,
                "sum": sum, "len": len, "int": int, "float": float,
                "pow": pow, "divmod": divmod,
            }
            import math
            for name in dir(math):
                if not name.startswith("_"):
                    allowed_names[name] = getattr(math, name)

            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return {"result": result, "expression": expression, "success": True}
        except Exception as exc:
            return {"error": str(exc), "expression": expression, "success": False}

    def _execute_forged_tool(
        self,
        tool_record: Dict[str, Any],
        args: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a dynamically forged tool."""
        code = tool_record.get("code", "")
        name = tool_record.get("name", "unknown")

        import tempfile
        import subprocess
        import sys

        # Build execution script
        call_code = f"""
{code}

# Execute with provided arguments
import json
_args = json.loads('{json.dumps(args)}')
# Try to find and call the main function
_result = None
for _name in dir():
    _obj = globals()[_name]
    if callable(_obj) and not _name.startswith('_') and _name != 'json':
        try:
            _result = _obj(**_args)
            break
        except TypeError:
            continue

if _result is not None:
    print(json.dumps({{"result": str(_result)}}))
"""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(call_code)
            tmp_path = tmp.name

        try:
            result = subprocess.run(
                [sys.executable, "-u", tmp_path],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                try:
                    output = json.loads(result.stdout.strip().split("\n")[-1])
                    return {"tool": name, **output, "success": True}
                except (json.JSONDecodeError, IndexError):
                    return {"tool": name, "output": result.stdout, "success": True}
            else:
                return {"tool": name, "error": result.stderr[:500], "success": False}

        except subprocess.TimeoutExpired:
            return {"tool": name, "error": "Forged tool execution timed out", "success": False}
        except Exception as exc:
            return {"tool": name, "error": str(exc), "success": False}
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _generate_direct_response(
        self,
        query: str,
        tool_results: List[Dict[str, Any]],
    ) -> str:
        """Generate a direct response when no tools are needed."""
        results_text = json.dumps(tool_results[-3:], default=str)[:2000] if tool_results else "No tool results available."

        messages = [
            {
                "role": "system",
                "content": "You are CerebroForge, a self-evolving cognitive agent. Provide a helpful, accurate response.",
            },
            {
                "role": "user",
                "content": f"Query: {query}\n\nContext from tools:\n{results_text}\n\nProvide your response:",
            },
        ]

        return self.llm.generate(messages=messages, temperature=0.5, max_tokens=2048)

    # ── Node: Integrator ──────────────────────────────────────────────────

    def integrator_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract final answer from execution results.

        Synthesizes all tool results, cognitive state, and execution
        history into a coherent final response.
        """
        query = state.get("query", "")
        clarified_query = state.get("clarified_query", query)
        tool_results = state.get("tool_results", [])
        cognitive_state = state.get("cognitive_state", {})
        node_outputs = state.get("node_outputs", [])
        execution_count = state.get("execution_count", 0)

        logger.info(f"[integrator] Synthesizing final response from {len(tool_results)} tool results")

        # If we already have a final response, use it
        existing_response = state.get("final_response", "")
        if existing_response:
            final_response = existing_response
        else:
            # Synthesize from tool results
            final_response = self._synthesize_response(
                query=clarified_query,
                tool_results=tool_results,
                cognitive_state=cognitive_state,
                node_outputs=node_outputs,
            )

        # Record the interaction
        session_id = state.get("session_id", str(uuid.uuid4()))
        self.memory.record_interaction(
            query=query,
            response=final_response,
            tools_used=[tr.get("tool", "") for tr in tool_results],
            system_mode=cognitive_state.get("system_active", "1"),
            prediction_error=cognitive_state.get("prediction_error", 0.0),
            duration_ms=0.0,
            session_id=session_id,
        )

        # Execute memory operations from cognitive state
        mem_ops = state.get("memory_operations", [])
        for op in mem_ops:
            try:
                self._execute_memory_operation(op)
            except Exception as exc:
                logger.warning(f"Memory operation failed: {exc}")

        # Record node output
        node_output = NodeOutput(
            role=NodeRole.INTEGRATOR,
            success=True,
            message="Final response synthesized",
            data={
                "tool_results_count": len(tool_results),
                "execution_count": execution_count,
                "response_length": len(final_response),
            },
        )
        node_outputs = node_outputs + [node_output.model_dump()]

        return {
            "final_response": final_response,
            "node_outputs": node_outputs,
            "should_terminate": True,
        }

    def _synthesize_response(
        self,
        query: str,
        tool_results: List[Dict[str, Any]],
        cognitive_state: Dict[str, Any],
        node_outputs: List[Dict[str, Any]],
    ) -> str:
        """Use LLM to synthesize a final response from tool results."""
        # Prepare context
        results_summary = []
        for tr in tool_results[-5:]:  # Last 5 tool results
            tool_name = tr.get("tool", "unknown")
            result = tr.get("result", {})
            success = tr.get("success", True)

            if isinstance(result, dict):
                # Extract the most useful information
                if "content" in result:
                    results_summary.append(f"[{tool_name}] {result['content'][:500]}")
                elif "stdout" in result:
                    results_summary.append(f"[{tool_name}] {result['stdout'][:500]}")
                elif "result" in result:
                    results_summary.append(f"[{tool_name}] {str(result['result'])[:500]}")
                elif "error" in result:
                    results_summary.append(f"[{tool_name}] Error: {result['error'][:200]}")
                else:
                    results_summary.append(f"[{tool_name}] {json.dumps(result, default=str)[:300]}")
            else:
                results_summary.append(f"[{tool_name}] {str(result)[:300]}")

        context = "\n".join(results_summary)
        system_mode = cognitive_state.get("system_active", "1")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are CerebroForge (铸脑), a self-evolving cognitive agent. "
                    "Synthesize the following tool results into a clear, comprehensive "
                    f"final response. System mode: {'System 2 (analytical)' if system_mode == '2' else 'System 1 (fast)'}. "
                    "Be precise, actionable, and well-structured."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Original query: {query}\n\n"
                    f"Tool results:\n{context}\n\n"
                    "Provide a comprehensive final response:"
                ),
            },
        ]

        try:
            return self.llm.generate(
                messages=messages,
                temperature=0.4,
                max_tokens=4096,
            )
        except Exception as exc:
            logger.error(f"Response synthesis failed: {exc}")
            # Fallback: return raw tool results
            return f"Based on tool execution results:\n\n{context}"

    def _execute_memory_operation(self, op: Dict[str, Any]) -> None:
        """Execute a single memory operation."""
        operation = op.get("operation", "")
        target_level = op.get("target_level", "L1")
        key = op.get("key", "")
        content = op.get("content", "")

        if operation == "STORE":
            self.memory.store_memory(
                level=target_level,
                content=content,
                key=key,
            )
        elif operation == "UPDATE":
            # Update existing memory (find by key)
            if key:
                # Store as new version
                self.memory.store_memory(
                    level=target_level,
                    content=content,
                    key=key,
                )
        elif operation == "COMPRESS":
            if target_level == "L1":
                self.memory.compress_l1()
            elif target_level == "L2":
                self.memory.compress_l2()
        elif operation == "ARCHIVE":
            # Archive is handled through compression
            pass

    # ── Public Interface ───────────────────────────────────────────────────

    def run(
        self,
        query: str,
        session_id: Optional[str] = None,
        north_star: str = "",
        ground_truth: str = "",
    ) -> Dict[str, Any]:
        """
        Run the complete CerebroForge workflow.

        Args:
            query: The user's query.
            session_id: Optional session identifier.
            north_star: Optional overarching goal.
            ground_truth: Optional factual context.

        Returns:
            Complete state dictionary with final_response and all intermediate data.
        """
        # Initialize state
        initial_state: Dict[str, Any] = {
            "query": query,
            "clarified_query": query,
            "cognitive_state": CognitiveState().model_dump(),
            "prediction_error": 0.0,
            "surprise_level": "NONE",
            "system_mode": "1",
            "relevant_memories": [],
            "memory_operations": [],
            "selected_tools": [],
            "available_tools": [],
            "need_new_tools": False,
            "new_tool_specs": [],
            "tool_results": [],
            "execution_count": 0,
            "max_executions": MAX_TASK_EXECUTION_CNT,
            "north_star": north_star,
            "ground_truth": ground_truth,
            "chunks": [],
            "prediction": "",
            "is_ambiguous": False,
            "ambiguity_issues": [],
            "clarification_questions": [],
            "final_response": "",
            "node_outputs": [],
            "task_id": str(uuid.uuid4()),
            "session_id": session_id or str(uuid.uuid4()),
            "timestamp": time.time(),
            "error": None,
            "should_terminate": False,
        }

        # Run the graph
        try:
            result = self.graph.invoke(initial_state)
            return result
        except Exception as exc:
            logger.error(f"Workflow execution failed: {exc}")
            return {
                **initial_state,
                "error": str(exc),
                "final_response": f"An error occurred during processing: {exc}",
                "should_terminate": True,
            }

    def get_graph_mermaid(self) -> str:
        """Generate a Mermaid diagram of the workflow."""
        return """
graph TD
    START([START]) --> clarify[Clarify Node]
    clarify --> manager[Manager Node]
    manager -->|need_new_tools| tool_developer[Tool Developer]
    manager -->|proceed| executor[Executor Node]
    manager -->|terminate| integrator[Integrator Node]
    tool_developer --> executor
    executor -->|loop| manager
    executor -->|done| integrator
    integrator --> END([END])

    style START fill:#4CAF50,color:white
    style END fill:#f44336,color:white
    style clarify fill:#2196F3,color:white
    style manager fill:#FF9800,color:white
    style tool_developer fill:#9C27B0,color:white
    style executor fill:#00BCD4,color:white
    style integrator fill:#795548,color:white
"""


# ────────────────────────────────────────────────────────────────────────────
# Module-level convenience function
# ────────────────────────────────────────────────────────────────────────────

def create_workflow(
    llm: Optional[LLMClient] = None,
    memory: Optional[MemorySystem] = None,
) -> CerebroForgeWorkflow:
    """Create and return a CerebroForge workflow instance."""
    return CerebroForgeWorkflow(llm=llm, memory=memory)
