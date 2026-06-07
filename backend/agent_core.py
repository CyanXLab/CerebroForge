"""
CerebroForge (铸脑) - Agent Core
==================================
The main ZhuNaoAgent class that orchestrates the entire cognitive architecture.

Implements the full agent loop:
  1. Record real timestamp
  2. Check ambiguity via cognitive engine
  3. If ambiguous → generate clarification, return without executing
  4. Classify intent and select DAG
  5. Build 4±1 cognitive chunks
  6. Generate prior prediction
  7. Decide system mode (1 or 2)
  8. Run LangGraph workflow (cast-iron skeleton)
  9. Execute tools with prediction error tracking
 10. Validate with Pydantic + jsonschema + critic
 11. If prediction error ≥ 0.7, switch to System 2
 12. Record interaction with real timestamps
 13. Auto-compress L1 if threshold exceeded
 14. Light evolve (update EGL)
 15. Return structured response with cognitive_state, trace, memory_stats, evolution

Key constraints enforced:
- 4±1 cognitive chunks (never more than 5)
- Prediction before every action
- No guessing on ambiguous intent (must clarify)
- System 1 only when: high_freq AND error < 0.3 AND not ambiguous AND matches DAG node
- System 2 when: error ≥ 0.3 OR ambiguous OR tool failure
- Memory compression preserves index keys (reversible)
- Real timestamps on everything
- Max 2 new tool forges per task
"""

import time
import json
import re
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Flexible imports — work with both "from backend.xxx" and "from xxx" styles
try:
    from backend.config import (
        L1_MAX_ENTRIES, L1_MAX_TOKENS, PREDICTION_ERROR_SYS2,
        MAX_COGNITIVE_CHUNKS, MAX_TOOL_FORGE_PER_TASK, HIGH_FREQ_THRESHOLD,
        NVIDIA_BASE_URL, NVIDIA_API_KEY, DEFAULT_MODEL,
    )
except ImportError:
    from config import (
        L1_MAX_ENTRIES, L1_MAX_TOKENS, PREDICTION_ERROR_SYS2,
        MAX_COGNITIVE_CHUNKS, MAX_TOOL_FORGE_PER_TASK, HIGH_FREQ_THRESHOLD,
        NVIDIA_BASE_URL, NVIDIA_API_KEY, DEFAULT_MODEL,
    )

try:
    from backend.llm_client import LLMClient, llm
except ImportError:
    from llm_client import LLMClient, llm

try:
    from backend.memory import MemorySystem
except ImportError:
    from memory import MemorySystem

try:
    from backend.tools import ToolRegistry
except ImportError:
    from tools import ToolRegistry

try:
    from backend.schemas import (
        AgentDecision, AgentState, CognitiveChunk as SchemaCognitiveChunk,
        CognitiveState, SystemMode, MemoryLevel, MemoryOp,
        IntentCategory, ToolStatus, PriorPrediction, PredictionError,
        ToolDefinition, ExtendedToolCall, MemoryItem, MemoryStats,
        InteractionRecord, EvolutionMetrics, ClarificationPayload,
        AgentFullState, WorkflowState, ClarificationOutput,
        AGENT_DECISION_JSON_SCHEMA,
    )
except ImportError:
    from schemas import (
        AgentDecision, AgentState, CognitiveChunk as SchemaCognitiveChunk,
        CognitiveState, SystemMode, MemoryLevel, MemoryOp,
        IntentCategory, ToolStatus, PriorPrediction, PredictionError,
        ToolDefinition, ExtendedToolCall, MemoryItem, MemoryStats,
        InteractionRecord, EvolutionMetrics, ClarificationPayload,
        AgentFullState, WorkflowState, ClarificationOutput,
        AGENT_DECISION_JSON_SCHEMA,
    )

try:
    from backend.cognitive import CognitiveEngine
except ImportError:
    from cognitive import CognitiveEngine

try:
    from backend.skill_forge import SkillForge, get_skill_forge
except ImportError:
    from skill_forge import SkillForge, get_skill_forge

try:
    from backend.workflow import CerebroForgeWorkflow, create_workflow
except ImportError:
    from workflow import CerebroForgeWorkflow, create_workflow

import jsonschema

logger = logging.getLogger(__name__)


class ZhuNaoAgent:
    """
    The core cognitive agent for CerebroForge (铸脑).

    Orchestrates dual-process cognition, memory, tool execution,
    validation, and self-evolution in a single unified loop.
    """

    def __init__(self):
        """Initialize all subsystems."""
        # Core subsystems
        self.memory = MemorySystem()
        self.tools = ToolRegistry(memory=self.memory, llm_client=llm)
        self.cognitive = CognitiveEngine(llm=llm)
        self.skill_forge = get_skill_forge()

        # LangGraph workflow
        self._workflow = None
        self._workflow_error: Optional[str] = None
        try:
            self._workflow = create_workflow()
        except Exception as e:
            logger.error(f"LangGraph workflow creation failed: {e}")
            self._workflow_error = str(e)

        # Agent state tracking
        self._state = AgentFullState()
        self._interactions: List[Dict[str, Any]] = []
        self._evolution_metrics = EvolutionMetrics()
        self._clarification_store: Dict[str, ClarificationPayload] = {}
        self._task_forge_count: int = 0

        # Decision schema for validation
        self._decision_schema = AGENT_DECISION_JSON_SCHEMA

        logger.info("ZhuNaoAgent initialized")

    # ────────────────────────────────────────────────────────────────────
    # Main Entry Point
    # ────────────────────────────────────────────────────────────────────

    def process(self, user_query: str, stream: bool = False) -> Dict[str, Any]:
        """
        Main entry point for processing a user query.

        Implements the full 15-step cognitive pipeline.

        Args:
            user_query: The user's input query.
            stream: Whether to enable streaming (for SSE endpoint).

        Returns:
            Structured response dictionary.
        """
        # Step 1: Record real timestamp
        start_time = time.time()
        timestamp = datetime.now()

        trace: Dict[str, Any] = {
            "steps": [],
            "timestamps": {},
            "tool_results": [],
            "predictions": [],
            "errors": [],
        }
        trace["timestamps"]["start"] = timestamp.isoformat()

        # Step 2: Check ambiguity via cognitive engine
        is_ambiguous, ambiguity_issues = self.cognitive.check_ambiguity(user_query)
        trace["steps"].append({
            "step": "ambiguity_check",
            "result": "ambiguous" if is_ambiguous else "clear",
            "issues": ambiguity_issues if is_ambiguous else [],
        })
        trace["timestamps"]["ambiguity_check"] = datetime.now().isoformat()

        # Step 3: If ambiguous → generate clarification, return without executing
        if is_ambiguous and ambiguity_issues:
            clarification_data = self.cognitive.generate_clarification(user_query, ambiguity_issues)
            clarification_id = f"clarify_{int(time.time()*1000)}"

            questions = clarification_data.get("questions", [])
            suggested = clarification_data.get("suggested_answers", [])

            payload = ClarificationPayload(
                original_query=user_query,
                clarification_question=questions[0] if questions else "Could you please clarify your request?",
                suggested_answers=suggested,
            )
            self._clarification_store[clarification_id] = payload

            self._state.interaction_count += 1
            self._evolution_metrics.total_interactions += 1

            return {
                "response": payload.clarification_question,
                "needs_clarification": True,
                "clarification_id": clarification_id,
                "suggested_answers": payload.suggested_answers,
                "cognitive_state": self._state.model_dump(mode="json"),
                "trace": trace,
                "memory_stats": self.memory.get_memory_stats(),
                "evolution": self._evolution_metrics.model_dump(mode="json"),
            }

        # Step 4: Classify intent and select tools
        intent = self.cognitive.classify_intent(user_query)
        trace["steps"].append({"step": "intent_classification", "intent": intent})
        trace["timestamps"]["intent_classification"] = datetime.now().isoformat()

        # Step 5: Build 4±1 cognitive chunks
        context_items = self.memory.retrieve_relevant(user_query, top_k=3)
        chunks = self.cognitive.build_4_chunks(
            query=user_query,
            intent=intent,
            memories=context_items,
        )

        # Enforce 4±1 constraint strictly
        if len(chunks) > MAX_COGNITIVE_CHUNKS:
            chunks = chunks[:MAX_COGNITIVE_CHUNKS]

        trace["steps"].append({"step": "chunk_building", "chunk_count": len(chunks)})
        trace["timestamps"]["chunk_building"] = datetime.now().isoformat()

        # Step 6: Generate prior prediction
        prediction = self.cognitive.generate_prior_prediction(
            query=user_query,
            chunks=chunks,
            intent=intent,
        )
        trace["predictions"].append(prediction)
        trace["timestamps"]["prediction"] = datetime.now().isoformat()

        # Step 7: Decide system mode
        # Get tool usage stats for system mode decision
        tool_usage_stats = {}
        try:
            for tool_name in self.tools.get_tool_names():
                tool_info = self.tools.get_tool(tool_name)
                if tool_info:
                    tool_usage_stats[tool_name] = {
                        "usage_count": tool_info.get("usage_count", 0)
                        if isinstance(tool_info, dict) else 0,
                    }
        except Exception:
            pass

        prediction_error_val = prediction.get("prediction_error", 0.5) if isinstance(prediction, dict) else 0.5
        system_mode = self.cognitive.decide_system(
            prediction_error=prediction_error_val,
            is_ambiguous=is_ambiguous,
            tool_usage_stats=tool_usage_stats,
        )
        trace["steps"].append({"step": "mode_decision", "mode": system_mode})
        trace["timestamps"]["mode_decision"] = datetime.now().isoformat()

        # Step 8: Run LangGraph workflow (cast-iron skeleton)
        workflow_result = None
        workflow_error = None
        response_text = ""
        tools_used: List[str] = []

        if self._workflow:
            try:
                workflow_result = self._run_langgraph_workflow({
                    "query": user_query,
                    "system_mode": system_mode,
                    "cognitive_state": {
                        "system_active": system_mode,
                        "prediction_error": prediction_error_val,
                    },
                    "chunks": [c.to_dict() if hasattr(c, "to_dict") else c for c in chunks],
                    "prediction": json.dumps(prediction) if isinstance(prediction, dict) else str(prediction),
                })
                trace["steps"].append({"step": "langgraph_workflow", "status": "success"})
                trace["timestamps"]["workflow_complete"] = datetime.now().isoformat()

                # Extract response from workflow result
                if isinstance(workflow_result, dict):
                    response_text = workflow_result.get("final_response", "")
                    tools_used = workflow_result.get("selected_tools", [])
                    tool_results = workflow_result.get("tool_results", [])
                    for tr in tool_results:
                        trace["tool_results"].append(tr if isinstance(tr, dict) else str(tr))

            except Exception as e:
                workflow_error = str(e)
                trace["steps"].append({"step": "langgraph_workflow", "status": "failed", "error": workflow_error})
                trace["errors"].append(f"Workflow error: {workflow_error}")
                logger.error(f"LangGraph workflow failed: {e}")
        else:
            workflow_error = self._workflow_error or "Workflow not initialized"
            trace["steps"].append({"step": "langgraph_workflow", "status": "unavailable", "error": workflow_error})

        # Fallback: if workflow failed, use cognitive engine direct
        if workflow_error and not response_text:
            trace["steps"].append({"step": "cognitive_fallback", "triggered": True})
            try:
                cycle_result = self.cognitive.run_cognitive_cycle(user_query)
                if isinstance(cycle_result, dict):
                    response_text = cycle_result.get("response", "")
                    tools_used = cycle_result.get("tools_used", [])
                else:
                    response_text = str(cycle_result)
            except Exception as e:
                trace["errors"].append(f"Cognitive fallback failed: {e}")
                # Final fallback: direct LLM call
                try:
                    response_text = llm.chat(
                        messages=[
                            {"role": "system", "content": "You are ZhuNaoAgent (铸脑), a helpful AI assistant."},
                            {"role": "user", "content": user_query},
                        ],
                        temperature=0.5,
                    )
                except Exception as e2:
                    response_text = f"I apologize, but I encountered an error: {e2}"

        # Step 9: Compute prediction error
        actual_prediction_error = prediction_error_val
        if response_text and prediction:
            try:
                actual_prediction_error = self.cognitive.compute_prediction_error(
                    prediction=prediction,
                    actual=response_text,
                )
            except Exception:
                actual_prediction_error = prediction_error_val

        trace["predictions"].append({
            "type": "error_computation",
            "prediction_error": actual_prediction_error,
        })

        # Step 10: Validate with Pydantic + jsonschema + critic
        validation_result = self._validate_and_critic(response_text, user_query)
        trace["steps"].append({
            "step": "validation",
            "jsonschema_valid": validation_result.get("jsonschema_valid", False),
            "critic_score": validation_result.get("critic_score", 0.5),
        })
        trace["timestamps"]["validation"] = datetime.now().isoformat()

        # Step 11: If prediction error ≥ 0.7, switch to System 2
        if actual_prediction_error >= PREDICTION_ERROR_SYS2 and system_mode == "1":
            logger.info(f"Prediction error {actual_prediction_error:.2f} >= {PREDICTION_ERROR_SYS2}, switching to System 2")
            system_mode = "2"
            self._state.mode = "2"

            try:
                enhanced_response = llm.chat(
                    messages=[
                        {"role": "system", "content": "You are in deep-thinking mode. Use step-by-step reasoning. Be thorough and deliberate."},
                        {"role": "user", "content": f"Original query: {user_query}\n\nInitial response (may have errors): {response_text}\n\nPlease provide a more thorough and accurate response."},
                    ],
                    temperature=0.3,
                    max_tokens=2000,
                )
                response_text = enhanced_response
                trace["steps"].append({"step": "system2_switch", "reason": "high_prediction_error"})
            except Exception as e:
                trace["errors"].append(f"System 2 switch failed: {e}")

        # Step 12: Record interaction with real timestamps
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000

        try:
            interaction_id = self.memory.record_interaction(
                query=user_query,
                response=response_text,
                tools_used=tools_used,
                system_mode=system_mode,
                prediction_error=actual_prediction_error,
                duration_ms=duration_ms,
            )
        except Exception as e:
            logger.error(f"Failed to record interaction: {e}")
            interaction_id = f"int_{int(time.time()*1000)}"

        self._interactions.append({
            "id": interaction_id,
            "user_query": user_query,
            "agent_response": response_text[:500],
            "intent": intent,
            "system_mode": system_mode,
            "tools_used": tools_used,
            "prediction_error": actual_prediction_error,
            "chunks_built": len(chunks),
            "duration_ms": duration_ms,
            "timestamp": datetime.now().isoformat(),
        })
        self._state.interaction_count += 1

        # Store in memory
        try:
            self.memory.store_memory(
                level="L1",
                content=f"Q: {user_query}\nA: {response_text[:500]}",
                key=user_query[:50],
                metadata={"intent": intent, "system_mode": system_mode},
            )
        except Exception as e:
            logger.warning(f"Failed to store memory: {e}")

        trace["timestamps"]["end"] = datetime.now().isoformat()

        # Step 13: Auto-compress L1 if threshold exceeded
        try:
            if self.memory.should_compress_l1():
                compressed = self.memory.compress_l1()
                trace["steps"].append({"step": "auto_compress", "items_compressed": compressed})
        except Exception as e:
            logger.warning(f"Auto-compress check failed: {e}")

        # Step 14: Light evolve (update EGL)
        self._light_evolve()

        # Update state
        self._state.mode = system_mode
        self._state.avg_prediction_error = actual_prediction_error
        self._state.l1_entry_count = self.memory.get_memory_stats().get("L1_count", 0)
        self._state.l2_entry_count = self.memory.get_memory_stats().get("L2_count", 0)
        self._state.l3_entry_count = self.memory.get_memory_stats().get("L3_count", 0)

        # Compute system1 ratio
        total = self._state.interaction_count
        sys1_count = sum(1 for i in self._interactions if i.get("system_mode") == "1")
        self._state.system1_ratio = sys1_count / max(total, 1)

        # Reset task-level forge budget
        self._task_forge_count = 0

        # Step 15: Return structured response
        return {
            "response": response_text,
            "needs_clarification": False,
            "cognitive_state": {
                "mode": system_mode,
                "intent": intent,
                "chunks": len(chunks),
                "prediction_error": actual_prediction_error,
                "system_mode_reason": "high_freq_low_error" if system_mode == "1" else "deliberate_or_high_error",
            },
            "trace": trace,
            "memory_stats": self.memory.get_memory_stats(),
            "evolution": self._evolution_metrics.model_dump(mode="json"),
            "tools_used": tools_used,
            "duration_ms": duration_ms,
        }

    # ────────────────────────────────────────────────────────────────────
    # Validation: jsonschema + Critic
    # ────────────────────────────────────────────────────────────────────

    def _validate_and_critic(
        self,
        response_text: str,
        ground_truth: str,
    ) -> Dict[str, Any]:
        """
        Two-stage validation:
        1. jsonschema validation of structured decision
        2. Secondary critic prompt (LLM-based)

        Args:
            response_text: The agent's response to validate.
            ground_truth: The original user query.

        Returns:
            Validation result dict.
        """
        result = {
            "jsonschema_valid": True,
            "critic_score": 0.7,
            "jsonschema_errors": None,
            "critic_feedback": None,
        }

        # Stage 1: Try jsonschema validation if response is JSON
        try:
            data = json.loads(response_text)
            try:
                jsonschema.validate(instance=data, schema=self._decision_schema)
                result["jsonschema_valid"] = True
            except jsonschema.ValidationError as e:
                result["jsonschema_valid"] = False
                result["jsonschema_errors"] = e.message
        except (json.JSONDecodeError, ValueError):
            # Not JSON - that's fine, just skip schema validation
            result["jsonschema_valid"] = True  # Non-JSON responses are OK

        # Stage 2: Secondary critic prompt
        try:
            critic_prompt = f"""You are a quality critic for an AI agent's response. Evaluate the response quality.

Original Query: "{ground_truth}"
Agent Response: "{response_text[:1000]}"

Rate the response quality on these dimensions (0-1 each):
1. Relevance: Does it address the query?
2. Accuracy: Is it factually reasonable?
3. Completeness: Is it sufficiently thorough?
4. Clarity: Is it well-organized and clear?

Respond with JSON:
{{
    "relevance": 0.0-1.0,
    "accuracy": 0.0-1.0,
    "completeness": 0.0-1.0,
    "clarity": 0.0-1.0,
    "overall": 0.0-1.0,
    "feedback": "brief improvement suggestion"
}}"""

            critic_result = llm.chat_json(
                messages=[
                    {"role": "system", "content": "You are a response quality critic. Be objective and fair. Respond only with valid JSON."},
                    {"role": "user", "content": critic_prompt},
                ],
                temperature=0.1,
            )
            result["critic_score"] = critic_result.get("overall", 0.5)
            result["critic_feedback"] = critic_result.get("feedback", "")

        except Exception as e:
            logger.error(f"Critic validation failed: {e}")
            result["critic_score"] = 0.5  # Neutral on failure

        return result

    # ────────────────────────────────────────────────────────────────────
    # LangGraph Workflow Execution
    # ────────────────────────────────────────────────────────────────────

    def _run_langgraph_workflow(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the LangGraph compiled graph.

        Args:
            initial_state: Initial workflow state dictionary.

        Returns:
            Final state from the workflow execution.
        """
        if self._workflow is None:
            raise RuntimeError("LangGraph workflow not initialized")

        logger.info("Running LangGraph workflow...")
        result = self._workflow.run(initial_state)
        return result

    # ────────────────────────────────────────────────────────────────────
    # Evolution
    # ────────────────────────────────────────────────────────────────────

    def _light_evolve(self):
        """
        Update Evolutionary Growth Log (EGL) metrics.

        Lightweight evolution step that updates:
        - Average prediction error
        - System 1 ratio
        - Tool success rates
        - Intent distribution
        - EGL version
        """
        # Update average prediction error
        errors = [i.get("prediction_error", 0.5) for i in self._interactions]
        self._evolution_metrics.avg_prediction_error = sum(errors) / max(len(errors), 1)

        # Update system 1 ratio
        total = self._state.interaction_count
        sys1_count = sum(1 for i in self._interactions if i.get("system_mode") == "1")
        self._evolution_metrics.system1_ratio = sys1_count / max(total, 1)
        self._evolution_metrics.total_interactions = total

        # Update tool success rates
        try:
            for tool_info in self.memory.get_all_tools():
                name = tool_info.get("name", "")
                usage = tool_info.get("usage_count", 0)
                success = tool_info.get("success_count", 0)
                if usage > 0:
                    self._evolution_metrics.tool_success_rates[name] = success / usage
        except Exception:
            pass

        # Update intent distribution
        intent_dist: Dict[str, int] = {}
        for interaction in self._interactions:
            intent_val = interaction.get("intent", "conversation")
            intent_dist[intent_val] = intent_dist.get(intent_val, 0) + 1
        self._evolution_metrics.intent_distribution = intent_dist

        # Update forge/compression counts
        self._evolution_metrics.total_tools_forged = self.skill_forge._count_synthesized() if hasattr(self.skill_forge, '_count_synthesized') else 0
        try:
            evo_stats = self.memory.get_evolution_stats()
            self._evolution_metrics.total_compressions = evo_stats.get("total_events", 0)
        except Exception:
            pass

        # Bump EGL version
        self._evolution_metrics.egl_version += 0.001
        self._evolution_metrics.last_evolved_at = datetime.now().isoformat()

        # Log evolution event
        try:
            self.memory.log_evolution(
                event_type="light_evolve",
                description=f"EGL updated to v{self._evolution_metrics.egl_version:.3f}",
                egl_delta=0.001,
            )
        except Exception:
            pass

    # ────────────────────────────────────────────────────────────────────
    # Manual Operations
    # ────────────────────────────────────────────────────────────────────

    def force_compress(self) -> Dict[str, Any]:
        """
        Manual L1 compression trigger.

        Returns:
            Compression result with count of items compressed.
        """
        try:
            count = self.memory.compress_l1()
            self._evolution_metrics.total_compressions += 1
            self._state.last_compression = datetime.now().isoformat()
            return {
                "compressed_items": count,
                "l1_remaining": self.memory.get_memory_stats().get("L1_count", 0),
                "l2_count": self.memory.get_memory_stats().get("L2_count", 0),
                "l3_count": self.memory.get_memory_stats().get("L3_count", 0),
            }
        except Exception as e:
            return {"compressed_items": 0, "error": str(e)}

    # ────────────────────────────────────────────────────────────────────
    # State Access
    # ────────────────────────────────────────────────────────────────────

    def get_full_state(self) -> Dict[str, Any]:
        """
        Return complete agent state for API.
        """
        memory_stats = {}
        try:
            memory_stats = self.memory.get_memory_stats()
        except Exception:
            pass

        evolution_stats = {}
        try:
            evolution_stats = self.memory.get_evolution_stats()
        except Exception:
            pass

        tools_list = []
        try:
            for name in self.tools.get_tool_names():
                tool_info = self.tools.get_tool(name)
                if tool_info:
                    tools_list.append({
                        "name": name,
                        "doc": tool_info.get("doc", "") if isinstance(tool_info, dict) else str(tool_info),
                        "is_dynamic": tool_info.get("is_dynamic", False) if isinstance(tool_info, dict) else False,
                    })
        except Exception:
            pass

        return {
            "agent_state": self._state.model_dump(mode="json"),
            "memory_stats": memory_stats,
            "evolution": {
                **self._evolution_metrics.model_dump(mode="json"),
                **evolution_stats,
            },
            "tools": tools_list,
            "interactions_count": len(self._interactions),
            "recent_interactions": self._interactions[-10:],
            "llm_stats": llm.stats,
            "forge_stats": {
                "synthesized": self.skill_forge._count_synthesized() if hasattr(self.skill_forge, '_count_synthesized') else 0,
                "invocations": self.skill_forge._count_invocations() if hasattr(self.skill_forge, '_count_invocations') else 0,
                "should_evolve": self.skill_forge.should_evolve() if hasattr(self.skill_forge, 'should_evolve') else False,
            },
            "workflow_available": self._workflow is not None,
        }

    # ────────────────────────────────────────────────────────────────────
    # Clarification Handler
    # ────────────────────────────────────────────────────────────────────

    def get_clarify_response(self, orig: str, answers: List[str]) -> Dict[str, Any]:
        """
        Process clarification answers and re-run the agent.

        Args:
            orig: The original user query.
            answers: The user's answers to clarification questions.

        Returns:
            The result of re-processing with clarified intent.
        """
        clarified_query = f"{orig}\n\n[Clarification: {'; '.join(answers)}]"
        logger.info(f"Processing clarified query: {clarified_query[:100]}...")
        return self.process(clarified_query)
