"""
CerebroForge (铸脑) - Pydantic Schemas & Data Models
====================================================
Complete type definitions for the cognitive agent framework.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field, field_validator


# ────────────────────────────────────────────────────────────────────────────
# Enumerations
# ────────────────────────────────────────────────────────────────────────────

class SystemMode(str, Enum):
    """Dual-process system selection inspired by Kahneman."""
    SYSTEM_1 = "1"  # Fast, pattern-matched, heuristic
    SYSTEM_2 = "2"  # Slow, deliberate, analytical


class SurpriseLevel(str, Enum):
    """How surprised the agent is by the current observation."""
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class MemoryOp(str, Enum):
    """Memory operation types."""
    STORE = "STORE"
    UPDATE = "UPDATE"
    COMPRESS = "COMPRESS"
    ARCHIVE = "ARCHIVE"


class MemoryLevel(str, Enum):
    """Three-layer memory hierarchy."""
    L1 = "L1"  # Working / episodic (short-term)
    L2 = "L2"  # Semantic / compressed patterns
    L3 = "L3"  # Wisdom / deep abstractions


class NodeRole(str, Enum):
    """Roles for workflow nodes."""
    CLARIFY = "clarify"
    MANAGER = "manager"
    TOOL_DEVELOPER = "tool_developer"
    EXECUTOR = "executor"
    INTEGRATOR = "integrator"


# ────────────────────────────────────────────────────────────────────────────
# Cognitive Models
# ────────────────────────────────────────────────────────────────────────────

class CognitiveState(BaseModel):
    """Snapshot of the agent's cognitive state at a given moment."""
    system_active: SystemMode = SystemMode.SYSTEM_1
    prediction_error: float = Field(default=0.0, ge=0.0, le=1.0)
    surprise_level: SurpriseLevel = SurpriseLevel.NONE
    workspace_occupancy: float = Field(default=0.0, ge=0.0, le=1.0)

    class Config:
        use_enum_values = True


class MemoryOperation(BaseModel):
    """A single memory operation to be carried out."""
    operation: MemoryOp
    target_level: MemoryLevel
    key: str = ""
    content: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


class ToolCall(BaseModel):
    """A tool invocation specification."""
    name: str
    args: Dict[str, Any] = Field(default_factory=dict)


class AgentDecision(BaseModel):
    """The agent's decision output at each cognitive cycle."""
    cognitive_state: CognitiveState
    memory_operations: List[MemoryOperation] = Field(default_factory=list)
    tool_calls: List[ToolCall] = Field(default_factory=list)
    response: str = ""

    @field_validator("tool_calls")
    @classmethod
    def limit_tool_calls(cls, v: List[ToolCall]) -> List[ToolCall]:
        if len(v) > 3:
            raise ValueError(
                f"AgentDecision can have at most 3 tool_calls, got {len(v)}. "
                "Prioritize the most important actions."
            )
        return v

    class Config:
        use_enum_values = True


# ────────────────────────────────────────────────────────────────────────────
# Clarification Models
# ────────────────────────────────────────────────────────────────────────────

class ClarificationOutput(BaseModel):
    """Output from the ambiguity check / clarification step."""
    is_ambiguous: bool = False
    issues: List[str] = Field(default_factory=list)
    questions: List[str] = Field(default_factory=list)
    clarified_query: str = ""


# ────────────────────────────────────────────────────────────────────────────
# Node Output Models
# ────────────────────────────────────────────────────────────────────────────

class NodeOutput(BaseModel):
    """Base model for all node outputs."""
    role: NodeRole
    success: bool = True
    message: str = ""
    data: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


class ResearchNodeOutput(NodeOutput):
    """Output from a research-oriented execution step."""
    findings: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class CodeNodeOutput(NodeOutput):
    """Output from a code-generation / execution step."""
    code: str = ""
    language: str = "python"
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    artifacts: List[str] = Field(default_factory=list)


# ────────────────────────────────────────────────────────────────────────────
# JSON Schema for AgentDecision (used in structured generation)
# ────────────────────────────────────────────────────────────────────────────

AGENT_DECISION_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "cognitive_state": {
            "type": "object",
            "properties": {
                "system_active": {
                    "type": "string",
                    "enum": ["1", "2"],
                    "description": "System 1 (fast) or System 2 (deliberate)",
                },
                "prediction_error": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Difference between predicted and actual outcome",
                },
                "surprise_level": {
                    "type": "string",
                    "enum": ["NONE", "LOW", "MEDIUM", "HIGH"],
                },
                "workspace_occupancy": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
            },
            "required": ["system_active", "prediction_error", "surprise_level", "workspace_occupancy"],
        },
        "memory_operations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": ["STORE", "UPDATE", "COMPRESS", "ARCHIVE"]},
                    "target_level": {"type": "string", "enum": ["L1", "L2", "L3"]},
                    "key": {"type": "string"},
                    "content": {"type": "string"},
                    "metadata": {"type": "object"},
                },
                "required": ["operation", "target_level"],
            },
        },
        "tool_calls": {
            "type": "array",
            "minItems": 0,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "args": {"type": "object"},
                },
                "required": ["name"],
            },
        },
        "response": {"type": "string"},
    },
    "required": ["cognitive_state", "memory_operations", "tool_calls", "response"],
}


# ────────────────────────────────────────────────────────────────────────────
# LangGraph AgentState (TypedDict for graph state)
# ────────────────────────────────────────────────────────────────────────────

class AgentState(TypedDict, total=False):
    """Full state carried through the LangGraph workflow."""
    # ── Input ──
    query: str                                    # Original user query
    clarified_query: str                          # After clarification
    # ── Cognitive ──
    cognitive_state: Dict[str, Any]               # Serialized CognitiveState
    prediction_error: float
    surprise_level: str
    system_mode: str                              # "1" or "2"
    # ── Memory ──
    relevant_memories: List[Dict[str, Any]]
    memory_operations: List[Dict[str, Any]]
    # ── Tools ──
    selected_tools: List[str]
    available_tools: List[Dict[str, Any]]
    need_new_tools: bool
    new_tool_specs: List[Dict[str, Any]]
    # ── Execution ──
    tool_calls: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]
    execution_count: int
    max_executions: int
    # ── Chunks ──
    north_star: str
    ground_truth: str
    chunks: List[Dict[str, Any]]
    prediction: str
    # ── Clarification ──
    is_ambiguous: bool
    ambiguity_issues: List[str]
    clarification_questions: List[str]
    # ── Output ──
    final_response: str
    node_outputs: List[Dict[str, Any]]
    # ── Metadata ──
    task_id: str
    session_id: str
    timestamp: float
    error: Optional[str]
    should_terminate: bool


# ────────────────────────────────────────────────────────────────────────────
# Extended Schemas (for agent_core.py and supporting modules)
# ────────────────────────────────────────────────────────────────────────────

class IntentCategory(str, Enum):
    """User intent categories."""
    QUESTION = "question"
    COMMAND = "command"
    CREATION = "creation"
    ANALYSIS = "analysis"
    CONVERSATION = "conversation"
    AMBIGUOUS = "ambiguous"


class ToolStatus(str, Enum):
    """Tool execution status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class CognitiveChunk(BaseModel):
    """A single chunk in working memory (4 ± 1 constraint)."""
    id: str = Field(default_factory=lambda: f"chunk_{int(time.time()*1000)}")
    label: str
    content: str
    source: str = "agent"
    priority: float = 1.0
    created_at: str = Field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    class Config:
        use_enum_values = True


class PriorPrediction(BaseModel):
    """Prediction generated before action execution."""
    predicted_tool: Optional[str] = None
    predicted_outcome_type: str = "text"
    predicted_key_entities: List[str] = Field(default_factory=list)
    confidence: float = 0.5


class PredictionError(BaseModel):
    """Computed error between prior prediction and actual outcome."""
    tool_match: bool = False
    entity_overlap: float = 0.0
    outcome_type_match: bool = False
    composite_error: float = 0.0  # 0 = perfect prediction, 1 = completely wrong


class ToolDefinition(BaseModel):
    """Definition of a tool (base or evolved)."""
    name: str
    description: str
    parameters_schema: Dict[str, Any] = Field(default_factory=dict)
    is_evolved: bool = False
    usage_count: int = 0
    success_count: int = 0
    created_at: str = Field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))


class ExtendedToolCall(BaseModel):
    """Extended tool call with execution tracking."""
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    status: ToolStatus = ToolStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    prediction_error: Optional[PredictionError] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class MemoryItem(BaseModel):
    """A single memory entry at any level."""
    id: str = Field(default_factory=lambda: f"mem_{int(time.time()*1000)}")
    level: MemoryLevel = MemoryLevel.L1
    content: str
    summary: Optional[str] = None
    embedding_id: Optional[str] = None
    weight: float = 1.0
    tags: List[str] = Field(default_factory=list)
    source_interaction_id: Optional[str] = None
    index_keys: List[str] = Field(default_factory=list)
    access_count: int = 0
    created_at: str = Field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    last_accessed: Optional[str] = None
    is_frozen: bool = False


class MemoryStats(BaseModel):
    """Statistics about the memory system."""
    l1_count: int = 0
    l2_count: int = 0
    l3_count: int = 0
    l1_token_estimate: int = 0
    compression_count: int = 0
    last_compression: Optional[str] = None


class InteractionRecord(BaseModel):
    """Record of a single user-agent interaction."""
    id: str = Field(default_factory=lambda: f"int_{int(time.time()*1000)}")
    user_query: str
    agent_response: str = ""
    intent: str = "conversation"
    system_mode: str = "2"
    tools_used: List[str] = Field(default_factory=list)
    prediction_errors: List[float] = Field(default_factory=list)
    chunks_built: int = 0
    dag_path: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    duration_ms: float = 0.0


class EvolutionMetrics(BaseModel):
    """Metrics tracked for the Evolutionary Growth Log (EGL)."""
    egl_version: float = 1.0
    total_interactions: int = 0
    total_tools_forged: int = 0
    total_compressions: int = 0
    avg_prediction_error: float = 0.0
    system1_ratio: float = 0.0
    tool_success_rates: Dict[str, float] = Field(default_factory=dict)
    intent_distribution: Dict[str, int] = Field(default_factory=dict)
    last_evolved_at: Optional[str] = None


class ClarificationPayload(BaseModel):
    """Payload for ambiguity clarification."""
    original_query: str
    clarification_question: str
    suggested_answers: List[str] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))


class AgentFullState(BaseModel):
    """Full state snapshot of the agent (Pydantic model for API responses)."""
    mode: str = "2"
    current_chunks: List[Dict[str, Any]] = Field(default_factory=list)
    active_tools: List[str] = Field(default_factory=list)
    interaction_count: int = 0
    total_tools_forged: int = 0
    l1_entry_count: int = 0
    l2_entry_count: int = 0
    l3_entry_count: int = 0
    last_compression: Optional[str] = None
    egl_version: float = 1.0
    avg_prediction_error: float = 0.0
    system1_ratio: float = 0.0


class WorkflowState(BaseModel):
    """State carried through the LangGraph workflow (Pydantic model)."""
    user_query: str
    intent: str = "conversation"
    system_mode: str = "2"
    chunks: List[Dict[str, Any]] = Field(default_factory=list)
    prediction: Optional[Dict[str, Any]] = None
    prediction_error: Optional[Dict[str, Any]] = None
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    decision: Optional[Dict[str, Any]] = None
    raw_llm_output: Optional[str] = None
    validated: bool = False
    critic_passed: bool = False
    response_text: str = ""
    error: Optional[str] = None
    retry_count: int = 0
    timestamp: str = Field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
