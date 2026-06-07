"""
CerebroForge (铸脑) - Self-Evolving Cognitive Agent Framework
=============================================================

A brain-inspired cognitive agent framework implementing:
  - Dual-process cognition (System 1 / System 2)
  - 3-layer memory (L1/L2/L3) with SQLite + ChromaDB
  - Tool self-evolution and skill forging
  - LangGraph-based CAST-IRON workflow
  - Prediction error-driven surprise and learning
"""

__version__ = "0.1.0"
__author__ = "CerebroForge Team"

try:
    from backend.config import (
        BASE_TOOLS,
        CHROMA_DIR,
        DATA_DIR,
        DB_PATH,
        DEFAULT_MODEL,
        FALLBACK_MODEL,
        HIGH_FREQ_THRESHOLD,
        L1_MAX_ENTRIES,
        L1_MAX_TOKENS,
        MAX_COGNITIVE_CHUNKS,
        MAX_TASK_EXECUTION_CNT,
        MAX_TOOL_FORGE_PER_TASK,
        NVIDIA_API_KEY,
        NVIDIA_BASE_URL,
        PREDICTION_ERROR_SYS1,
        PREDICTION_ERROR_SYS2,
        PROJECT_ROOT,
        SUCCESS_RATE_THRESHOLD,
        TOOL_EVO_BUDGET_TOKENS,
        WORKSPACE_DIR,
    )
except ImportError:
    from config import (
        BASE_TOOLS,
        CHROMA_DIR,
        DATA_DIR,
        DB_PATH,
        DEFAULT_MODEL,
        FALLBACK_MODEL,
        HIGH_FREQ_THRESHOLD,
        L1_MAX_ENTRIES,
        L1_MAX_TOKENS,
        MAX_COGNITIVE_CHUNKS,
        MAX_TASK_EXECUTION_CNT,
        MAX_TOOL_FORGE_PER_TASK,
        NVIDIA_API_KEY,
        NVIDIA_BASE_URL,
        PREDICTION_ERROR_SYS1,
        PREDICTION_ERROR_SYS2,
        PROJECT_ROOT,
        SUCCESS_RATE_THRESHOLD,
        TOOL_EVO_BUDGET_TOKENS,
        WORKSPACE_DIR,
    )

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
        ResearchNodeOutput,
        CodeNodeOutput,
        SurpriseLevel,
        SystemMode,
        ToolCall,
        AGENT_DECISION_JSON_SCHEMA,
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
        ResearchNodeOutput,
        CodeNodeOutput,
        SurpriseLevel,
        SystemMode,
        ToolCall,
        AGENT_DECISION_JSON_SCHEMA,
    )

try:
    from backend.llm_client import LLMClient
except ImportError:
    from llm_client import LLMClient

try:
    from backend.cognitive import CognitiveChunk, CognitiveEngine
except ImportError:
    from cognitive import CognitiveChunk, CognitiveEngine

try:
    from backend.memory import MemorySystem
except ImportError:
    from memory import MemorySystem

try:
    from backend.computer_use import ComputerUseToolkit
except ImportError:
    from computer_use import ComputerUseToolkit

try:
    from backend.workflow import CerebroForgeWorkflow, create_workflow
except ImportError:
    from workflow import CerebroForgeWorkflow, create_workflow

__all__ = [
    "BASE_TOOLS", "CHROMA_DIR", "DATA_DIR", "DB_PATH",
    "DEFAULT_MODEL", "FALLBACK_MODEL", "HIGH_FREQ_THRESHOLD",
    "L1_MAX_ENTRIES", "L1_MAX_TOKENS", "MAX_COGNITIVE_CHUNKS",
    "MAX_TASK_EXECUTION_CNT", "MAX_TOOL_FORGE_PER_TASK",
    "NVIDIA_API_KEY", "NVIDIA_BASE_URL",
    "PREDICTION_ERROR_SYS1", "PREDICTION_ERROR_SYS2",
    "PROJECT_ROOT", "SUCCESS_RATE_THRESHOLD",
    "TOOL_EVO_BUDGET_TOKENS", "WORKSPACE_DIR",
    "AgentDecision", "AgentState", "ClarificationOutput",
    "CognitiveState", "MemoryLevel", "MemoryOp", "MemoryOperation",
    "NodeOutput", "NodeRole", "ResearchNodeOutput", "CodeNodeOutput",
    "SurpriseLevel", "SystemMode", "ToolCall", "AGENT_DECISION_JSON_SCHEMA",
    "LLMClient", "CognitiveChunk", "CognitiveEngine",
    "MemorySystem", "ComputerUseToolkit",
    "CerebroForgeWorkflow", "create_workflow",
]
