"""
CerebroForge (铸脑) - Configuration Module
==========================================
Central configuration for the self-evolving cognitive agent framework.
All paths are computed dynamically based on the project root.
"""

import os
from pathlib import Path
from typing import List

# ────────────────────────────────────────────────────────────────────────────
# Project Root & Directory Layout
# ────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent  # CerebroForge/
WORKSPACE_DIR: Path = PROJECT_ROOT / "workspace"
DATA_DIR: Path = PROJECT_ROOT / "data"
DB_PATH: Path = DATA_DIR / "cerebroforge.db"
CHROMA_DIR: Path = DATA_DIR / "chroma"

# Ensure directories exist
for _dir in (WORKSPACE_DIR, DATA_DIR, CHROMA_DIR):
    os.makedirs(_dir, exist_ok=True)

# ────────────────────────────────────────────────────────────────────────────
# NVIDIA / LLM API Configuration
# ────────────────────────────────────────────────────────────────────────────
NVIDIA_BASE_URL: str = "https://integrate.api.nvidia.com/v1"
NVIDIA_API_KEY: str = os.environ.get(
    "NVIDIA_API_KEY",
    "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",
)
DEFAULT_MODEL: str = "minimaxai/minimax-m2.7"
FALLBACK_MODEL: str = "meta/llama-3.1-70b-instruct"

# ────────────────────────────────────────────────────────────────────────────
# Memory System Thresholds
# ────────────────────────────────────────────────────────────────────────────
L1_MAX_ENTRIES: int = 100
L1_MAX_TOKENS: int = 50000

# ────────────────────────────────────────────────────────────────────────────
# Cognitive Engine Thresholds
# ────────────────────────────────────────────────────────────────────────────
HIGH_FREQ_THRESHOLD: int = 10         # tool usage count to qualify as high-frequency
SUCCESS_RATE_THRESHOLD: float = 0.80  # minimum success rate for System-1 eligibility

PREDICTION_ERROR_SYS1: float = 0.3    # below this → System 1 (fast)
PREDICTION_ERROR_SYS2: float = 0.7    # above this → System 2 (slow, deliberate)

MAX_COGNITIVE_CHUNKS: int = 4         # 4 +/- 1 chunks in working memory
MAX_TOOL_FORGE_PER_TASK: int = 2      # max new tools forged per task
TOOL_EVO_BUDGET_TOKENS: int = 5000    # token budget for tool evolution LLM calls

# ────────────────────────────────────────────────────────────────────────────
# Execution Limits
# ────────────────────────────────────────────────────────────────────────────
MAX_TASK_EXECUTION_CNT: int = 5       # maximum re-execution loops before forced termination

# ────────────────────────────────────────────────────────────────────────────
# Forgetting & Weight Rules
# ────────────────────────────────────────────────────────────────────────────
L3_FREEZE_DAYS: int = 30              # days without access before L3 item is frozen
L3_FROZEN_WEIGHT: float = 0.1         # weight assigned to frozen L3 items
RETRIEVAL_MIN_WEIGHT: float = 0.6     # only load fragments with weight above this

# ────────────────────────────────────────────────────────────────────────────
# Base Tool Set
# ────────────────────────────────────────────────────────────────────────────
BASE_TOOLS: List[str] = [
    "web_search",
    "python_exec",
    "file_read",
    "file_write",
    "text_extract",
    "image_query",
    "web_fetch",
    "calculate",
    "run_terminal",
    "list_files",
    "grep_files",
    "computer_use",
]

# ────────────────────────────────────────────────────────────────────────────
# LLM Generation Defaults
# ────────────────────────────────────────────────────────────────────────────
DEFAULT_TEMPERATURE: float = 0.7
DEFAULT_TOP_P: float = 0.9
DEFAULT_MAX_TOKENS: int = 2048
MAX_RETRIES: int = 3

# ────────────────────────────────────────────────────────────────────────────
# Think Mode Markers (for models like Qwen that support /think /no_think)
# ────────────────────────────────────────────────────────────────────────────
THINK_MODE_SYSTEM_PROMPT: str = (
    "You are in deep-thinking mode. Use <think>...</think> tags to show your "
    "step-by-step reasoning process before giving your final answer. Take your "
    "time and be thorough."
)
NO_THINK_MODE_SYSTEM_PROMPT: str = (
    "You are in direct-answer mode. Respond immediately without showing "
    "intermediate reasoning. Be concise and direct."
)
