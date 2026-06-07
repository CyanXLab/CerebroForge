"""
CerebroForge (铸脑) - Cognitive Engine
======================================
Brain-inspired cognitive architecture implementing:
  - 4-chunk working memory (Miller's Law ±1)
  - Intent classification
  - Ambiguity detection (4 rules)
  - Prior prediction generation
  - Prediction error computation
  - System 1 / System 2 routing (Kahneman)
  - Clarification generation
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    from backend.config import (
        HIGH_FREQ_THRESHOLD,
        MAX_COGNITIVE_CHUNKS,
        PREDICTION_ERROR_SYS1,
        PREDICTION_ERROR_SYS2,
        SUCCESS_RATE_THRESHOLD,
    )
except ImportError:
    from config import (
        HIGH_FREQ_THRESHOLD,
        MAX_COGNITIVE_CHUNKS,
        PREDICTION_ERROR_SYS1,
        PREDICTION_ERROR_SYS2,
        SUCCESS_RATE_THRESHOLD,
    )

try:
    from backend.llm_client import LLMClient
except ImportError:
    from llm_client import LLMClient

try:
    from backend.schemas import (
        CognitiveState,
        ClarificationOutput,
        MemoryLevel,
        MemoryOperation,
        MemoryOp,
        SurpriseLevel,
        SystemMode,
    )
except ImportError:
    from schemas import (
        CognitiveState,
        ClarificationOutput,
        MemoryLevel,
        MemoryOperation,
        MemoryOp,
        SurpriseLevel,
        SystemMode,
    )

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Cognitive Chunk
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class CognitiveChunk:
    """A unit of information in the agent's working memory.

    Inspired by Miller's Law: humans hold ~7±2 chunks; we enforce 4±1
    to account for the additional cognitive overhead of meta-reasoning.
    """

    name: str
    content: str  # Truncated to 700 chars to fit token budget
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.content) > 700:
            self.content = self.content[:697] + "..."

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "content": self.content, "meta": self.meta}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CognitiveChunk":
        return cls(name=d["name"], content=d["content"], meta=d.get("meta", {}))


# ────────────────────────────────────────────────────────────────────────────
# Intent Classification (Rule-Based)
# ────────────────────────────────────────────────────────────────────────────

_INTENT_RULES: Dict[str, List[str]] = {
    "research": [
        r"\b(research|investigate|find\s+out|look\s+up|search|explore|analyze)\b",
        r"\b(what\s+is|who\s+is|when\s+did|where\s+is|how\s+does)\b",
        r"\b(tell\s+me\s+about|explain|describe|overview)\b",
    ],
    "code": [
        r"\b(write|create|implement|develop|build|code|program|script)\b",
        r"\b(function|class|module|api|endpoint|algorithm)\b",
        r"\b(debug|fix|refactor|optimize|test)\b",
    ],
    "analysis": [
        r"\b(analyze|compare|evaluate|assess|judge|rate)\b",
        r"\b(pros?\s+and\s+cons?|advantages|disadvantages|trade-?offs?)\b",
        r"\b(benchmark|metric|performance|statistics)\b",
    ],
    "creative": [
        r"\b(design|brainstorm|ideate|imagine|create|invent|compose)\b",
        r"\b(story|poem|essay|article|blog|content)\b",
        r"\b(suggest|recommend|propose)\b",
    ],
    "execution": [
        r"\b(run|execute|launch|start|deploy|install)\b",
        r"\b(command|terminal|shell|bash|script)\b",
        r"\b(docker|container|server|service)\b",
    ],
    "conversation": [
        r"\b(hello|hi|hey|greetings|good\s+(morning|afternoon|evening))\b",
        r"\b(thanks?|thank\s+you|please|help)\b",
        r"\b(how\s+are\s+you|what\s+can\s+you\s+do)\b",
    ],
}


class CognitiveEngine:
    """The brain-inspired cognitive engine at the core of CerebroForge.

    Responsibilities:
      1. Build and manage 4-chunk working memory
      2. Classify user intent
      3. Detect ambiguity via 4 rules
      4. Generate prior predictions
      5. Compute prediction error
      6. Route between System 1 and System 2
      7. Generate clarification when needed
    """

    def __init__(self, llm: Optional[LLMClient] = None) -> None:
        self.llm = llm or LLMClient()

    # ── Chunk Building ─────────────────────────────────────────────────────

    def build_4_chunks(
        self,
        north_star: str,
        ground_truth: str,
        relevant_memories: List[Dict[str, Any]],
        prediction: str,
    ) -> List[CognitiveChunk]:
        """
        Build exactly 4 (±1) cognitive chunks for working memory.

        Chunk 1 — North Star: the overarching goal
        Chunk 2 — Ground Truth: current factual context
        Chunk 3 — Relevant Memories: compressed L1/L2/L3 fragments
        Chunk 4 — Prediction: prior expectation of outcome

        Returns a list of 3–5 CognitiveChunk instances.
        """
        # Chunk 1: North Star
        chunks = [
            CognitiveChunk(
                name="north_star",
                content=north_star,
                meta={"priority": "critical", "mutable": False},
            )
        ]

        # Chunk 2: Ground Truth
        if ground_truth:
            chunks.append(
                CognitiveChunk(
                    name="ground_truth",
                    content=ground_truth,
                    meta={"priority": "high", "mutable": True},
                )
            )

        # Chunk 3: Relevant Memories (compressed)
        if relevant_memories:
            memory_text = self._compress_memories_for_chunk(relevant_memories)
            chunks.append(
                CognitiveChunk(
                    name="relevant_memories",
                    content=memory_text,
                    meta={
                        "priority": "medium",
                        "mutable": True,
                        "source_count": len(relevant_memories),
                    },
                )
            )

        # Chunk 4: Prediction
        if prediction:
            chunks.append(
                CognitiveChunk(
                    name="prediction",
                    content=prediction,
                    meta={"priority": "high", "mutable": True},
                )
            )

        # Enforce 4±1 limit
        if len(chunks) > MAX_COGNITIVE_CHUNKS + 1:
            # Keep north_star, ground_truth, merge rest
            excess = chunks[MAX_COGNITIVE_CHUNKS:]
            merged = " | ".join(c.content[:200] for c in excess)
            chunks = chunks[: MAX_COGNITIVE_CHUNKS]
            chunks.append(
                CognitiveChunk(
                    name="merged_context",
                    content=merged,
                    meta={"priority": "low", "merged_from": [c.name for c in excess]},
                )
            )

        return chunks[: MAX_COGNITIVE_CHUNKS + 1]

    @staticmethod
    def _compress_memories_for_chunk(memories: List[Dict[str, Any]]) -> str:
        """Compress a list of memory records into a single chunk-friendly string."""
        parts: List[str] = []
        for mem in memories[:10]:  # Cap at 10 for token budget
            level = mem.get("level", "L1")
            key = mem.get("key", "")
            content = mem.get("content", "")
            weight = mem.get("weight", 1.0)
            if key:
                parts.append(f"[{level}|{key}|w={weight:.1f}] {content[:150]}")
            else:
                parts.append(f"[{level}|w={weight:.1f}] {content[:150]}")
        return "\n".join(parts)

    # ── Intent Classification ──────────────────────────────────────────────

    def classify_intent(self, query: str) -> str:
        """
        Classify the user's intent using rule-based matching.

        Returns one of: research, code, analysis, creative, execution, conversation.
        """
        query_lower = query.lower()

        best_intent = "conversation"
        best_score = 0

        for intent, patterns in _INTENT_RULES.items():
            score = 0
            for pattern in patterns:
                matches = re.findall(pattern, query_lower)
                score += len(matches)
            if score > best_score:
                best_score = score
                best_intent = intent

        return best_intent

    # ── Ambiguity Detection (4 Rules) ─────────────────────────────────────

    def check_ambiguity(self, query: str) -> Tuple[bool, List[str]]:
        """
        Detect ambiguity in the user query using 4 rules.

        Rule 1: Pronouns without clear referent (it, this, that, they)
        Rule 2: More than 2 interpretation paths
        Rule 3: Missing key parameters (which, what specifically, etc.)
        Rule 4: Prediction error oscillating (requires external state)

        Returns (is_ambiguous, list_of_issues).
        """
        issues: List[str] = []

        # Rule 1: Pronouns without context
        pronoun_pattern = r"\b(it|this|that|they|them|these|those|he|she)\b"
        pronoun_matches = re.findall(pronoun_pattern, query, re.IGNORECASE)
        if pronoun_matches and len(query.split()) < 15:
            issues.append(
                f"Pronoun(s) without clear referent: {', '.join(set(pronoun_matches))}. "
                "Short queries with pronouns often lack sufficient context."
            )

        # Rule 2: Multiple interpretation paths
        intent_scores: Dict[str, int] = {}
        query_lower = query.lower()
        for intent, patterns in _INTENT_RULES.items():
            score = sum(len(re.findall(p, query_lower)) for p in patterns)
            if score > 0:
                intent_scores[intent] = score

        if len(intent_scores) > 2:
            top_intents = sorted(intent_scores, key=intent_scores.get, reverse=True)[:3]
            issues.append(
                f"Multiple interpretation paths detected: {', '.join(top_intents)}. "
                "The query could be interpreted in several different ways."
            )

        # Rule 3: Missing key parameters
        vague_indicators = [
            r"\bsome\b",
            r"\bsomething\b",
            r"\banything\b",
            r"\beverything\b",
            r"\ba\s+thing\b",
            r"\bthat\s+thing\b",
            r"\bit\b\s*(?:for\s+me)?$",
        ]
        vague_matches = []
        for pattern in vague_indicators:
            if re.search(pattern, query_lower):
                vague_matches.append(pattern.replace(r"\b", "").replace(r"\s+", " "))

        if vague_matches:
            issues.append(
                f"Vague or missing key parameters: {', '.join(vague_matches)}. "
                "The request lacks specific details needed for precise execution."
            )

        # Rule 4: Check if query is too short and open-ended
        words = query.split()
        if len(words) <= 3 and not query.endswith("?"):
            issues.append(
                "Query is very short and open-ended. Additional context or "
                "specification would help produce a more accurate response."
            )

        is_ambiguous = len(issues) >= 1
        return is_ambiguous, issues

    # ── Prior Prediction ───────────────────────────────────────────────────

    def generate_prior_prediction(
        self,
        chunks: List[CognitiveChunk],
        next_action: str,
    ) -> str:
        """
        Generate a prior prediction of what the outcome of the next action
        should be, given the current cognitive chunks.

        This implements the predictive coding principle: the brain constantly
        generates predictions and computes error signals.
        """
        chunks_text = "\n".join(
            f"[{c.name}] {c.content}" for c in chunks
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a predictive reasoning engine. Given the current "
                    "cognitive context and a planned next action, predict the "
                    "most likely outcome. Be specific and concise (2-3 sentences)."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Current cognitive context:\n{chunks_text}\n\n"
                    f"Planned next action: {next_action}\n\n"
                    "Predict the most likely outcome:"
                ),
            },
        ]

        try:
            return self.llm.generate(
                messages=messages,
                temperature=0.3,
                max_tokens=256,
            )
        except RuntimeError as exc:
            logger.warning(f"Prediction generation failed: {exc}")
            return "Unable to generate prediction due to LLM error."

    # ── Prediction Error Computation ───────────────────────────────────────

    def compute_prediction_error(
        self,
        predicted: str,
        actual: str,
    ) -> Tuple[float, SurpriseLevel]:
        """
        Compute prediction error as (1 - similarity) between predicted and
        actual outcomes. Also determine surprise level.

        Returns (error_value, surprise_level).
        """
        if not predicted or not actual:
            return 0.5, SurpriseLevel.MEDIUM

        # Quick text overlap check before expensive LLM call
        overlap = self._text_overlap_ratio(predicted, actual)

        if overlap > 0.9:
            return 1.0 - overlap, SurpriseLevel.NONE
        elif overlap > 0.7:
            # Moderate overlap — use LLM for finer judgment
            similarity = self.llm.judge_similarity(predicted, actual)
        elif overlap > 0.4:
            similarity = self.llm.judge_similarity(predicted, actual)
        else:
            # Very low overlap — likely high error
            similarity = overlap

        error = 1.0 - similarity
        error = max(0.0, min(1.0, error))

        # Map error to surprise level
        if error < 0.2:
            surprise = SurpriseLevel.NONE
        elif error < 0.4:
            surprise = SurpriseLevel.LOW
        elif error < 0.65:
            surprise = SurpriseLevel.MEDIUM
        else:
            surprise = SurpriseLevel.HIGH

        return error, surprise

    @staticmethod
    def _text_overlap_ratio(text_a: str, text_b: str) -> float:
        """Fast approximate text overlap using word sets."""
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union) if union else 0.0

    # ── System 1 / System 2 Routing ───────────────────────────────────────

    def decide_system(
        self,
        prediction_error: float,
        high_freq_match: bool,
        ambiguous: bool,
    ) -> SystemMode:
        """
        Decide whether to use System 1 (fast, pattern-matched) or
        System 2 (slow, deliberate) based on:

        1. Prediction error magnitude
        2. Whether the task matches a high-frequency tool pattern
        3. Whether the query is ambiguous

        Routing logic:
        - Low error + high-freq match + not ambiguous → System 1
        - High error OR ambiguous → System 2
        - Medium → System 2 (err on the side of caution)
        """
        # System 1 conditions: low error, known pattern, not ambiguous
        if (
            prediction_error < PREDICTION_ERROR_SYS1
            and high_freq_match
            and not ambiguous
        ):
            return SystemMode.SYSTEM_1

        # System 2 conditions: high error OR ambiguous
        if prediction_error > PREDICTION_ERROR_SYS2 or ambiguous:
            return SystemMode.SYSTEM_2

        # Default to System 2 for medium-error scenarios
        return SystemMode.SYSTEM_2

    # ── Clarification Generation ───────────────────────────────────────────

    def generate_clarification(
        self,
        query: str,
        ambiguity_issues: List[str],
    ) -> ClarificationOutput:
        """
        Generate clarification questions for an ambiguous query.

        Delegates to the LLM client's specialized method.
        """
        try:
            return self.llm.generate_clarifying_questions(query, ambiguity_issues)
        except RuntimeError as exc:
            logger.warning(f"Clarification generation failed: {exc}")
            # Fallback: return simple clarification
            return ClarificationOutput(
                is_ambiguous=True,
                issues=ambiguity_issues,
                questions=["Could you please provide more details about what you're looking for?"],
                clarified_query=query,
            )

    # ── Full Cognitive Cycle ───────────────────────────────────────────────

    def run_cognitive_cycle(
        self,
        query: str,
        north_star: str = "",
        ground_truth: str = "",
        relevant_memories: Optional[List[Dict[str, Any]]] = None,
        high_freq_tools: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run a complete cognitive cycle and return the cognitive state
        plus all intermediate results.

        This is the main entry point for the cognitive engine.
        """
        relevant_memories = relevant_memories or []
        high_freq_tools = high_freq_tools or []

        # Step 1: Check ambiguity
        is_ambiguous, ambiguity_issues = self.check_ambiguity(query)

        # Step 2: Build cognitive chunks
        prediction = ""  # Will be generated below
        chunks = self.build_4_chunks(
            north_star=north_star or "Assist the user effectively",
            ground_truth=ground_truth,
            relevant_memories=relevant_memories,
            prediction=prediction,
        )

        # Step 3: Generate prior prediction
        next_action = f"Process query: {query}"
        prediction = self.generate_prior_prediction(chunks, next_action)

        # Rebuild chunks with prediction
        chunks = self.build_4_chunks(
            north_star=north_star or "Assist the user effectively",
            ground_truth=ground_truth,
            relevant_memories=relevant_memories,
            prediction=prediction,
        )

        # Step 4: Determine high-frequency match
        intent = self.classify_intent(query)
        high_freq_match = any(
            tool in high_freq_tools for tool in self._intent_to_tools(intent)
        )

        # Step 5: Initial system routing (error=0 before execution)
        initial_error = 0.0
        system_mode = self.decide_system(
            prediction_error=initial_error,
            high_freq_match=high_freq_match,
            ambiguous=is_ambiguous,
        )

        # Step 6: Build cognitive state
        workspace_occupancy = len(chunks) / (MAX_COGNITIVE_CHUNKS + 1)
        cognitive_state = CognitiveState(
            system_active=system_mode,
            prediction_error=initial_error,
            surprise_level=SurpriseLevel.NONE,
            workspace_occupancy=workspace_occupancy,
        )

        # Step 7: Generate clarification if ambiguous
        clarification = None
        clarified_query = query
        if is_ambiguous:
            clarification = self.generate_clarification(query, ambiguity_issues)
            clarified_query = clarification.clarified_query or query

        return {
            "query": query,
            "clarified_query": clarified_query,
            "intent": intent,
            "is_ambiguous": is_ambiguous,
            "ambiguity_issues": ambiguity_issues,
            "clarification": clarification.model_dump() if clarification else None,
            "chunks": [c.to_dict() for c in chunks],
            "prediction": prediction,
            "cognitive_state": cognitive_state.model_dump(),
            "system_mode": system_mode.value,
            "high_freq_match": high_freq_match,
        }

    @staticmethod
    def _intent_to_tools(intent: str) -> List[str]:
        """Map an intent classification to likely needed tools."""
        mapping = {
            "research": ["web_search", "web_fetch", "text_extract"],
            "code": ["python_exec", "file_read", "file_write", "run_terminal"],
            "analysis": ["python_exec", "calculate", "grep_files"],
            "creative": ["web_search", "text_extract", "file_write"],
            "execution": ["run_terminal", "python_exec", "computer_use"],
            "conversation": [],
        }
        return mapping.get(intent, [])
