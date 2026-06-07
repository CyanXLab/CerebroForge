"""
CerebroForge (铸脑) - LLM Client
=================================
Unified LLM client using OpenAI SDK against NVIDIA API.
Supports generate, streaming, structured JSON, think/no-think modes,
fallback model on errors, and token usage tracking.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Type, TypeVar

from openai import OpenAI, APIError, APITimeoutError, RateLimitError

try:
    from backend.config import (
        DEFAULT_MAX_TOKENS,
        DEFAULT_MODEL,
        DEFAULT_TEMPERATURE,
        DEFAULT_TOP_P,
        FALLBACK_MODEL,
        MAX_RETRIES,
        NVIDIA_API_KEY,
        NVIDIA_BASE_URL,
        NO_THINK_MODE_SYSTEM_PROMPT,
        THINK_MODE_SYSTEM_PROMPT,
    )
except ImportError:
    from config import (
        DEFAULT_MAX_TOKENS,
        DEFAULT_MODEL,
        DEFAULT_TEMPERATURE,
        DEFAULT_TOP_P,
        FALLBACK_MODEL,
        MAX_RETRIES,
        NVIDIA_API_KEY,
        NVIDIA_BASE_URL,
        NO_THINK_MODE_SYSTEM_PROMPT,
        THINK_MODE_SYSTEM_PROMPT,
    )

try:
    from backend.schemas import ClarificationOutput
except ImportError:
    from schemas import ClarificationOutput

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ────────────────────────────────────────────────────────────────────────────
# Token Usage Tracker
# ────────────────────────────────────────────────────────────────────────────

class TokenUsageTracker:
    """Tracks cumulative token usage across LLM calls."""

    def __init__(self) -> None:
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0
        self.call_count: int = 0

    def record(self, prompt: int, completion: int) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += prompt + completion
        self.call_count += 1

    def summary(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "call_count": self.call_count,
        }

    def reset(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.call_count = 0


# ────────────────────────────────────────────────────────────────────────────
# LLM Client
# ────────────────────────────────────────────────────────────────────────────

class LLMClient:
    """Unified LLM client with retry, fallback, and structured output."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
        fallback_model: Optional[str] = None,
        default_temperature: Optional[float] = None,
        default_top_p: Optional[float] = None,
        default_max_tokens: Optional[int] = None,
    ) -> None:
        self.api_key = api_key or NVIDIA_API_KEY
        self.base_url = base_url or NVIDIA_BASE_URL
        self.default_model = default_model or DEFAULT_MODEL
        self.fallback_model = fallback_model or FALLBACK_MODEL
        self.default_temperature = default_temperature or DEFAULT_TEMPERATURE
        self.default_top_p = default_top_p or DEFAULT_TOP_P
        self.default_max_tokens = default_max_tokens or DEFAULT_MAX_TOKENS
        self.usage = TokenUsageTracker()

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    # ── Core Generation ────────────────────────────────────────────────────

    def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        think_mode: Optional[bool] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate a completion from the LLM.

        Args:
            messages: OpenAI-style chat messages.
            model: Override the default model.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            max_tokens: Maximum tokens to generate.
            think_mode: True → inject /think system prompt; False → /no_think; None → no injection.
            response_format: Optional response format (e.g. {"type": "json_object"}).

        Returns:
            The generated text content.
        """
        effective_model = model or self.default_model
        effective_temperature = temperature if temperature is not None else self.default_temperature
        effective_top_p = top_p if top_p is not None else self.default_top_p
        effective_max_tokens = max_tokens or self.default_max_tokens

        # Inject think-mode system prompt if requested
        processed_messages = self._inject_think_mode(messages, think_mode)

        last_error: Optional[Exception] = None

        for attempt in range(MAX_RETRIES):
            try:
                kwargs: Dict[str, Any] = {
                    "model": effective_model,
                    "messages": processed_messages,
                    "temperature": effective_temperature,
                    "top_p": effective_top_p,
                    "max_tokens": effective_max_tokens,
                }
                if response_format is not None:
                    kwargs["response_format"] = response_format

                response = self._client.chat.completions.create(**kwargs)

                # Track token usage
                if response.usage:
                    self.usage.record(
                        prompt=response.usage.prompt_tokens or 0,
                        completion=response.usage.completion_tokens or 0,
                    )

                content = response.choices[0].message.content or ""
                # Strip thinking tags from output if present
                content = self._strip_think_tags(content)
                return content

            except (APIError, APITimeoutError, RateLimitError) as exc:
                last_error = exc
                logger.warning(
                    f"LLM call attempt {attempt + 1}/{MAX_RETRIES} failed "
                    f"with model {effective_model}: {exc}"
                )
                # Fallback to secondary model on last retry
                if attempt == MAX_RETRIES - 2:
                    logger.info(f"Falling back to model: {self.fallback_model}")
                    effective_model = self.fallback_model
                time.sleep(0.5 * (attempt + 1))

        raise RuntimeError(
            f"LLM generation failed after {MAX_RETRIES} retries. "
            f"Last error: {last_error}"
        )

    def generate_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        think_mode: Optional[bool] = None,
    ) -> Any:
        """
        Stream a completion from the LLM.

        Returns a generator yielding delta content strings.
        """
        effective_model = model or self.default_model
        effective_temperature = temperature if temperature is not None else self.default_temperature
        effective_top_p = top_p if top_p is not None else self.default_top_p
        effective_max_tokens = max_tokens or self.default_max_tokens

        processed_messages = self._inject_think_mode(messages, think_mode)

        last_error: Optional[Exception] = None

        for attempt in range(MAX_RETRIES):
            try:
                stream = self._client.chat.completions.create(
                    model=effective_model,
                    messages=processed_messages,
                    temperature=effective_temperature,
                    top_p=effective_top_p,
                    max_tokens=effective_max_tokens,
                    stream=True,
                )

                def _stream_generator():
                    for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content

                return _stream_generator()

            except (APIError, APITimeoutError, RateLimitError) as exc:
                last_error = exc
                logger.warning(
                    f"LLM stream attempt {attempt + 1}/{MAX_RETRIES} failed: {exc}"
                )
                if attempt == MAX_RETRIES - 2:
                    effective_model = self.fallback_model
                time.sleep(0.5 * (attempt + 1))

        raise RuntimeError(
            f"LLM stream failed after {MAX_RETRIES} retries. Last error: {last_error}"
        )

    def structured_json(
        self,
        messages: List[Dict[str, str]],
        response_model: Type[T],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        think_mode: Optional[bool] = None,
    ) -> T:
        """
        Generate a structured JSON response and parse it into a Pydantic model.

        Args:
            messages: OpenAI-style chat messages.
            response_model: Pydantic model class to parse the response into.
            model: Override the default model.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens.
            think_mode: Think mode injection.

        Returns:
            An instance of response_model.
        """
        # Add JSON instruction to the last user message
        json_instruction = (
            f"\n\nIMPORTANT: You MUST respond with valid JSON that matches "
            f"the following schema. Do NOT include any text outside the JSON:\n"
            f"{json.dumps(response_model.model_json_schema(), indent=2)}"
        )

        augmented_messages = list(messages)
        if augmented_messages and augmented_messages[-1]["role"] == "user":
            augmented_messages[-1] = {
                "role": "user",
                "content": augmented_messages[-1]["content"] + json_instruction,
            }
        else:
            augmented_messages.append({"role": "user", "content": json_instruction})

        raw = self.generate(
            messages=augmented_messages,
            model=model,
            temperature=temperature or 0.3,
            max_tokens=max_tokens or 4096,
            think_mode=think_mode,
            response_format={"type": "json_object"},
        )

        # Parse JSON
        try:
            data = json.loads(raw)
            return response_model(**data)
        except (json.JSONDecodeError, Exception) as exc:
            logger.error(f"Failed to parse structured JSON: {exc}\nRaw: {raw[:500]}")
            # Attempt recovery: try to extract JSON from markdown code block
            recovered = self._extract_json_from_markdown(raw)
            if recovered:
                data = json.loads(recovered)
                return response_model(**data)
            raise

    # ── Specialized Methods ────────────────────────────────────────────────

    def judge_similarity(
        self,
        text_a: str,
        text_b: str,
    ) -> float:
        """
        Compute a similarity score between two texts using the LLM.

        Returns a float between 0.0 (completely different) and 1.0 (identical).
        Used for prediction error computation.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a semantic similarity judge. Compare the two texts "
                    "and output a single float between 0.0 and 1.0 representing "
                    "their semantic similarity. Output ONLY the number, nothing else."
                ),
            },
            {
                "role": "user",
                "content": f"Text A:\n{text_a}\n\nText B:\n{text_b}\n\nSimilarity score (0.0-1.0):",
            },
        ]

        try:
            raw = self.generate(
                messages=messages,
                temperature=0.0,
                max_tokens=10,
                think_mode=False,
            )
            score = float(raw.strip())
            return max(0.0, min(1.0, score))
        except (ValueError, RuntimeError) as exc:
            logger.warning(f"Similarity judge failed: {exc}, defaulting to 0.5")
            return 0.5

    def generate_clarifying_questions(
        self,
        query: str,
        ambiguity_issues: List[str],
    ) -> ClarificationOutput:
        """
        Generate clarifying questions for an ambiguous query.
        """
        issues_text = "\n".join(f"- {issue}" for issue in ambiguity_issues)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert at clarifying ambiguous requests. "
                    "Given a user query and identified ambiguity issues, "
                    "generate clarifying questions and a clarified version "
                    "of the query. Respond in JSON matching the ClarificationOutput schema."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"User Query: {query}\n\n"
                    f"Ambiguity Issues:\n{issues_text}\n\n"
                    "Generate a ClarificationOutput JSON with:\n"
                    "- is_ambiguous: true\n"
                    "- issues: the list of issues above\n"
                    "- questions: clarifying questions to resolve ambiguity\n"
                    "- clarified_query: your best interpretation of what the user means"
                ),
            },
        ]

        return self.structured_json(
            messages=messages,
            response_model=ClarificationOutput,
            temperature=0.3,
            max_tokens=1024,
        )

    # ── Helpers ────────────────────────────────────────────────────────────

    def _inject_think_mode(
        self,
        messages: List[Dict[str, str]],
        think_mode: Optional[bool],
    ) -> List[Dict[str, str]]:
        """Inject think/no-think system prompt if specified."""
        if think_mode is None:
            return messages

        injection = THINK_MODE_SYSTEM_PROMPT if think_mode else NO_THINK_MODE_SYSTEM_PROMPT
        result = list(messages)

        # Find and modify existing system message, or prepend one
        for i, msg in enumerate(result):
            if msg["role"] == "system":
                result[i] = {"role": "system", "content": injection + "\n\n" + msg["content"]}
                return result

        # No system message found — prepend
        result.insert(0, {"role": "system", "content": injection})
        return result

    @staticmethod
    def _strip_think_tags(text: str) -> str:
        """Remove <think ...>...</think/> tags from model output."""
        import re
        # Remove <think ...>...</think/> blocks
        text = re.sub(r"<think[^>]*>.*?</think\s*>", "", text, flags=re.DOTALL)
        return text.strip()

    @staticmethod
    def _extract_json_from_markdown(text: str) -> Optional[str]:
        """Try to extract JSON from a markdown code block."""
        import re
        match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def get_usage_summary(self) -> Dict[str, int]:
        """Return current token usage summary."""
        return self.usage.summary()

    def reset_usage(self) -> None:
        """Reset token usage tracker."""
        self.usage.reset()

    def reconfigure(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        """Reconfigure the client with new settings."""
        if base_url:
            self.base_url = base_url
        if api_key:
            self.api_key = api_key
        if model:
            self.default_model = model
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    @property
    def stats(self) -> Dict[str, Any]:
        """Return client statistics."""
        return {
            "call_count": self.usage.call_count,
            "total_tokens": self.usage.total_tokens,
            "model": self.default_model,
            "base_url": self.base_url,
        }

    # ── Convenience aliases (for backward compat) ───────────────────────

    def chat(self, messages, **kwargs) -> str:
        """Alias for generate()."""
        return self.generate(messages=messages, **kwargs)

    def chat_json(self, messages, schema_hint=None, **kwargs) -> Dict[str, Any]:
        """
        Generate JSON output (convenience method).

        Unlike structured_json(), this returns a raw dict, not a Pydantic model.
        """
        if schema_hint:
            augmented = list(messages)
            if augmented and augmented[-1]["role"] == "system":
                augmented[-1]["content"] += f"\n\nYou MUST respond with valid JSON matching this structure:\n{schema_hint}"
            else:
                augmented.append({"role": "system", "content": f"You MUST respond with valid JSON matching this structure:\n{schema_hint}"})
            messages = augmented

        raw = self.generate(messages=messages, response_format={"type": "json_object"}, **kwargs)

        text = raw.strip()
        if text.startswith("```"):
            import re
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                text = match.group()

        return json.loads(text)


# ────────────────────────────────────────────────────────────────────────────
# Singleton Instance
# ────────────────────────────────────────────────────────────────────────────

llm = LLMClient()
