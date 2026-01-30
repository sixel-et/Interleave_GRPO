"""
backends.py

Pluggable completion backends with a common interface for evaluating
frontier models on the interleave task.

Supports:
- AnthropicBackend: Claude models via Anthropic API

Future backends (interface documented, stub implementations):
- OpenAIBackend: GPT-4, GPT-4o
- GoogleBackend: Gemini
- XAIBackend: Grok
- DeepSeekBackend: DeepSeek
- HFInferenceBackend: HuggingFace serverless
- LocalHFBackend: Local HuggingFace models

Usage:
    from backends import AnthropicBackend

    backend = AnthropicBackend(model="claude-sonnet-4-20250514")
    response = backend.generate(
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100
    )
    print(f"Response: {response}")
    print(f"Cost so far: ${backend.get_cost():.4f}")
"""

from abc import ABC, abstractmethod
import asyncio
import os
import time
from typing import Optional

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional


# ============================================================================
# PRICING TABLE
# ============================================================================

# Pricing in USD per 1M tokens (easily updatable)
PRICING = {
    # Anthropic models
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
    "claude-opus-4-20250514": {"input": 15.0, "output": 75.0},
    "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.0},
    # OpenAI models (for future use)
    "gpt-4o": {"input": 2.50, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    # Google models (for future use)
    "gemini-1.5-pro": {"input": 1.25, "output": 5.0},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    # xAI models (for future use)
    "grok-2": {"input": 2.0, "output": 10.0},
    # DeepSeek models (for future use)
    "deepseek-chat": {"input": 0.14, "output": 0.28},
    "deepseek-reasoner": {"input": 0.55, "output": 2.19},
}


# ============================================================================
# ABSTRACT BASE CLASS
# ============================================================================

class CompletionBackend(ABC):
    """
    Abstract base class for completion backends.

    All backends must implement:
    - generate(): Generate completion from chat messages
    - name: Model identifier for logging/export

    Optional tracking (recommended):
    - total_input_tokens: Running count of input tokens
    - total_output_tokens: Running count of output tokens
    - get_cost(): Calculate total cost so far
    """

    @abstractmethod
    def generate(self, messages: list[dict], max_tokens: int) -> tuple[str, dict]:
        """
        Generate completion from chat messages.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
                     Example: [{"role": "user", "content": "Hello"}]
            max_tokens: Maximum tokens to generate

        Returns:
            Tuple of (text, metadata) where metadata includes:
            - api_time: Time spent on API call (seconds)
            - stop_reason: Why generation stopped ("end_turn" or "max_tokens")
            - input_tokens: Number of input tokens used
            - output_tokens: Number of output tokens generated
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Model identifier for logging/export."""
        pass

    def get_cost(self) -> float:
        """Return total cost so far in USD. Override in subclasses."""
        return 0.0


# ============================================================================
# ANTHROPIC BACKEND
# ============================================================================

class AnthropicBackend(CompletionBackend):
    """
    Anthropic API backend for Claude models.

    Requires ANTHROPIC_API_KEY environment variable or .env file.

    Usage:
        backend = AnthropicBackend(model="claude-sonnet-4-20250514")
        response = backend.generate(
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100
        )
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize Anthropic backend.

        Args:
            model: Model ID (e.g., "claude-sonnet-4-20250514", "claude-opus-4-20250514")
            max_retries: Maximum retry attempts for API errors
            retry_delay: Base delay between retries (exponential backoff)
        """
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "anthropic package required. Install with: pip install anthropic"
            )

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable not set. "
                "Set it directly or create a .env file."
            )

        self.client = Anthropic()
        self.async_client = None  # Lazy init for async
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Token tracking (with lock for async safety)
        self._lock = asyncio.Lock()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0

    def generate(self, messages: list[dict], max_tokens: int) -> tuple[str, dict]:
        """Generate completion using Anthropic API with timing and metadata.

        Uses streaming for large requests (>4096 tokens) to avoid timeout issues.
        """
        from anthropic import APIError, RateLimitError

        last_error = None
        use_streaming = max_tokens > 4096  # Stream for long outputs

        for attempt in range(self.max_retries):
            try:
                start = time.perf_counter()

                if use_streaming:
                    # Streaming for long requests
                    text = ""
                    input_tokens = 0
                    output_tokens = 0
                    stop_reason = ""

                    with self.client.messages.stream(
                        model=self.model,
                        max_tokens=max_tokens,
                        messages=messages,
                    ) as stream:
                        for text_chunk in stream.text_stream:
                            text += text_chunk

                        # Get final message for usage stats
                        final_message = stream.get_final_message()
                        input_tokens = final_message.usage.input_tokens
                        output_tokens = final_message.usage.output_tokens
                        stop_reason = final_message.stop_reason
                else:
                    # Non-streaming for short requests
                    response = self.client.messages.create(
                        model=self.model,
                        max_tokens=max_tokens,
                        messages=messages,
                    )
                    input_tokens = response.usage.input_tokens
                    output_tokens = response.usage.output_tokens
                    stop_reason = response.stop_reason
                    text = ""
                    if response.content and len(response.content) > 0:
                        text = response.content[0].text

                elapsed = time.perf_counter() - start

                # Track usage
                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens
                self.total_requests += 1

                # Build metadata
                metadata = {
                    "api_time": elapsed,
                    "stop_reason": stop_reason,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                }

                return text, metadata

            except RateLimitError as e:
                last_error = e
                wait_time = self.retry_delay * (2 ** attempt)
                print(f"Rate limit hit, waiting {wait_time:.1f}s before retry...")
                time.sleep(wait_time)

            except APIError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    print(f"API error: {e}, retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    raise

        raise last_error

    async def async_generate(self, messages: list[dict], max_tokens: int) -> tuple[str, dict]:
        """Async version of generate for parallel evaluation.

        Uses streaming for large requests (>4096 tokens) to avoid timeout issues.
        """
        from anthropic import AsyncAnthropic, APIError, RateLimitError

        # Lazy init async client
        if self.async_client is None:
            self.async_client = AsyncAnthropic()

        last_error = None
        use_streaming = max_tokens > 4096  # Stream for long outputs

        for attempt in range(self.max_retries):
            try:
                start = time.perf_counter()

                if use_streaming:
                    # Streaming for long requests
                    text = ""
                    input_tokens = 0
                    output_tokens = 0
                    stop_reason = ""

                    async with self.async_client.messages.stream(
                        model=self.model,
                        max_tokens=max_tokens,
                        messages=messages,
                    ) as stream:
                        async for text_chunk in stream.text_stream:
                            text += text_chunk

                        # Get final message for usage stats
                        final_message = await stream.get_final_message()
                        input_tokens = final_message.usage.input_tokens
                        output_tokens = final_message.usage.output_tokens
                        stop_reason = final_message.stop_reason
                else:
                    # Non-streaming for short requests
                    response = await self.async_client.messages.create(
                        model=self.model,
                        max_tokens=max_tokens,
                        messages=messages,
                    )
                    input_tokens = response.usage.input_tokens
                    output_tokens = response.usage.output_tokens
                    stop_reason = response.stop_reason
                    text = ""
                    if response.content and len(response.content) > 0:
                        text = response.content[0].text

                elapsed = time.perf_counter() - start

                # Track usage (thread-safe)
                async with self._lock:
                    self.total_input_tokens += input_tokens
                    self.total_output_tokens += output_tokens
                    self.total_requests += 1

                # Build metadata
                metadata = {
                    "api_time": elapsed,
                    "stop_reason": stop_reason,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                }

                return text, metadata

            except RateLimitError as e:
                last_error = e
                wait_time = self.retry_delay * (2 ** attempt)
                print(f"Rate limit hit, waiting {wait_time:.1f}s before retry...")
                await asyncio.sleep(wait_time)

            except APIError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    print(f"API error: {e}, retrying in {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise

        raise last_error

    @property
    def name(self) -> str:
        return f"anthropic/{self.model}"

    def get_cost(self) -> float:
        """Calculate total cost in USD."""
        pricing = PRICING.get(self.model, {"input": 0, "output": 0})
        return (
            self.total_input_tokens * pricing["input"] +
            self.total_output_tokens * pricing["output"]
        ) / 1_000_000

    def get_usage_summary(self) -> dict:
        """Get detailed usage summary."""
        pricing = PRICING.get(self.model, {"input": 0, "output": 0})
        return {
            "model": self.model,
            "total_requests": self.total_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "input_cost": self.total_input_tokens * pricing["input"] / 1_000_000,
            "output_cost": self.total_output_tokens * pricing["output"] / 1_000_000,
            "total_cost": self.get_cost(),
        }


# ============================================================================
# STUB BACKENDS (FOR FUTURE IMPLEMENTATION)
# ============================================================================

class OpenAIBackend(CompletionBackend):
    """
    OpenAI API backend for GPT models.

    NOT YET IMPLEMENTED - interface documented for future use.

    Will require OPENAI_API_KEY environment variable.
    """

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        raise NotImplementedError(
            "OpenAIBackend not yet implemented. "
            "Interface documented for future use."
        )

    def generate(self, messages: list[dict], max_tokens: int) -> tuple[str, dict]:
        raise NotImplementedError()

    @property
    def name(self) -> str:
        return f"openai/{self.model}"


class GoogleBackend(CompletionBackend):
    """
    Google AI backend for Gemini models.

    NOT YET IMPLEMENTED - interface documented for future use.

    Will require GOOGLE_API_KEY environment variable.
    """

    def __init__(self, model: str = "gemini-1.5-pro"):
        self.model = model
        raise NotImplementedError(
            "GoogleBackend not yet implemented. "
            "Interface documented for future use."
        )

    def generate(self, messages: list[dict], max_tokens: int) -> tuple[str, dict]:
        raise NotImplementedError()

    @property
    def name(self) -> str:
        return f"google/{self.model}"


class XAIBackend(CompletionBackend):
    """
    xAI backend for Grok models.

    NOT YET IMPLEMENTED - interface documented for future use.

    Will require XAI_API_KEY environment variable.
    """

    def __init__(self, model: str = "grok-2"):
        self.model = model
        raise NotImplementedError(
            "XAIBackend not yet implemented. "
            "Interface documented for future use."
        )

    def generate(self, messages: list[dict], max_tokens: int) -> tuple[str, dict]:
        raise NotImplementedError()

    @property
    def name(self) -> str:
        return f"xai/{self.model}"


class DeepSeekBackend(CompletionBackend):
    """
    DeepSeek API backend.

    NOT YET IMPLEMENTED - interface documented for future use.

    Will require DEEPSEEK_API_KEY environment variable.
    """

    def __init__(self, model: str = "deepseek-chat"):
        self.model = model
        raise NotImplementedError(
            "DeepSeekBackend not yet implemented. "
            "Interface documented for future use."
        )

    def generate(self, messages: list[dict], max_tokens: int) -> tuple[str, dict]:
        raise NotImplementedError()

    @property
    def name(self) -> str:
        return f"deepseek/{self.model}"


class HFInferenceBackend(CompletionBackend):
    """
    HuggingFace Inference API backend (serverless).

    NOT YET IMPLEMENTED - interface documented for future use.

    Will require HF_TOKEN environment variable.
    """

    def __init__(self, model: str = "meta-llama/Llama-3.2-3B-Instruct"):
        self.model = model
        raise NotImplementedError(
            "HFInferenceBackend not yet implemented. "
            "Interface documented for future use."
        )

    def generate(self, messages: list[dict], max_tokens: int) -> tuple[str, dict]:
        raise NotImplementedError()

    @property
    def name(self) -> str:
        return f"hf-inference/{self.model}"


# ============================================================================
# BACKEND FACTORY
# ============================================================================

BACKENDS = {
    "anthropic": AnthropicBackend,
    "openai": OpenAIBackend,
    "google": GoogleBackend,
    "xai": XAIBackend,
    "deepseek": DeepSeekBackend,
    "hf-inference": HFInferenceBackend,
}


def get_backend(backend_name: str, model: Optional[str] = None, **kwargs) -> CompletionBackend:
    """
    Factory function to get a backend by name.

    Args:
        backend_name: Backend identifier (e.g., "anthropic", "openai")
        model: Model ID to use (uses backend default if not specified)
        **kwargs: Additional arguments passed to backend constructor

    Returns:
        Initialized CompletionBackend instance
    """
    if backend_name not in BACKENDS:
        available = ", ".join(BACKENDS.keys())
        raise ValueError(
            f"Unknown backend: {backend_name}. Available: {available}"
        )

    backend_class = BACKENDS[backend_name]

    if model:
        return backend_class(model=model, **kwargs)
    else:
        return backend_class(**kwargs)


# ============================================================================
# CLI TEST
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test completion backends")
    parser.add_argument(
        "--backend",
        default="anthropic",
        choices=list(BACKENDS.keys()),
        help="Backend to test"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model to use (uses backend default if not specified)"
    )
    parser.add_argument(
        "--prompt",
        default="Say hello in exactly 5 words.",
        help="Prompt to test"
    )
    args = parser.parse_args()

    print(f"Testing {args.backend} backend...")

    try:
        backend = get_backend(args.backend, args.model)
        print(f"Model: {backend.name}")
        print(f"Prompt: {args.prompt}")
        print()

        response, metadata = backend.generate(
            messages=[{"role": "user", "content": args.prompt}],
            max_tokens=100
        )

        print(f"Response: {response}")
        print()
        print(f"Metadata: {metadata}")
        print(f"Usage: {backend.get_usage_summary()}")

    except NotImplementedError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error: {e}")
