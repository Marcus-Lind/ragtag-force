"""Anthropic API client wrapper.

Provides a configured Anthropic client for generating LLM responses.
Handles API key validation and graceful error reporting.
"""

from typing import Optional

import anthropic

from src.config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL


_UNSET = object()


class LLMClient:
    """Wrapper around the Anthropic API client."""

    def __init__(
        self,
        api_key: Optional[str] = _UNSET,
        model: Optional[str] = None,
    ) -> None:
        """Initialize the LLM client.

        Args:
            api_key: Anthropic API key. Defaults to config value.
                     Pass empty string to explicitly disable.
            model: Model identifier. Defaults to config value.
        """
        self.api_key = ANTHROPIC_API_KEY if api_key is _UNSET else api_key
        self.model = model or ANTHROPIC_MODEL
        self._client: Optional[anthropic.Anthropic] = None

    @property
    def is_configured(self) -> bool:
        """Check if the API key is set."""
        return bool(self.api_key and self.api_key != "sk-ant-your-key-here")

    @property
    def client(self) -> anthropic.Anthropic:
        """Get or create the Anthropic client."""
        if self._client is None:
            if not self.is_configured:
                raise ValueError(
                    "Anthropic API key not configured. "
                    "Set ANTHROPIC_API_KEY in your .env file."
                )
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> str:
        """Generate a response from the LLM.

        Args:
            system_prompt: System instructions for the model.
            user_message: User query with context.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature (lower = more focused).

        Returns:
            Generated text response.

        Raises:
            ValueError: If API key is not configured.
            anthropic.APIError: If the API call fails.
        """
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text


# Module-level singleton
_default_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get the default LLM client singleton.

    Returns:
        Configured LLMClient instance.
    """
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client
