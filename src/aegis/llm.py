"""Abstract LLM interface for AEGIS."""

from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Abstract interface for LLM-backed text generation.

    AEGIS uses this interface so the rest of the code can depend on a stable
    contract instead of a specific provider SDK. As long as a concrete client
    implements ``generate(prompt) -> str``, AEGIS can swap providers without
    changing the callers.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send a prompt to the model and return the generated text.

        Args:
            prompt: The instruction or prompt to send to the model.

        Returns:
            The model-generated string response.

        Raises:
            NotImplementedError: If a concrete subclass does not override this
                method.
        """
        ...
