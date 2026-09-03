"""Abstract LLM interface for AEGIS."""

from abc import ABC, abstractmethod

from pydantic import BaseModel


class LLMClient(ABC):
    """Abstract interface for LLM-backed text generation.

    AEGIS uses this interface so the rest of the code can depend on a stable
    contract instead of a specific provider SDK. As long as a concrete client
    implements ``generate(prompt) -> str``, AEGIS can swap providers without
    changing the callers.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        response_schema: type[BaseModel] | None = None,
    ) -> str:
        """Send a prompt to the model and return the generated text.

        Args:
            prompt: The instruction or prompt to send to the model.
            response_schema: Optional Pydantic model class used to request
                structured output from the model. If provided, the client may
                use this schema to guide output format. If None, the client
                may still request JSON output where supported.

        Returns:
            The model-generated string response.

        Raises:
            NotImplementedError: If a concrete subclass does not override this
                method.
        """
        ...
