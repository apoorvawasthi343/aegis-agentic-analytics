"""Local Ollama LLM client for AEGIS."""

from ollama import Client
from pydantic import BaseModel

from src.aegis.llm import LLMClient


class OllamaClient(LLMClient):
    """LLM client backed by a local Ollama instance.

    Uses the official ``ollama`` Python package to talk to a local Ollama
    server. The default model and host match common local setups.
    """

    def __init__(
        self,
        model: str = "qwen3:1.7b",
        host: str = "http://localhost:11434",
    ) -> None:
        """Initialize the local Ollama client.

        Args:
            model: The Ollama model tag to use for generation.
            host: The base URL of the local Ollama server.
        """
        self.model = model
        self.host = host

        self._client = Client(host=self.host)

    def generate(
        self,
        prompt: str,
        response_schema: type[BaseModel] | None = None,
    ) -> str:
        """Send a prompt to the local Ollama model and return the response.

        Args:
            prompt: The instruction or prompt to send to the model.
            response_schema: Optional Pydantic model class used to request
                structured output. When provided, the model is instructed to
                output JSON matching that schema.

        Returns:
            The model-generated string response.

        Raises:
            RuntimeError: If the local Ollama server cannot be reached,
                the request fails, or the response is missing the expected
                ``response`` field.
        """
        format_spec: dict | str = (
            response_schema.model_json_schema()
            if response_schema is not None
            else "json"
        )

        try:
            response = self._client.generate(
                model=self.model,
                prompt=prompt,
                stream=False,
                format=format_spec,
                options={"temperature": 0.1},
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not reach the local Ollama server. "
                "Please verify that: Ollama is running, "
                "qwen3:1.7b is installed, and localhost:11434 is reachable."
            ) from exc

        response_text = response.get("response")
        if response_text is None:
            raise RuntimeError(
                "Received an unexpected response from Ollama."
            )

        return response_text
