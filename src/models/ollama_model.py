from typing import List
from openai import OpenAI

from .base_language_model import BaseLanguageModel


class OllamaModel(BaseLanguageModel):
    """
    Ollama language model implementation using OpenAI-compatible client.

    This class provides a direct interface to Ollama models through the OpenAI client.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ):
        """
        Initialize the Ollama model.

        Args:
            model_name: Ollama model name (e.g., "gemma3:1b-it-qat")
            base_url: Ollama's OpenAI-compatible URL (must include /v1)
            api_key: API key (dummy for Ollama, but required by OpenAI client)
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
        """
        super().__init__(model_name, **kwargs)

        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize OpenAI client pointed at Ollama
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a single text response.

        Args:
            prompt: The input prompt
            **kwargs: Additional generation parameters (temperature, max_tokens, etc.)

        Returns:
            str: Generated text
        """
        try:
            # Use provided kwargs or fall back to instance defaults
            temperature = kwargs.get("temperature", self.temperature)
            max_tokens = kwargs.get("max_tokens", self.max_tokens)

            messages = [{"role": "user", "content": prompt}]

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            raise RuntimeError(f"Failed to generate text with Ollama: {str(e)}")

    def generate_multiple(self, prompt: str, num_responses: int = 1, **kwargs) -> List[str]:
        """
        Generate multiple text responses.

        Args:
            prompt: The input prompt
            num_responses: Number of responses to generate
            **kwargs: Additional generation parameters

        Returns:
            List[str]: List of generated texts
        """
        try:
            responses = []
            for _ in range(num_responses):
                response_text = self.generate(prompt, **kwargs)
                responses.append(response_text)

            return responses

        except Exception as e:
            raise RuntimeError(f"Failed to generate multiple responses with Ollama: {str(e)}")

    def test_connection(self) -> bool:
        """
        Test the connection to the Ollama server.

        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            # Try a simple generation to test the connection
            test_response = self.generate("Hello", max_tokens=5)
            return True
        except Exception:
            return False

    def get_model_info(self) -> dict:
        """
        Get detailed information about the model.

        Returns:
            dict: Model information including Ollama-specific details
        """
        base_info = super().get_model_info()
        base_info.update(
            {
                "base_url": self.base_url,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "model_type": "ollama",
            }
        )
        return base_info
