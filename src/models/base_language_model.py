from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseLanguageModel(ABC):
    """
    Base class for language models.

    This provides a clean interface for different language model implementations.
    """

    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the language model.

        Args:
            model_name: Name/identifier of the model
            **kwargs: Additional model-specific parameters
        """
        self.model_name = model_name
        self.config = kwargs

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text based on a prompt.

        Args:
            prompt: The input prompt
            **kwargs: Additional generation parameters

        Returns:
            str: Generated text
        """
        pass

    @abstractmethod
    def generate_multiple(self, prompt: str, num_responses: int = 1, **kwargs) -> List[str]:
        """
        Generate multiple text responses based on a prompt.

        Args:
            prompt: The input prompt
            num_responses: Number of responses to generate
            **kwargs: Additional generation parameters

        Returns:
            List[str]: List of generated texts
        """
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.

        Returns:
            Dict[str, Any]: Model information
        """
        return {"model_name": self.model_name, "config": self.config}
