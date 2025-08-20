from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseEvaluator(ABC):
    """
    Abstract base class for all evaluators.
    Ensures that each evaluator has a consistent interface.
    """

    @abstractmethod
    async def evaluate(self, **kwargs) -> Dict[str, Any]:
        """
        Run the evaluation.

        Subclasses will implement this method with specific arguments
        (e.g., html_content, property_data, language, tone).

        Returns:
            A dictionary with standardized evaluation results.
        """
        pass
