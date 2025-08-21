from typing import Dict, List, Any
import asyncio
from agents.evaluation.fact_checker_agent import FactCheckerAgent
from .base_evaluator import BaseEvaluator


class FactEvaluator(BaseEvaluator):
    """Evaluator for fact accuracy using LLM-based fact checking."""

    def __init__(self):
        self.fact_checker_agent = FactCheckerAgent()

    async def evaluate(self, html_content: str, property_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Evaluate factual accuracy of HTML content against property data.

        Args:
            html_content: HTML content to verify
            property_data: Property data to compare against
            **kwargs: Additional parameters

        Returns:
            Standardized evaluation results dictionary
        """
        # Extract text content from HTML
        import re

        text = re.sub(pattern=r"<[^>]+>", repl="", string=html_content)
        text = " ".join(text.split())  # Clean whitespace

        # Run async evaluation using the agent
        result = await self.fact_checker_agent.evaluate(content=text, property_data=property_data)

        return result
