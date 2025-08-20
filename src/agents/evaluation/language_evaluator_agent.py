from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient
from typing import Dict, Any
import asyncio


class LanguageEvaluatorAgent(AssistantAgent):
    """LLM-based agent for evaluating if content matches the expected language."""

    def __init__(
        self,
        name="language_evaluator_agent",
        model="gemma3:1b-it-qat",
        model_info={
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "unknown",
            "structured_output": True,
        },
    ):
        model_client = OllamaChatCompletionClient(model=model, model_info=model_info)
        super().__init__(
            name=name,
            model_client=model_client,
            system_message="You are a language detection expert. You evaluate if text content matches the expected language. Always respond with only a score from 0 to 100, where 100 means perfect language match and 0 means completely wrong language.",
        )

    def build_evaluation_prompt(self, content: str, expected_language: str = "en") -> str:
        """Build prompt for language evaluation."""
        language_names = {"en": "English", "es": "Spanish", "pt": "Portuguese"}
        expected_language_name = language_names.get(expected_language, "English")

        return f"""Evaluate if the following content is written in {expected_language_name}.

Expected language: {expected_language_name}
Content to evaluate: {content}

Respond with only a number from 0 to 100:
- 100: Content is perfectly written in {expected_language_name}
- 80-99: Content is mostly in {expected_language_name} with minor issues
- 50-79: Content is partially in {expected_language_name}
- 20-49: Content is mostly in another language
- 0-19: Content is completely in the wrong language

Score:"""

    async def evaluate(self, content: str, expected_language: str = "en") -> Dict[str, Any]:
        """
        Evaluate if content matches the expected language.

        Args:
            content: Text content to evaluate
            expected_language: Expected language code (en, es, pt)

        Returns:
            Standardized evaluation results dictionary
        """
        prompt = self.build_evaluation_prompt(content, expected_language)
        response = await self.run(task=prompt)

        try:
            # Extract score from response
            response_text = response.messages[-1].content.strip()
            # Try to extract number from response
            import re

            score_match = re.search(r"\b(\d+)\b", response_text)
            if score_match:
                raw_score = int(score_match.group(1))
                score = min(100, max(0, raw_score)) / 100.0  # Convert to 0-1 scale
            else:
                score = 0.5  # Default neutral score if no number found
        except (ValueError, AttributeError):
            score = 0.5  # Default neutral score on error

        language_names = {"en": "English", "es": "Spanish", "pt": "Portuguese"}
        expected_language_name = language_names.get(expected_language, expected_language)

        return {
            "evaluator": "LanguageEvaluatorAgent",
            "score": score,
            "passed": score >= 0.7,
            "summary": f"Language evaluation for {expected_language_name}: {score:.2f}",
            "findings": [
                {
                    "component": "Language Detection",
                    "message": f"Content language match score: {score:.2f}",
                    "severity": "info" if score >= 0.7 else "warning",
                }
            ],
        }
