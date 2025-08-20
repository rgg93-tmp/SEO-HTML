from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient
from typing import Dict, Any, List
import asyncio
import json


class FactCheckerAgent(AssistantAgent):
    """Agent that verifies factual accuracy of content against property data."""

    def __init__(
        self,
        name="fact_checker_agent",
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
            system_message="You are a real estate fact-checking expert. Compare content against property data to verify accuracy. Identify factual errors and inconsistencies. Always respond with valid JSON containing 'score' (0.0-1.0) and 'feedback' (a summary of findings).",
        )

    async def evaluate(self, content: str, property_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check factual accuracy of content against property data.

        Args:
            content: Text content to verify
            property_data: Property data to compare against

        Returns:
            Standardized evaluation results dictionary
        """
        prompt = self.build_fact_checking_prompt(content=content, property_data=property_data)
        response = await self.run(task=prompt)

        # Parse response (expecting JSON format)
        try:
            result = json.loads(s=response.messages[-1].content.strip())
            score = float(result.get("score", 0.5))
            feedback = result.get("feedback", "No feedback available.")
        except (json.JSONDecodeError, ValueError, KeyError):
            # Fallback if JSON is invalid
            score = 0.5
            feedback = "Could not parse fact-checking results from the model."

        # Build findings from feedback
        findings = []
        if score < 0.9:  # Add finding only if there are potential issues
            findings.append(
                {"component": "Fact Accuracy", "message": feedback, "severity": "error" if score < 0.7 else "warning"}
            )

        return {
            "evaluator": "FactCheckerAgent",
            "score": score,
            "passed": score >= 0.7,
            "summary": f"Fact checking completed with score: {score:.2f}. {feedback}",
            "findings": findings,
        }

    def build_fact_checking_prompt(self, content: str, property_data: Dict[str, Any]) -> str:
        """Build prompt for fact-checking evaluation."""
        return f"""Compare the following content against the property data and identify any factual errors.

Property Data (Source of Truth):
{json.dumps(property_data, indent=2)}

Content to Verify:
"{content}"

Instructions:
1. Compare the content against the property data.
2. Identify any statements in the content that contradict the property data.
3. Provide a score from 0.0 (many errors) to 1.0 (factually perfect).
4. Provide brief feedback summarizing the errors found.

Respond with a JSON containing:
- "score": a number from 0.0 to 1.0
- "feedback": a string summarizing your findings

Example response for content with errors:
{{"score": 0.6, "feedback": "The content mentions 4 bedrooms, but the data shows only 3. The price is also listed incorrectly."}}

Example response for perfect content:
{{"score": 1.0, "feedback": "The content is factually accurate according to the provided data."}}
"""
