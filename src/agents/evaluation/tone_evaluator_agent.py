from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient
from typing import Dict, Any
from config.options import TONE_OPTIONS


class ToneEvaluatorAgent(AssistantAgent):
    """Agent that evaluates tone and style appropriateness for real estate content."""

    def __init__(
        self,
        name="tone_evaluator_agent",
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
            system_message="You are a real estate content expert. Evaluate the tone, style, and appropriateness of real estate content. Provide a score from 0.0 to 1.0 and specific feedback on tone quality. Only output a JSON with 'score' and 'feedback' fields.",
        )

    async def evaluate(self, content: str, expected_tone: str = "professional") -> Dict[str, Any]:
        """
        Evaluate tone appropriateness for content.

        Args:
            content: Text content to evaluate
            expected_tone: Expected tone (professional, friendly, luxury, etc.)

        Returns:
            Standardized evaluation results dictionary
        """
        prompt = self.build_tone_prompt(content=content, expected_tone=expected_tone)
        response = await self.run(task=prompt)

        # Parse response (expecting JSON format)
        try:
            import json

            result = json.loads(s=response.messages[-1].content.strip())
            score = float(result.get("score", 0.5))
            feedback = result.get("feedback", "No feedback available")
        except (json.JSONDecodeError, ValueError, KeyError):
            # Fallback parsing
            response_text = response.messages[-1].content.strip()
            score = self._extract_score_from_text(text=response_text)
            feedback = "Tone evaluation completed"

        return {
            "evaluator": "ToneEvaluatorAgent",
            "score": score,
            "passed": score >= 0.7,
            "summary": f"Tone evaluation for {expected_tone}: {score:.2f}",
            "findings": [
                {"component": "Tone Analysis", "message": feedback, "severity": "info" if score >= 0.7 else "warning"}
            ],
        }

    def build_tone_prompt(self, content: str, expected_tone: str = "professional") -> str:
        """Build prompt for tone evaluation."""
        tone_description = TONE_OPTIONS.get(expected_tone, {}).get(
            "description", f"Tone matching '{expected_tone}' style"
        )

        return f"""Evaluate the tone and style of this real estate content.

Expected Tone: {expected_tone} ({tone_description})

Content: "{content}"

Analyze if the content matches the expected tone. Consider:
- Word choice and vocabulary
- Sentence structure and flow
- Overall feel and personality
- Target audience appropriateness

Respond with a JSON containing:
- "score": a number from 0.0 to 1.0 (1.0 = perfectly matches expected tone)
- "feedback": specific feedback about tone quality and suggestions

Example response: {{"score": 0.85, "feedback": "Content matches professional tone well, but could be more engaging."}}"""

    def _extract_score_from_text(self, text: str) -> float:
        """Extract numeric score from text response as fallback."""
        import re

        # Look for decimal numbers between 0 and 1
        score_match = re.search(r"\b(0?\.\d+|1\.0)\b", text)
        if score_match:
            return float(score_match.group(1))

        # Look for percentages and convert
        percent_match = re.search(r"\b(\d{1,3})%\b", text)
        if percent_match:
            return float(percent_match.group(1)) / 100.0

        return 0.5  # Default neutral score
