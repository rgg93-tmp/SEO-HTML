from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient
from typing import Dict, Any


class CallToActionAgent(AssistantAgent):
    def __init__(
        self,
        name="call_to_action_agent",
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
            system_message="You are a real estate copywriting expert. Only output a single plain call-to-action string for the property, with no extra content, questions, or options. Do not include any explanations or formatting beyond the required string.",
        )

    def build_user_prompt(self, property_data, language="English", tone="professional"):
        return f"""Respond exclusively in {language}. Do not use any other language.
Generate a compelling call to action for the following property in {language} with a {tone} tone.
Only output the call to action text. Do not include any extra content, questions, or options.

Language: {language}
Tone: {tone}
Property data: {property_data}
"""

    def build_refinement_prompt(
        self, property_data: Dict[str, Any], current_cta: str, suggestion: str, language="English", tone="professional"
    ) -> str:
        return f"""
Respond exclusively in {language}. Do not use any other language.
Refine this call to action based on the suggestion provided. Keep it compelling and action-oriented.
Write in {language} with a {tone} tone. Only output the improved call to action string.

Current call to action: {current_cta}

Property data: {property_data}

Suggestion for improvement:
- {suggestion}

Provide only the refined call to action with no explanations or extra content.
"""

    async def generate_initial(self, property_data: Dict[str, Any], language="English", tone="professional") -> str:
        """Generate initial call to action draft."""
        prompt = self.build_user_prompt(property_data, language, tone)
        response = await self.run(task=prompt)
        return response.messages[-1].content.strip()

    async def refine(
        self, property_data: Dict[str, Any], current_cta: str, suggestion: str, language="English", tone="professional"
    ) -> str:
        """Refine existing call to action based on a suggestion."""
        prompt = self.build_refinement_prompt(property_data, current_cta, suggestion, language, tone)
        response = await self.run(task=prompt)
        return response.messages[-1].content.strip()
