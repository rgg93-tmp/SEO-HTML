from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient
from typing import Dict, Any


class DescriptionAgent(AssistantAgent):
    def __init__(
        self,
        name="description_agent",
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
            system_message="You are a real estate copywriting expert. Only output a single plain description string for the property, between 500 and 700 characters, with no extra content, questions, or options. Do not include any explanations or formatting beyond the required string.",
        )

    def build_user_prompt(self, property_data, language="English", tone="professional"):
        return f"""Respond exclusively in {language}. Do not use any other language.
Generate a comprehensive property description for the following property in {language} with a {tone} tone.
Only output the description text (500-700 characters). Do not include any extra content, questions, or options.

Language: {language}
Tone: {tone}
Property data: {property_data}
"""

    def build_refinement_prompt(
        self,
        property_data: Dict[str, Any],
        current_description: str,
        suggestion: str,
        language="English",
        tone="professional",
    ) -> str:
        return f"""
Respond exclusively in {language}. Do not use any other language.
Refine this property description based on the suggestion provided. Keep it between 500-700 characters.
Write in {language} with a {tone} tone. Only output the improved description string.

Current description: {current_description}

Property data: {property_data}

Suggestion for improvement:
- {suggestion}

Provide only the refined description with no explanations or extra content.
"""

    async def generate_initial(self, property_data: Dict[str, Any], language="English", tone="professional") -> str:
        """Generate initial description draft."""
        prompt = self.build_user_prompt(property_data, language, tone)
        response = await self.run(task=prompt)
        return response.messages[-1].content.strip()

    async def refine(
        self,
        property_data: Dict[str, Any],
        current_description: str,
        suggestion: str,
        language="English",
        tone="professional",
    ) -> str:
        """Refine existing description based on a suggestion."""
        prompt = self.build_refinement_prompt(property_data, current_description, suggestion, language, tone)
        response = await self.run(task=prompt)
        return response.messages[-1].content.strip()
