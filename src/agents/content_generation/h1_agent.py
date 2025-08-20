from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient
from typing import Dict, Any
from config.options import TONE_OPTIONS


class H1Agent(AssistantAgent):
    def __init__(
        self,
        name="h1_agent",
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
            system_message="You are a real estate SEO expert. Only output a single plain headline string for the property, with no extra content, questions, or options. Do not include any explanations or formatting beyond the required string.",
        )

    def build_user_prompt(self, property_data, language="English", tone="professional"):
        tone_description = TONE_OPTIONS.get(tone, {}).get("description", "")
        return f"""Respond exclusively in {language}. Do not use any other language.
Generate an H1 tag for the following property in {language} with a {tone} tone.
Tone details: {tone_description}
Only output the H1 string content (without HTML tags). Do not include any extra content, questions, or options.

Language: {language}
Tone: {tone}
Property data: {property_data}
"""

    def build_refinement_prompt(
        self,
        property_data: Dict[str, Any],
        current_content: str,
        suggestion: str,
        language="English",
        tone="professional",
    ) -> str:
        tone_description = TONE_OPTIONS.get(tone, {}).get("description", "")
        return f"""
Respond exclusively in {language}. Do not use any other language.
Refine this property H1 heading based on the suggestion provided. Keep it under 70 characters.
Write in {language} with a {tone} tone. Tone details: {tone_description}
Only output the improved H1 string.

Current H1: {current_content}

Property data: {property_data}

Suggestion for improvement:
- {suggestion}

Provide only the refined H1 with no explanations or extra content.
"""

    async def generate_initial(self, property_data: Dict[str, Any], language="English", tone="professional") -> str:
        """Generate initial H1 draft."""
        prompt = self.build_user_prompt(property_data=property_data, language=language, tone=tone)
        response = await self.run(task=prompt)
        return response.messages[-1].content.strip()

    async def refine(
        self,
        property_data: Dict[str, Any],
        current_content: str,
        suggestion: str,
        language="English",
        tone="professional",
    ) -> str:
        """Refine existing H1 based on a suggestion."""
        prompt = self.build_refinement_prompt(
            property_data=property_data,
            current_content=current_content,
            suggestion=suggestion,
            language=language,
            tone=tone,
        )
        response = await self.run(task=prompt)
        return response.messages[-1].content.strip()
