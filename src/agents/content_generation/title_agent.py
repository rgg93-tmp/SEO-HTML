from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient
from typing import Dict, Any


class TitleAgent(AssistantAgent):
    def __init__(
        self,
        name="title_agent",
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
            system_message="You are a real estate SEO expert. Only output a single plain title string for the property, under 60 characters, with no extra content, questions, or options. Do not include any explanations or formatting beyond the required string.",
        )

    def build_user_prompt(self, property_data, language="English", tone="professional"):
        return f"""Respond exclusively in {language}. Do not use any other language.
Generate a title for the following property in {language} with a {tone} tone. 
Only output the title string, under 60 characters. Do not include any extra content, questions, or options.

Language: {language}
Tone: {tone}
Property data: {property_data}
"""

    def build_refinement_prompt(
        self,
        property_data: Dict[str, Any],
        current_title: str,
        suggestion: str,
        language="English",
        tone="professional",
    ) -> str:
        return f"""
Respond exclusively in {language}. Do not use any other language.
Refine this property title based on the suggestion provided. Only output the improved title string, under 60 characters.
Write in {language} with a {tone} tone.

Current title: {current_title}

Property data: {property_data}

Suggestion for improvement:
- {suggestion}

Provide only the refined title with no explanations or extra content.
"""

    async def generate_initial(self, property_data: Dict[str, Any], language="English", tone="professional") -> str:
        """Generate initial title draft."""
        prompt = self.build_user_prompt(property_data, language, tone)
        response = await self.run(task=prompt)
        return response.messages[-1].content.strip()

    async def refine(
        self,
        property_data: Dict[str, Any],
        current_title: str,
        suggestion: str,
        language="English",
        tone="professional",
    ) -> str:
        """Refine existing title based on a suggestion."""
        prompt = self.build_refinement_prompt(property_data, current_title, suggestion, language, tone)
        response = await self.run(task=prompt)
        return response.messages[-1].content.strip()
