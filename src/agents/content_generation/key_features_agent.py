from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient
from typing import Dict, Any


class KeyFeaturesAgent(AssistantAgent):
    def __init__(
        self,
        name="key_features_agent",
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
            system_message="You are a real estate copywriting expert. Generate 3-5 key property features as a simple list, with each feature on a new line. Start each line with a hyphen (-). Only output the list, no extra content or explanations.",
        )

    def build_user_prompt(self, property_data, language="English", tone="professional"):
        return f"""Respond exclusively in {language}. Do not use any other language.
Generate key features for the following property in {language} with a {tone} tone.
Output exactly 3-5 key features as a simple list. Each feature should be on a new line starting with a hyphen (-).

Example format:
- 120 sqm of living space
- 3 bedrooms and 2 bathrooms
- Private balcony
- Elevator access

Language: {language}
Tone: {tone}
Property data: {property_data}

Only output the list, no explanations or extra content:"""

    def build_refinement_prompt(
        self,
        property_data: Dict[str, Any],
        current_features: str,
        suggestion: str,
        language="English",
        tone="professional",
    ) -> str:
        return f"""
Respond exclusively in {language}. Do not use any other language.
Refine this property key features list based on the suggestion provided. Keep it concise (3-5 features).
Write in {language} with a {tone} tone. 

Output exactly 3-5 key features as a simple list. Each feature should be on a new line starting with a hyphen (-).

Example format:
- 120 sqm of living space
- 3 bedrooms and 2 bathrooms
- Private balcony

Current key features: {current_features}

Property data: {property_data}

Suggestion for improvement:
- {suggestion}

Only output the improved list, no explanations or extra content:"""

    async def generate_initial(self, property_data: Dict[str, Any], language="English", tone="professional") -> str:
        """Generate initial key features draft."""
        prompt = self.build_user_prompt(property_data, language, tone)
        response = await self.run(task=prompt)
        return response.messages[-1].content.strip()

    async def refine(
        self,
        property_data: Dict[str, Any],
        current_features: str,
        suggestion: str,
        language="English",
        tone="professional",
    ) -> str:
        """Refine existing key features based on a suggestion."""
        prompt = self.build_refinement_prompt(property_data, current_features, suggestion, language, tone)
        response = await self.run(task=prompt)
        return response.messages[-1].content.strip()
