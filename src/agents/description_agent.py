from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient


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

    def build_user_prompt(self, property_data):
        return (
            "Generate a property description (500-700 characters) for the following property. Only output the description string. Do not include any extra content, questions, or options.\n"
            f"Property data: {property_data}"
        )