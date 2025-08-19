from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient


class MetaDescriptionAgent(AssistantAgent):
    def __init__(
        self,
        name="meta_description_agent",
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
            system_message="You are a real estate SEO expert. Only output a single plain meta description string for the property, under 155 characters, with no extra content, questions, or options. Do not include any explanations or formatting beyond the required string.",
        )

    def build_user_prompt(self, property_data):
        return (
            "Generate a meta description for the following property. Only output the meta description string, under 155 characters. Do not include any extra content, questions, or options.\n"
            f"Property data: {property_data}"
        )
