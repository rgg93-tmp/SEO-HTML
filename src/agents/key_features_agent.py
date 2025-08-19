from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient


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
            system_message="You are a real estate copywriting expert. Only output a single plain string with 3 to 5 key features for the property, separated by commas, with no extra content, questions, or options. Do not include any explanations or formatting beyond the required string.",
        )

    def build_user_prompt(self, property_data):
        return (
            "Generate a list of 3 to 5 key features for the following property, separated by commas. Only output the key features string. Do not include any extra content, questions, or options.\n"
            f"Property data: {property_data}"
        )
