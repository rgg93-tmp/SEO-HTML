from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient


class NeighborhoodAgent(AssistantAgent):
    def __init__(
        self,
        name="neighborhood_agent",
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
            system_message="You are a real estate copywriting expert. Only output a single plain paragraph string about the neighborhood for the property, with no extra content, questions, or options. Do not include any explanations or formatting beyond the required string.",
        )

    def build_user_prompt(self, property_data):
        return (
            "Generate a lifestyle paragraph about the neighborhood for the following property. Only output the paragraph string. Do not include any extra content, questions, or options.\n"
            f"Property data: {property_data}"
        )
        super().__init__(
            name=name,
            model_client=model_client,
            system_message="You are a real estate copywriting expert. Generate lifestyle paragraphs about the neighborhood (<p>), using the provided data.",
        )
