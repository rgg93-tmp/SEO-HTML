from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient


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

    def build_user_prompt(self, property_data):
        return (
            "Generate a call-to-action closing line for the following property. Only output the call-to-action string. Do not include any extra content, questions, or options.\n"
            f"Property data: {property_data}"
        )
