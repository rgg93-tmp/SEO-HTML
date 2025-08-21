from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient
from typing import Dict, Any
from config.options import TONE_OPTIONS, LANGUAGE_OPTIONS

# Language-specific prompts
CALL_TO_ACTION_PROMPTS = {
    "en": {
        "initial": """Respond exclusively in {language_name}. Generate a compelling call to action for the following property with a {tone} tone.
Tone details: {tone_description}
Only output the call to action text. Do not include any extra content, questions, or options.

Language: {language_name}
Tone: {tone}
Property data: {property_data}""",
        "refinement": """Respond exclusively in {language_name}. Refine this call to action based on the suggestion provided. Keep it compelling and action-oriented.
Write with a {tone} tone. Tone details: {tone_description}

Language: {language_name}
Current call to action: {current_content}

Property data: {property_data}

Suggestion for improvement:
- {suggestion}

Provide only the refined call to action with no explanations or extra content.""",
    },
    "es": {
        "initial": """Responde exclusivamente en {language_name}. Genera una llamada a la acción convincente para la siguiente propiedad con un tono {tone}.
Detalles del tono: {tone_description}
Solo proporciona el texto de la llamada a la acción. No incluyas contenido adicional, preguntas u opciones.

Idioma: {language_name}
Tono: {tone}
Datos de la propiedad: {property_data}""",
        "refinement": """Responde exclusivamente en {language_name}. Refina esta llamada a la acción basándote en la sugerencia proporcionada. Manténla convincente y orientada a la acción.
Escribe con un tono {tone}. Detalles del tono: {tone_description}

Idioma: {language_name}
Llamada a la acción actual: {current_content}

Datos de la propiedad: {property_data}

Sugerencia de mejora:
- {suggestion}

Proporciona solo la llamada a la acción refinada sin explicaciones o contenido adicional.""",
    },
    "pt": {
        "initial": """Responda exclusivamente em {language_name}. Gere uma chamada à ação convincente para a seguinte propriedade com um tom {tone}.
Detalhes do tom: {tone_description}
Apenas forneça o texto da chamada à ação. Não inclua conteúdo extra, perguntas ou opções.

Idioma: {language_name}
Tom: {tone}
Dados da propriedade: {property_data}""",
        "refinement": """Responda exclusivamente em {language_name}. Refine esta chamada à ação com base na sugestão fornecida. Mantenha-a convincente e orientada à ação.
Escreva com um tom {tone}. Detalhes do tom: {tone_description}

Idioma: {language_name}
Chamada à ação atual: {current_content}

Dados da propriedade: {property_data}

Sugerência de melhoria:
- {suggestion}

Forneça apenas a chamada à ação refinada sem explicações ou conteúdo extra.""",
    },
}


class CallToActionAgent(AssistantAgent):
    def __init__(
        self,
        name="call_to_action_agent",
        model="gemma3n:e2b",
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

    def build_user_prompt(self, property_data, language="en", tone="family-oriented"):
        tone_description = TONE_OPTIONS.get(tone, {}).get("description", "")
        language_name = LANGUAGE_OPTIONS[language]["name"]
        prompt_template = CALL_TO_ACTION_PROMPTS.get(language, CALL_TO_ACTION_PROMPTS["en"])["initial"]
        return prompt_template.format(
            language_name=language_name, tone=tone, tone_description=tone_description, property_data=property_data
        )

    def build_refinement_prompt(
        self,
        property_data: Dict[str, Any],
        current_content: str,
        suggestion: str,
        language="en",
        tone="family-oriented",
    ) -> str:
        tone_description = TONE_OPTIONS.get(tone, {}).get("description", "")
        language_name = LANGUAGE_OPTIONS[language]["name"]
        prompt_template = CALL_TO_ACTION_PROMPTS.get(language, CALL_TO_ACTION_PROMPTS["en"])["refinement"]
        return prompt_template.format(
            language_name=language_name,
            tone=tone,
            tone_description=tone_description,
            current_content=current_content,
            property_data=property_data,
            suggestion=suggestion,
        )

    async def generate_initial(self, property_data: Dict[str, Any], language="en", tone="family-oriented") -> str:
        """Generate initial call to action draft."""
        prompt = self.build_user_prompt(property_data=property_data, language=language, tone=tone)
        response = await self.run(task=prompt)
        return response.messages[-1].content.strip()

    async def refine(
        self,
        property_data: Dict[str, Any],
        current_content: str,
        suggestion: str,
        language="en",
        tone="family-oriented",
    ) -> str:
        """Refine existing call to action based on a suggestion."""
        prompt = self.build_refinement_prompt(
            property_data=property_data,
            current_content=current_content,
            suggestion=suggestion,
            language=language,
            tone=tone,
        )
        response = await self.run(task=prompt)
        return response.messages[-1].content.strip()
