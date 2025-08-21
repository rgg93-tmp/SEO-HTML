from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient
from typing import Dict, Any
from config.options import TONE_OPTIONS, LANGUAGE_OPTIONS

# Language-specific prompts
META_DESCRIPTION_PROMPTS = {
    "en": {
        "initial": """Respond exclusively in {language_name}. Generate a meta description for the following property with a {tone} tone.
Tone details: {tone_description}

CRITICAL: The meta description must be maximum 155 characters (including spaces). This is essential for SEO and search engine display.

Only output the meta description string. Do not include any extra content, questions, or options.

Language: {language_name}
Tone: {tone}
Property data: {property_data}

Generate a meta description (max 155 characters):""",
        "refinement": """Respond exclusively in {language_name}. Refine this property meta description based on the suggestion provided.
Write with a {tone} tone. Tone details: {tone_description}

CRITICAL: The meta description must be maximum 155 characters (including spaces). This is essential for SEO and search engine display.

Language: {language_name}
Current meta description: {current_content}

Property data: {property_data}

Suggestion for improvement:
- {suggestion}

Provide only the refined meta description (max 155 characters) with no explanations or extra content.""",
    },
    "es": {
        "initial": """Responde exclusivamente en {language_name}. Genera una meta descripción para la siguiente propiedad con un tono {tone}.
Detalles del tono: {tone_description}

CRÍTICO: La meta descripción debe tener máximo 155 caracteres (incluyendo espacios). Esto es esencial para SEO y visualización en motores de búsqueda.

Solo proporciona la meta descripción. No incluyas contenido adicional, preguntas u opciones.

Idioma: {language_name}
Tono: {tone}
Datos de la propiedad: {property_data}

Genera una meta descripción (máx 155 caracteres):""",
        "refinement": """Responde exclusivamente en {language_name}. Refina esta meta descripción de propiedad basándote en la sugerencia proporcionada.
Escribe con un tono {tone}. Detalles del tono: {tone_description}

CRÍTICO: La meta descripción debe tener máximo 155 caracteres (incluyendo espacios). Esto es esencial para SEO y visualización en motores de búsqueda.

Idioma: {language_name}
Meta descripción actual: {current_content}

Datos de la propiedad: {property_data}

Sugerencia de mejora:
- {suggestion}

Proporciona solo la meta descripción refinada (máx 155 caracteres) sin explicaciones o contenido adicional.""",
    },
    "pt": {
        "initial": """Responda exclusivamente em {language_name}. Gere uma meta descrição para a seguinte propriedade com um tom {tone}.
Detalhes do tom: {tone_description}

CRÍTICO: A meta descrição deve ter máximo 155 caracteres (incluindo espaços). Isso é essencial para SEO e exibição em motores de busca.

Apenas forneça a meta descrição. Não inclua conteúdo extra, perguntas ou opções.

Idioma: {language_name}
Tom: {tone}
Dados da propriedade: {property_data}

Gere uma meta descrição (máx 155 caracteres):""",
        "refinement": """Responda exclusivamente em {language_name}. Refine esta meta descrição de propriedade com base na sugestão fornecida.
Escreva com um tom {tone}. Detalhes do tom: {tone_description}

CRÍTICO: A meta descrição deve ter máximo 155 caracteres (incluindo espaços). Isso é essencial para SEO e exibição em motores de busca.

Idioma: {language_name}
Meta descrição atual: {current_content}

Dados da propriedade: {property_data}

Sugerência de melhoria:
- {suggestion}

Forneça apenas a meta descrição refinada (máx 155 caracteres) sem explicações ou conteúdo extra.""",
    },
}


class MetaDescriptionAgent(AssistantAgent):
    def __init__(
        self,
        name="meta_description_agent",
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
            system_message="You are a real estate SEO expert. Only output a single plain meta description string for the property, under 155 characters, with no extra content, questions, or options. Do not include any explanations or formatting beyond the required string.",
        )

    def build_user_prompt(self, property_data, language="en", tone="family-oriented"):
        tone_description = TONE_OPTIONS.get(tone, {}).get("description", "")
        language_name = LANGUAGE_OPTIONS.get(language, {}).get("name", language)
        prompt_template = META_DESCRIPTION_PROMPTS.get(language, META_DESCRIPTION_PROMPTS["en"])["initial"]
        return prompt_template.format(
            tone=tone, tone_description=tone_description, language_name=language_name, property_data=property_data
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
        language_name = LANGUAGE_OPTIONS.get(language, {}).get("name", language)
        prompt_template = META_DESCRIPTION_PROMPTS.get(language, META_DESCRIPTION_PROMPTS["en"])["refinement"]
        return prompt_template.format(
            tone=tone,
            tone_description=tone_description,
            language_name=language_name,
            current_content=current_content,
            property_data=property_data,
            suggestion=suggestion,
        )

    async def generate_initial(self, property_data: Dict[str, Any], language="en", tone="family-oriented") -> str:
        """Generate initial meta description draft."""
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
        """Refine existing meta description based on a suggestion."""
        prompt = self.build_refinement_prompt(
            property_data=property_data,
            current_content=current_content,
            suggestion=suggestion,
            language=language,
            tone=tone,
        )
        response = await self.run(task=prompt)
        return response.messages[-1].content.strip()
