from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient
from typing import Dict, Any
from config.options import TONE_OPTIONS, LANGUAGE_OPTIONS

# Language-specific prompts
TITLE_PROMPTS = {
    "en": {
        "initial": """Respond exclusively in {language_name}. Generate a title for the following property with a {tone} tone.
Tone details: {tone_description}

IMPORTANT: The title must be between 30-60 characters long (including spaces). This is critical for SEO optimization.

Only output the title string. Do not include any extra content, questions, or options.

Language: {language_name}
Tone: {tone}
Property data: {property_data}

Generate a title between 30-60 characters:""",
        "refinement": """Respond exclusively in {language_name}. Refine this property title based on the suggestion provided.
Write with a {tone} tone. Tone details: {tone_description}

IMPORTANT: The refined title must be between 30-60 characters long (including spaces). This is critical for SEO optimization.

Language: {language_name}
Current title: {current_content}

Property data: {property_data}

Suggestion for improvement:
- {suggestion}

Provide only the refined title (30-60 characters) with no explanations or extra content.""",
    },
    "es": {
        "initial": """Responde exclusivamente en {language_name}. Genera un título para la siguiente propiedad con un tono {tone}.
Detalles del tono: {tone_description}

IMPORTANTE: El título debe tener entre 30-60 caracteres (incluyendo espacios). Esto es crítico para optimización SEO.

Solo proporciona el título. No incluyas contenido adicional, preguntas u opciones.

Idioma: {language_name}
Tono: {tone}
Datos de la propiedad: {property_data}

Genera un título de 30-60 caracteres:""",
        "refinement": """Responde exclusivamente en {language_name}. Refina este título de propiedad basándote en la sugerencia proporcionada.
Escribe con un tono {tone}. Detalles del tono: {tone_description}

IMPORTANTE: El título refinado debe tener entre 30-60 caracteres (incluyendo espacios). Esto es crítico para optimización SEO.

Idioma: {language_name}
Título actual: {current_content}

Datos de la propiedad: {property_data}

Sugerencia de mejora:
- {suggestion}

Proporciona solo el título refinado (30-60 caracteres) sin explicaciones o contenido adicional.""",
    },
    "pt": {
        "initial": """Responda exclusivamente em {language_name}. Gere um título para a seguinte propriedade com um tom {tone}.
Detalhes do tom: {tone_description}

IMPORTANTE: O título deve ter entre 30-60 caracteres (incluindo espaços). Isso é crítico para otimização SEO.

Apenas forneça o título. Não inclua conteúdo extra, perguntas ou opções.

Idioma: {language_name}
Tom: {tone}
Dados da propriedade: {property_data}

Gere um título de 30-60 caracteres:""",
        "refinement": """Responda exclusivamente em {language_name}. Refine este título de propriedade com base na sugestão fornecida.
Escreva com um tom {tone}. Detalhes do tom: {tone_description}

IMPORTANTE: O título refinado deve ter entre 30-60 caracteres (incluindo espaços). Isso é crítico para otimização SEO.

Idioma: {language_name}
Título atual: {current_content}

Dados da propriedade: {property_data}

Sugerência de melhoria:
- {suggestion}

Forneça apenas o título refinado (30-60 caracteres) sem explicações ou conteúdo extra.""",
    },
}


class TitleAgent(AssistantAgent):
    def __init__(
        self,
        name="title_agent",
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
            system_message="You are a real estate SEO expert. Only output a single plain title string for the property, under 60 characters, with no extra content, questions, or options. Do not include any explanations or formatting beyond the required string.",
        )

    def build_user_prompt(self, property_data, language="en", tone="family-oriented"):
        tone_description = TONE_OPTIONS.get(tone, {}).get("description", "")
        language_name = LANGUAGE_OPTIONS.get(language, {}).get("name", language)
        prompt_template = TITLE_PROMPTS.get(language, TITLE_PROMPTS["en"])["initial"]
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
        prompt_template = TITLE_PROMPTS.get(language, TITLE_PROMPTS["en"])["refinement"]
        return prompt_template.format(
            tone=tone,
            tone_description=tone_description,
            language_name=language_name,
            current_content=current_content,
            property_data=property_data,
            suggestion=suggestion,
        )

    async def generate_initial(self, property_data: Dict[str, Any], language="en", tone="family-oriented") -> str:
        """Generate initial title draft."""
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
        """Refine existing title based on a suggestion."""
        prompt = self.build_refinement_prompt(
            property_data=property_data,
            current_content=current_content,
            suggestion=suggestion,
            language=language,
            tone=tone,
        )
        print(prompt)  # Debugging line, can be removed later
        response = await self.run(task=prompt)
        return response.messages[-1].content.strip()
