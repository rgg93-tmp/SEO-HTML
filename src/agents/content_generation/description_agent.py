from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient
from typing import Dict, Any
from config.options import TONE_OPTIONS, LANGUAGE_OPTIONS

# Language-specific prompts
DESCRIPTION_PROMPTS = {
    "en": {
        "initial": """Respond exclusively in {language_name}. Generate a rich, engaging property description paragraph with all key features for the following property with a {tone} tone.
Tone details: {tone_description}

IMPORTANT: The description must be between 500-700 characters (including spaces). Create a single, engaging paragraph that highlights all key features and makes the property attractive to potential buyers/renters.

Only output the description paragraph. Do not include any extra content, questions, or options.

Language: {language_name}
Tone: {tone}
Property data: {property_data}

Generate a rich, engaging paragraph (500-700 characters):""",
        "refinement": """Respond exclusively in {language_name}. Refine this property description based on the suggestion provided.
Write with a {tone} tone. Tone details: {tone_description}

IMPORTANT: The description must be between 500-700 characters (including spaces). Create a single, engaging paragraph that highlights all key features and makes the property attractive to potential buyers/renters.

Language: {language_name}
Current description: {current_content}

Property data: {property_data}

Suggestion for improvement:
- {suggestion}

Provide only the refined description paragraph (500-700 characters) with no explanations or extra content.""",
    },
    "es": {
        "initial": """Responde exclusivamente en {language_name}. Genera un párrafo de descripción rico y atractivo con todas las características clave para la siguiente propiedad con un tono {tone}.
Detalles del tono: {tone_description}

IMPORTANTE: La descripción debe tener entre 500-700 caracteres (incluyendo espacios). Crea un solo párrafo atractivo que destaque todas las características clave y haga la propiedad atractiva para potenciales compradores/inquilinos.

Solo proporciona el párrafo de descripción. No incluyas contenido adicional, preguntas u opciones.

Idioma: {language_name}
Tono: {tone}
Datos de la propiedad: {property_data}

Genera un párrafo rico y atractivo (500-700 caracteres):""",
        "refinement": """Responde exclusivamente en {language_name}. Refina esta descripción de propiedad basándote en la sugerencia proporcionada.
Escribe con un tono {tone}. Detalles del tono: {tone_description}

IMPORTANTE: La descripción debe tener entre 500-700 caracteres (incluyendo espacios). Crea un solo párrafo atractivo que destaque todas las características clave y haga la propiedad atractiva para potenciales compradores/inquilinos.

Idioma: {language_name}
Descripción actual: {current_content}

Datos de la propiedad: {property_data}

Sugerencia de mejora:
- {suggestion}

Proporciona solo el párrafo de descripción refinado (500-700 caracteres) sin explicaciones o contenido adicional.""",
    },
    "pt": {
        "initial": """Responda exclusivamente em {language_name}. Gere um parágrafo de descrição rico e envolvente com todas as características principais para a seguinte propriedade com um tom {tone}.
Detalhes do tom: {tone_description}

IMPORTANTE: A descrição deve ter entre 500-700 caracteres (incluindo espaços). Crie um único parágrafo envolvente que destaque todas as características principais e torne a propriedade atrativa para potenciais compradores/inquilinos.

Apenas forneça o parágrafo de descrição. Não inclua conteúdo extra, perguntas ou opções.

Idioma: {language_name}
Tom: {tone}
Dados da propriedade: {property_data}

Gere um parágrafo rico e envolvente (500-700 caracteres):""",
        "refinement": """Responda exclusivamente em {language_name}. Refine esta descrição de propriedade com base na sugestão fornecida.
Escreva com um tom {tone}. Detalhes do tom: {tone_description}

IMPORTANTE: A descrição deve ter entre 500-700 caracteres (incluindo espaços). Crie um único parágrafo envolvente que destaque todas as características principais e torne a propriedade atrativa para potenciais compradores/inquilinos.

Idioma: {language_name}
Descrição atual: {current_content}

Dados da propriedade: {property_data}

Sugerência de melhoria:
- {suggestion}

Forneça apenas o parágrafo de descrição refinado (500-700 caracteres) sem explicações ou conteúdo extra.""",
    },
}


class DescriptionAgent(AssistantAgent):
    def __init__(
        self,
        name="description_agent",
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
            system_message="You are a real estate copywriting expert. Only output a single plain description string for the property, between 500 and 700 characters, with no extra content, questions, or options. Do not include any explanations or formatting beyond the required string.",
        )

    def build_user_prompt(self, property_data, language="en", tone="family-oriented"):
        tone_description = TONE_OPTIONS.get(tone, {}).get("description", "")
        language_name = LANGUAGE_OPTIONS.get(language, {}).get("name", language)
        prompt_template = DESCRIPTION_PROMPTS.get(language, DESCRIPTION_PROMPTS["en"])["initial"]
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
        prompt_template = DESCRIPTION_PROMPTS.get(language, DESCRIPTION_PROMPTS["en"])["refinement"]
        return prompt_template.format(
            tone=tone,
            tone_description=tone_description,
            language_name=language_name,
            current_content=current_content,
            property_data=property_data,
            suggestion=suggestion,
        )

    async def generate_initial(self, property_data: Dict[str, Any], language="en", tone="family-oriented") -> str:
        """Generate initial description draft."""
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
        """Refine existing description based on a suggestion."""
        prompt = self.build_refinement_prompt(
            property_data=property_data,
            current_content=current_content,
            suggestion=suggestion,
            language=language,
            tone=tone,
        )
        response = await self.run(task=prompt)
        return response.messages[-1].content.strip()
