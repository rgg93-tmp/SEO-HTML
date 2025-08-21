from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient
from typing import Dict, Any
from config.options import TONE_OPTIONS, LANGUAGE_OPTIONS

# Language-specific prompts
# Language-specific prompts
H1_PROMPTS = {
    "en": {
        "initial": """Respond exclusively in {language_name}. Generate an H1 tag for the following property with a {tone} tone.
Tone details: {tone_description}
Only output the H1 string content (without HTML tags). Do not include any extra content, questions, or options.

Language: {language_name}
Tone: {tone}
Property data: {property_data}""",
        "refinement": """Respond exclusively in {language_name}. Refine this property H1 heading based on the suggestion provided. Keep it under 70 characters.
Write with a {tone} tone. Tone details: {tone_description}

Language: {language_name}
Current H1: {current_content}

Property data: {property_data}

Suggestion for improvement:
- {suggestion}

Provide only the refined H1 with no explanations or extra content.""",
    },
    "es": {
        "initial": """Responde exclusivamente en {language_name}. Genera un encabezado H1 para la siguiente propiedad con un tono {tone}.
Detalles del tono: {tone_description}
Solo proporciona el contenido del H1 (sin etiquetas HTML). No incluyas contenido adicional, preguntas u opciones.

Idioma: {language_name}
Tono: {tone}
Datos de la propiedad: {property_data}""",
        "refinement": """Responde exclusivamente en {language_name}. Refina este encabezado H1 de propiedad basándote en la sugerencia proporcionada. Manténlo bajo 70 caracteres.
Escribe con un tono {tone}. Detalles del tono: {tone_description}

Idioma: {language_name}
H1 actual: {current_content}

Datos de la propiedad: {property_data}

Sugerencia de mejora:
- {suggestion}

Proporciona solo el H1 refinado sin explicaciones o contenido adicional.""",
    },
    "pt": {
        "initial": """Responda exclusivamente em {language_name}. Gere um cabeçalho H1 para a seguinte propriedade com um tom {tone}.
Detalhes do tom: {tone_description}
Apenas forneça o conteúdo do H1 (sem tags HTML). Não inclua conteúdo extra, perguntas ou opções.

Idioma: {language_name}
Tom: {tone}
Dados da propriedade: {property_data}""",
        "refinement": """Responda exclusivamente em {language_name}. Refine este cabeçalho H1 de propriedade com base na sugestão fornecida. Mantenha-o com menos de 70 caracteres.
Escreva com um tom {tone}. Detalhes do tom: {tone_description}

Idioma: {language_name}
H1 atual: {current_content}

Dados da propriedade: {property_data}

Sugestão de melhoria:
- {suggestion}

Forneça apenas o H1 refinado sem explicações ou conteúdo extra.""",
    },
}


class H1Agent(AssistantAgent):
    def __init__(
        self,
        name="h1_agent",
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
            system_message="You are a real estate SEO expert. Only output a single plain headline string for the property, with no extra content, questions, or options. Do not include any explanations or formatting beyond the required string.",
        )

    def build_user_prompt(self, property_data, language="en", tone="family-oriented"):
        tone_description = TONE_OPTIONS.get(tone, {}).get("description", "")
        language_name = LANGUAGE_OPTIONS.get(language, {}).get("name", language)
        prompt_template = H1_PROMPTS.get(language, H1_PROMPTS["en"])["initial"]
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
        prompt_template = H1_PROMPTS.get(language, H1_PROMPTS["en"])["refinement"]
        return prompt_template.format(
            tone=tone,
            tone_description=tone_description,
            language_name=language_name,
            current_content=current_content,
            property_data=property_data,
            suggestion=suggestion,
        )

    async def generate_initial(self, property_data: Dict[str, Any], language="en", tone="family-oriented") -> str:
        """Generate initial H1 draft."""
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
        """Refine existing H1 based on a suggestion."""
        prompt = self.build_refinement_prompt(
            property_data=property_data,
            current_content=current_content,
            suggestion=suggestion,
            language=language,
            tone=tone,
        )
        response = await self.run(task=prompt)
        return response.messages[-1].content.strip()
