from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient
from typing import Dict, Any
from config.options import TONE_OPTIONS, LANGUAGE_OPTIONS

# Language-specific prompts
# Language-specific prompts
NEIGHBORHOOD_PROMPTS = {
    "en": {
        "initial": """Respond exclusively in {language_name}. Generate a neighborhood description for the following property with a {tone} tone.
Tone details: {tone_description}
Only output the neighborhood description text. Do not include any extra content, questions, or options.

Language: {language_name}
Tone: {tone}
Neighborhood location: {neighborhood_info}""",
        "refinement": """Respond exclusively in {language_name}. Refine this neighborhood description based on the suggestion provided. Keep it informative and appealing.
Write with a {tone} tone. Tone details: {tone_description}

Language: {language_name}
Current neighborhood description: {current_content}

Neighborhood location: {neighborhood_info}

Suggestion for improvement:
- {suggestion}

Provide only the refined neighborhood description with no explanations or extra content.""",
    },
    "es": {
        "initial": """Responde exclusivamente en {language_name}. Genera una descripción del vecindario para la siguiente propiedad con un tono {tone}.
Detalles del tono: {tone_description}
Solo proporciona el texto de descripción del vecindario. No incluyas contenido adicional, preguntas u opciones.

Idioma: {language_name}
Tono: {tone}
Ubicación del vecindario: {neighborhood_info}""",
        "refinement": """Responde exclusivamente en {language_name}. Refina esta descripción del vecindario basándote en la sugerencia proporcionada. Manténla informativa y atractiva.
Escribe con un tono {tone}. Detalles del tono: {tone_description}

Idioma: {language_name}
Descripción actual del vecindario: {current_content}

Ubicación del vecindario: {neighborhood_info}

Sugerencia de mejora:
- {suggestion}

Proporciona solo la descripción del vecindario refinada sin explicaciones o contenido adicional.""",
    },
    "pt": {
        "initial": """Responda exclusivamente em {language_name}. Gere uma descrição do bairro para a seguinte propriedade com um tom {tone}.
Detalhes do tom: {tone_description}
Apenas forneça o texto da descrição do bairro. Não inclua conteúdo extra, perguntas ou opções.

Idioma: {language_name}
Tom: {tone}
Localização do bairro: {neighborhood_info}""",
        "refinement": """Responda exclusivamente em {language_name}. Refine esta descrição do bairro com base na sugerência fornecida. Mantenha-a informativa e atraente.
Escreva com um tom {tone}. Detalhes do tom: {tone_description}

Idioma: {language_name}
Descrição atual do bairro: {current_content}

Localização do bairro: {neighborhood_info}

Sugerência de melhoria:
- {suggestion}

Forneça apenas a descrição do bairro refinada sem explicações ou conteúdo extra.""",
    },
}


class NeighborhoodAgent(AssistantAgent):
    def __init__(
        self,
        name="neighborhood_agent",
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
            system_message="You are a real estate copywriting expert. Only output a single plain paragraph string about the neighborhood for the property, with no extra content, questions, or options. Do not include any explanations or formatting beyond the required string.",
        )

    def build_user_prompt(self, property_data, language="en", tone="family-oriented"):
        tone_description = TONE_OPTIONS.get(tone, {}).get("description", "")
        language_name = LANGUAGE_OPTIONS[language]["name"]
        # Extract neighborhood-specific information
        neighborhood_info = self._extract_neighborhood_info(property_data)
        prompt_template = NEIGHBORHOOD_PROMPTS.get(language, NEIGHBORHOOD_PROMPTS["en"])["initial"]
        return prompt_template.format(
            language_name=language_name,
            tone=tone,
            tone_description=tone_description,
            neighborhood_info=neighborhood_info,
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
        # Extract neighborhood-specific information
        neighborhood_info = self._extract_neighborhood_info(property_data)
        prompt_template = NEIGHBORHOOD_PROMPTS.get(language, NEIGHBORHOOD_PROMPTS["en"])["refinement"]
        return prompt_template.format(
            language_name=language_name,
            tone=tone,
            tone_description=tone_description,
            current_content=current_content,
            neighborhood_info=neighborhood_info,
            suggestion=suggestion,
        )

    def _extract_neighborhood_info(self, property_data: Dict[str, Any]) -> str:
        """Extract only neighborhood-relevant information from property data."""
        relevant_fields = []

        # Extract location information from the nested location object
        location = property_data.get("location", {})

        if "neighborhood" in location:
            relevant_fields.append(f"Neighborhood: {location['neighborhood']}")
        if "city" in location:
            relevant_fields.append(f"City: {location['city']}")

        # If no specific neighborhood info found, return basic location
        if not relevant_fields:
            return "Property location information not specified"

        return ", ".join(relevant_fields)

    async def generate_initial(self, property_data: Dict[str, Any], language="en", tone="family-oriented") -> str:
        """Generate initial neighborhood description draft."""
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
        """Refine existing neighborhood description based on a suggestion."""
        prompt = self.build_refinement_prompt(
            property_data=property_data,
            current_content=current_content,
            suggestion=suggestion,
            language=language,
            tone=tone,
        )
        response = await self.run(task=prompt)
        return response.messages[-1].content.strip()
