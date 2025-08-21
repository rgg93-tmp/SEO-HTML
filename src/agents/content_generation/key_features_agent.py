from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient
from typing import Dict, Any
from config.options import TONE_OPTIONS, LANGUAGE_OPTIONS

# Language-specific prompts
KEY_FEATURES_PROMPTS = {
    "en": {
        "initial": """Respond exclusively in {language_name}. Generate key features for the following property with a {tone} tone.
Tone details: {tone_description}

IMPORTANT: Only use the features listed below. Do not invent or add any features not present in this data.

Available features:
{features_list}

Create exactly 3-5 key features as a simple list. Each feature should be on a new line starting with a hyphen (-).
Focus on the most important and attractive features from the available data.

Example format:
- 167 sqm of living space
- 3 bedrooms and 2 bathrooms
- Private balcony with outdoor space
- Convenient parking included

Language: {language_name}
Tone: {tone}

Only output the list of features based on the available data, no explanations or extra content:""",
        "refinement": """Respond exclusively in {language_name}. Refine this property key features list based on the suggestion provided. Keep 3-5 features, each on a new line starting with a hyphen (-).
Write with a {tone} tone. Tone details: {tone_description}

IMPORTANT: Only use the features listed below. Do not invent or add any features not present in this data.

Available features:
{features_list}

Language: {language_name}
Current features:
{current_content}

Suggestion for improvement:
- {suggestion}

Provide only the refined features list based on available data with no explanations or extra content.""",
    },
    "es": {
        "initial": """Responde exclusivamente en {language_name}. Genera características clave para la siguiente propiedad con un tono {tone}.
Detalles del tono: {tone_description}

IMPORTANTE: Solo usa las características listadas abajo. No inventes o agregues características que no estén presentes en estos datos.

Características disponibles:
{features_list}

Crea exactamente 3-5 características clave como una lista simple. Cada característica debe estar en una nueva línea empezando con un guión (-).
Enfócate en las características más importantes y atractivas de los datos disponibles.

Formato de ejemplo:
- 167 m² de espacio habitable
- 3 dormitorios y 2 baños
- Balcón privado con espacio exterior
- Estacionamiento incluido

Idioma: {language_name}
Tono: {tone}

Solo proporciona la lista de características basada en los datos disponibles, sin explicaciones o contenido extra:""",
        "refinement": """Responde exclusivamente en {language_name}. Refina esta lista de características clave de la propiedad basándote en la sugerencia proporcionada. Mantén 3-5 características, cada una en una nueva línea empezando con un guión (-).
Escribe con un tono {tone}. Detalles del tono: {tone_description}

IMPORTANTE: Solo usa las características listadas abajo. No inventes o agregues características que no estén presentes en estos datos.

Características disponibles:
{features_list}

Idioma: {language_name}
Características actuales:
{current_content}

Sugerencia de mejora:
- {suggestion}

Proporciona solo la lista de características refinada basada en datos disponibles sin explicaciones o contenido extra.""",
    },
    "pt": {
        "initial": """Responda exclusivamente em {language_name}. Gere características principais para a seguinte propriedade com um tom {tone}.
Detalhes do tom: {tone_description}

IMPORTANTE: Use apenas as características listadas abaixo. Não invente ou adicione características que não estejam presentes nestes dados.

Características disponíveis:
{features_list}

Crie exatamente 3-5 características principais como uma lista simples. Cada característica deve estar em uma nova linha começando com um hífen (-).
Foque nas características mais importantes e atrativas dos dados disponíveis.

Formato de exemplo:
- 167 m² de área habitável
- 3 quartos e 2 banheiros
- Varanda privativa com espaço exterior
- Estacionamento incluído

Idioma: {language_name}
Tom: {tone}

Apenas forneça a lista de características baseada nos dados disponíveis, sem explicações ou conteúdo extra:""",
        "refinement": """Responda exclusivamente em {language_name}. Refine esta lista de características principais da propriedade com base na sugestão fornecida. Mantenha 3-5 características, cada uma em uma nova linha começando com um hífen (-).
Escreva com um tom {tone}. Detalhes do tom: {tone_description}

IMPORTANTE: Use apenas as características listadas abaixo. Não invente ou adicione características que não estejam presentes nestes dados.

Características disponíveis:
{features_list}

Idioma: {language_name}
Características atuais:
{current_content}

Sugerência de melhoria:
- {suggestion}

Forneça apenas a lista de características refinada baseada em dados disponíveis sem explicações ou conteúdo extra.""",
    },
}


class KeyFeaturesAgent(AssistantAgent):
    def __init__(
        self,
        name="key_features_agent",
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
            system_message="You are a real estate copywriting expert. Generate 3-5 key property features as a simple list, with each feature on a new line. Start each line with a hyphen (-). Only use features that are explicitly provided in the property data. Do not invent or hallucinate features.",
        )

    def _extract_features_list(self, property_data: Dict[str, Any], language: str = "en") -> str:
        """Extract and format available features from property data."""
        features = property_data.get("features", {})
        features_list = []

        # Language-specific feature descriptions
        feature_translations = {
            "en": {
                "bedrooms": "bedrooms",
                "bathrooms": "bathrooms",
                "area_sqm": "square meters",
                "balcony": "balcony",
                "parking": "parking",
                "elevator": "elevator access",
                "floor": "floor",
                "year_built": "built in",
            },
            "es": {
                "bedrooms": "dormitorios",
                "bathrooms": "baños",
                "area_sqm": "metros cuadrados",
                "balcony": "balcón",
                "parking": "estacionamiento",
                "elevator": "acceso por ascensor",
                "floor": "piso",
                "year_built": "construido en",
            },
            "pt": {
                "bedrooms": "quartos",
                "bathrooms": "banheiros",
                "area_sqm": "metros quadrados",
                "balcony": "varanda",
                "parking": "estacionamento",
                "elevator": "acesso por elevador",
                "floor": "andar",
                "year_built": "construído em",
            },
        }

        translations = feature_translations.get(language, feature_translations["en"])

        for key, value in features.items():
            if key == "bedrooms" and value:
                features_list.append(f"{value} {translations['bedrooms']}")
            elif key == "bathrooms" and value:
                features_list.append(f"{value} {translations['bathrooms']}")
            elif key == "area_sqm" and value:
                features_list.append(f"{value} {translations['area_sqm']}")
            elif key == "balcony" and value:
                features_list.append(translations["balcony"])
            elif key == "parking" and value:
                features_list.append(translations["parking"])
            elif key == "elevator" and value:
                features_list.append(translations["elevator"])
            elif key == "floor" and value:
                features_list.append(f"{translations['floor']} {value}")
            elif key == "year_built" and value:
                features_list.append(f"{translations['year_built']} {value}")

        return "\n".join([f"- {feature}" for feature in features_list])

    def build_user_prompt(self, property_data, language="en", tone="family-oriented"):
        tone_description = TONE_OPTIONS.get(tone, {}).get("description", "")
        language_name = LANGUAGE_OPTIONS.get(language, {}).get("name", language)
        features_list = self._extract_features_list(property_data, language)
        prompt_template = KEY_FEATURES_PROMPTS.get(language, KEY_FEATURES_PROMPTS["en"])["initial"]
        return prompt_template.format(
            tone=tone, tone_description=tone_description, language_name=language_name, features_list=features_list
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
        features_list = self._extract_features_list(property_data, language)
        prompt_template = KEY_FEATURES_PROMPTS.get(language, KEY_FEATURES_PROMPTS["en"])["refinement"]
        return prompt_template.format(
            tone=tone,
            tone_description=tone_description,
            language_name=language_name,
            current_content=current_content,
            features_list=features_list,
            suggestion=suggestion,
        )

    async def generate_initial(self, property_data: Dict[str, Any], language="en", tone="family-oriented") -> str:
        """Generate initial key features draft."""
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
        """Refine existing key features based on a suggestion."""
        prompt = self.build_refinement_prompt(
            property_data=property_data,
            current_content=current_content,
            suggestion=suggestion,
            language=language,
            tone=tone,
        )
        response = await self.run(task=prompt)
        return response.messages[-1].content.strip()
