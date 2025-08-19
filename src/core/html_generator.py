import logging
from typing import Dict, Any, Optional

from models.base_language_model import BaseLanguageModel
from core.title_generator import TitleGenerator
from core.meta_description_generator import MetaDescriptionGenerator
from core.h1_generator import H1Generator
from core.description_generator import DescriptionGenerator
from core.key_features_generator import KeyFeaturesGenerator
from core.neighborhood_generator import NeighborhoodGenerator
from core.call_to_action_generator import CallToActionGenerator

from agents.title_agent import TitleAgent
from agents.meta_description_agent import MetaDescriptionAgent
from agents.h1_agent import H1Agent
from agents.description_agent import DescriptionAgent
from agents.key_features_agent import KeyFeaturesAgent
from agents.neighborhood_agent import NeighborhoodAgent
from agents.call_to_action_agent import CallToActionAgent


class HTMLGenerator:
    """
    A general HTML generator for real estate listings.

    This class can be initialized with a language model and parameters,
    and provides methods to generate HTML content from structured JSON data.
    """

    def __init__(
        self,
        language_model: Optional[BaseLanguageModel] = None,
        template_style: str = "modern",
        include_seo: bool = True,
        multilingual: bool = False,
        default_language: str = "en",
    ):
        """
        Initialize the HTML generator.

        Args:
            language_model: Optional language model for enhanced content generation
            template_style: Style template to use ("modern", "classic", "minimal")
            include_seo: Whether to include SEO meta tags
            multilingual: Whether to support multiple languages
            default_language: Default language code
        """
        self.language_model = language_model
        self.template_style = template_style
        self.include_seo = include_seo
        self.multilingual = multilingual
        self.default_language = default_language
        self.logger = logging.getLogger(__name__)

    async def generate_html(self, property_data: Dict[str, Any]) -> str:
        """
        Generate all HTML sections for a real estate listing using section generators.
        Returns the concatenated output as plain text, each section with its HTML tag.
        """
        try:
            # Aquí deberías inicializar los agentes reales
            title_agent = TitleAgent()
            meta_agent = MetaDescriptionAgent()
            h1_agent = H1Agent()
            desc_agent = DescriptionAgent()
            key_agent = KeyFeaturesAgent()
            nb_agent = NeighborhoodAgent()
            cta_agent = CallToActionAgent()

            title_gen = TitleGenerator(title_agent)
            meta_gen = MetaDescriptionGenerator(meta_agent)
            h1_gen = H1Generator(h1_agent)
            desc_gen = DescriptionGenerator(desc_agent)
            key_gen = KeyFeaturesGenerator(key_agent)
            nb_gen = NeighborhoodGenerator(nb_agent)
            cta_gen = CallToActionGenerator(cta_agent)

            # Generar cada sección
            title = await title_gen.generate(property_data)
            meta = await meta_gen.generate(property_data)
            h1 = await h1_gen.generate(property_data)
            description = await desc_gen.generate(property_data)
            key_features = await key_gen.generate(property_data)
            neighborhood = await nb_gen.generate(property_data)
            call_to_action = await cta_gen.generate(property_data)

            # Concatenar todas las secciones
            html_output = "\n".join([title, meta, h1, description, key_features, neighborhood, call_to_action])
            return html_output
        except Exception as e:
            self.logger.error(f"Error generating HTML: {str(e)}")
            return f"<p>Error generating content: {str(e)}</p>"
