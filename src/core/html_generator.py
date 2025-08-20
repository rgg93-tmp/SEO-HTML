import logging
from typing import Dict, Any, Optional
import asyncio

# Import agents directly instead of generators
from agents.content_generation.title_agent import TitleAgent
from agents.content_generation.meta_description_agent import MetaDescriptionAgent
from agents.content_generation.h1_agent import H1Agent
from agents.content_generation.description_agent import DescriptionAgent
from agents.content_generation.key_features_agent import KeyFeaturesAgent
from agents.content_generation.neighborhood_agent import NeighborhoodAgent
from agents.content_generation.call_to_action_agent import CallToActionAgent

# Import evaluator and improvement agents
from evaluate.complete_evaluator import CompleteEvaluator
from agents.evaluation.improvement_suggestion_agent import ImprovementSuggestionAgent


class HTMLGenerator:
    """
    A general HTML generator for real estate listings with holistic iterative refinement.

    This class can be initialized with a language model and parameters,
    and provides methods to generate HTML content from structured JSON data
    using an iterative process of generation, evaluation, and refinement
    applied to the complete HTML document.
    """

    def __init__(
        self,
        max_iterations: int = 1,
    ):
        """
        Initialize the HTML generator.

        Args:
            max_iterations: Maximum number of holistic refinement iterations
        """
        self.max_iterations = max_iterations
        self.logger = logging.getLogger(__name__)

        # Initialize agents once
        self.agents = {
            "title": TitleAgent(),
            "meta": MetaDescriptionAgent(),
            "h1": H1Agent(),
            "description": DescriptionAgent(),
            "key_features": KeyFeaturesAgent(),
            "neighborhood": NeighborhoodAgent(),
            "call_to_action": CallToActionAgent(),
        }
        self.complete_evaluator = CompleteEvaluator()
        self.improvement_agent = ImprovementSuggestionAgent()

    async def generate_html(
        self, property_data: Dict[str, Any], language: str = "en", tone: str = "professional"
    ) -> str:
        # Convertir código de idioma a nombre completo solo para los agentes
        language_names = {"en": "English", "es": "Spanish", "pt": "Portuguese"}
        agent_language = language_names.get(language, language)
        """
        Generate complete HTML for a real estate listing using holistic iterative refinement.
        """
        self.logger.info(f"Generating initial content drafts in {agent_language} with {tone} tone...")
        tasks = {
            section: agent.generate_initial(property_data, agent_language, tone)
            for section, agent in self.agents.items()
        }
        results = await asyncio.gather(*tasks.values())
        self.sections = dict(zip(tasks.keys(), results))
        print(self.sections)
        # Holistic iterative refinement process
        await self._refine_html_holistically(
            property_data,
            self.complete_evaluator,
            self.improvement_agent,
            self.agents,
            agent_language,
            tone,
        )
        return self._assemble_html_document(language)

    async def _refine_html_holistically(
        self,
        property_data: Dict[str, Any],
        complete_evaluator: CompleteEvaluator,
        improvement_agent: ImprovementSuggestionAgent,
        agents: Dict[str, Any],
        language: str = "English",
        tone: str = "professional",
    ) -> None:
        """
        Refine complete HTML through holistic evaluation and targeted improvements.
        """
        for iteration in range(self.max_iterations):
            self.logger.info(f"--- Holistic Refinement Iteration {iteration + 1} ---")
            # Ensamblar el HTML actual
            current_html = self._assemble_html_document(language)
            print(current_html)
            # Evaluar el HTML
            evaluation_results = await complete_evaluator.evaluate_html_complete(
                current_html, property_data, language, tone
            )
            self._display_evaluation_summary(evaluation_results)
            if not evaluation_results.get("needs_improvement", False):
                self.logger.info("Holistic refinement complete: Content quality is excellent.")
                break
            section_improvements = await improvement_agent.generate_section_improvements(
                evaluation_results, property_data, language, tone
            )
            if not section_improvements:
                self.logger.info("No actionable improvement suggestions were generated. Finalizing content.")
                break
            # Refinar las secciones
        self.sections = await self._apply_section_refinements(
            self.sections, section_improvements, property_data, agents, language, tone
        )
        # Evaluación final
        final_html = self._assemble_html_document(language)
        print(final_html)
        evaluation_results = await complete_evaluator.evaluate_html_complete(final_html, property_data, language, tone)
        self._display_evaluation_summary(evaluation_results)

    def _display_evaluation_summary(self, evaluation_results: Dict[str, Any]):
        """
        Display a summary of the evaluation scores and findings.
        """
        self.logger.debug("\n--- Evaluation Summary ---")
        self.logger.debug(f"Overall Score: {evaluation_results.get('overall_score', 0):.2f}")
        self.logger.debug(f"Needs Improvement: {evaluation_results.get('needs_improvement', False)}")

        for evaluator, results in evaluation_results.get("individual_evaluations", {}).items():
            self.logger.debug(
                f"\n- {evaluator} Score: {results.get('score', 0):.2f} (Passed: {results.get('passed', False)})"
            )
            if findings := results.get("findings"):
                self.logger.debug("  Suggestions:")
                for finding in findings:
                    self.logger.debug(f"    - [{finding.get('severity', 'info').upper()}] {finding.get('message', '')}")
        self.logger.debug("------------------------\n")

    async def _apply_section_refinements(
        self,
        sections: Dict[str, str],
        section_improvements: Dict[str, Dict[str, Any]],
        property_data: Dict[str, Any],
        agents: Dict[str, Any],
        language: str,
        tone: str,
    ) -> Dict[str, str]:
        """
        Apply specific refinements to each section based on improvement suggestions.

        Args:
            sections: Current HTML sections
            section_improvements: Section-specific improvement suggestions
            property_data: Property data for context
            agents: Content generation agents
            language: Target language
            tone: Target tone

        Returns:
            Dict with refined sections
        """
        refined_sections = sections.copy()

        for section_name, improvement_info in section_improvements.items():
            if section_name in agents and improvement_info.get("suggestion"):
                # Extract current content
                current_content = sections.get(section_name, "")

                # Get improvement suggestion
                suggestion = improvement_info["suggestion"]
                priority = improvement_info.get("priority", "MEDIUM")

                self.logger.info(f"Refining {section_name} (Priority: {priority}): {suggestion[:100]}...")

                # Apply refinement using the appropriate agent
                refined_content = await agents[section_name].refine(
                    property_data, current_content, suggestion, language, tone
                )

                # Asignar el contenido refinado directamente; el wrapping se hace en _assemble_html_document
                refined_sections[section_name] = refined_content
                self.logger.info(f"Successfully refined {section_name}")

        return refined_sections

    def _assemble_html_document(self, language: str = "en") -> str:
        """Assemble sections into a complete HTML document following the strict required format."""
        import re

        def wrap(section_name, content):
            if section_name == "title":
                return f"<title>{content}</title>"
            elif section_name == "meta":
                return f'<meta name="description" content="{content}">'
            elif section_name == "h1":
                return f"<h1>{content}</h1>"
            elif section_name == "description":
                return f'<section id="description"><p>{content}</p></section>'
            elif section_name == "key_features":
                # Parse plain text list and convert to <li> elements
                lines = [line.strip() for line in content.strip().split("\n") if line.strip()]
                list_items = []
                for line in lines:
                    clean_line = re.sub(r"^[•\-*\.\s]+", "", line).strip()
                    if clean_line:
                        list_items.append(f"<li>{clean_line}</li>")
                list_content = "\n".join(list_items) if list_items else "<li>No features listed</li>"
                return f'<ul id="key-features">{list_content}</ul>'
            elif section_name == "neighborhood":
                return f'<section id="neighborhood"><p>{content}</p></section>'
            elif section_name == "call_to_action":
                return f'<p class="call-to-action">{content}</p>'
            else:
                return content

        # Defaults for each section
        title = wrap("title", self.sections.get("title", "Property Listing"))
        meta = wrap("meta", self.sections.get("meta", ""))
        h1 = wrap("h1", self.sections.get("h1", "Property Listing"))
        description = wrap("description", self.sections.get("description", "Property description"))
        key_features = wrap("key_features", self.sections.get("key_features", "No features listed"))
        neighborhood = wrap("neighborhood", self.sections.get("neighborhood", "Neighborhood information"))
        call_to_action = wrap("call_to_action", self.sections.get("call_to_action", "Contact us for more information"))

        html_template = f"""<!DOCTYPE html>
<html lang="{language}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    {title}
    {meta}
    {h1}
    {description}
    {key_features}
    {neighborhood}
    {call_to_action}
</body>
</html>"""
        return html_template
