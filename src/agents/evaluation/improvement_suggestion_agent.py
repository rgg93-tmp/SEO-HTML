from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient
from typing import Dict, Any, List
import json


class ImprovementSuggestionAgent(AssistantAgent):
    """
    Agent that takes evaluation results and provides specific improvement suggestions
    for each content section of the HTML.
    """

    def __init__(
        self,
        name="improvement_suggestion_agent",
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
            system_message="You are a content improvement expert for real estate listings. Given evaluation results, provide specific, actionable improvement suggestions for each content section. Focus on the most impactful changes that will improve SEO, user experience, and conversion rates.",
        )

    async def generate_section_improvements(
        self,
        evaluation_results: Dict[str, Any],
        property_data: Dict[str, Any],
        language: str = "en",
        tone: str = "professional",
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate specific improvement suggestions for each HTML content section.

        Args:
            evaluation_results: Results from CompleteEvaluator
            property_data: Property data for context
            language: Target language
            tone: Target tone

        Returns:
            Dict with section-specific improvements and priorities
        """

        # Build comprehensive improvement prompt
        improvement_prompt = self._build_improvement_prompt(evaluation_results, property_data, language, tone)

        # Get AI-generated suggestions
        response = await self.run(task=improvement_prompt)
        suggestions_text = response.messages[-1].content.strip()

        # Parse and structure the suggestions
        structured_suggestions = self._parse_suggestions_response(suggestions_text, evaluation_results)

        return structured_suggestions

    def _build_improvement_prompt(
        self, evaluation_results: Dict[str, Any], property_data: Dict[str, Any], language: str, tone: str
    ) -> str:
        """Build comprehensive prompt for improvement suggestions."""

        language_names = {"en": "English", "es": "Spanish", "pt": "Portuguese"}
        language_name = language_names.get(language, "English")

        # Extract key information
        overall_score = evaluation_results.get("overall_score", 0)
        component_scores = evaluation_results.get("component_scores", {})
        content_sections = evaluation_results.get("content_sections", {})
        all_issues = evaluation_results.get("all_issues", [])

        # Format component scores
        scores_text = "\n".join(
            [f"  • {key.replace('_', ' ').title()}: {score:.2f}" for key, score in component_scores.items()]
        )

        # Format current content
        current_content = ""
        for section, content in content_sections.items():
            if content and content.strip():
                current_content += f"\n{section.replace('_', ' ').title()}: {content[:200]}...\n"

        # Format issues
        issues_text = "\n".join([f"  - {issue}" for issue in all_issues[:10]])

        return f"""Analyze this real estate content evaluation and provide specific improvement suggestions for each section.

PROPERTY DATA:
{json.dumps(property_data, indent=2)}

TARGET LANGUAGE: {language_name}
TARGET TONE: {tone}

EVALUATION RESULTS:
Overall Score: {overall_score:.2f}

Component Scores:
{scores_text}

CURRENT CONTENT:
{current_content}

IDENTIFIED ISSUES:
{issues_text}

TASK: Provide specific improvement suggestions for each content section. For each section that needs improvement, suggest:

1. TITLE improvements (if score < 0.8)
2. META DESCRIPTION improvements (if score < 0.8)  
3. H1 improvements (if score < 0.8)
4. DESCRIPTION improvements (if score < 0.8)
5. KEY FEATURES improvements (if score < 0.8)
6. NEIGHBORHOOD improvements (if score < 0.8)
7. CALL TO ACTION improvements (if score < 0.8)

For each section, provide:
- Priority: HIGH/MEDIUM/LOW
- Specific suggestion: What exactly to change
- Reason: Why this improvement is needed
- Expected impact: SEO/Engagement/Conversion/Accuracy

Format your response as:

SECTION: [section_name]
Priority: [HIGH/MEDIUM/LOW]
Suggestion: [specific improvement]
Reason: [why needed]
Impact: [expected benefit]

---

Only suggest improvements for sections that actually need them based on the evaluation results.
Focus on the most impactful changes first.
Keep suggestions specific and actionable.
"""

    def _parse_suggestions_response(
        self, suggestions_text: str, evaluation_results: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Parse AI response into structured improvement suggestions."""

        suggestions = {}
        current_section = None
        current_suggestion = {}

        lines = suggestions_text.split("\n")

        for line in lines:
            line = line.strip()

            if line.startswith("SECTION:"):
                # Save previous section if exists
                if current_section and current_suggestion:
                    suggestions[current_section] = current_suggestion

                # Start new section
                current_section = line.replace("SECTION:", "").strip().lower()
                current_suggestion = {
                    "priority": "MEDIUM",
                    "suggestion": "",
                    "reason": "",
                    "impact": "",
                    "component_scores": self._get_section_scores(current_section, evaluation_results),
                }

            elif line.startswith("Priority:") and current_section:
                current_suggestion["priority"] = line.replace("Priority:", "").strip()

            elif line.startswith("Suggestion:") and current_section:
                current_suggestion["suggestion"] = line.replace("Suggestion:", "").strip()

            elif line.startswith("Reason:") and current_section:
                current_suggestion["reason"] = line.replace("Reason:", "").strip()

            elif line.startswith("Impact:") and current_section:
                current_suggestion["impact"] = line.replace("Impact:", "").strip()

        # Save last section
        if current_section and current_suggestion:
            suggestions[current_section] = current_suggestion

        # Add fallback suggestions for critical issues
        suggestions = self._add_fallback_suggestions(suggestions, evaluation_results)

        return suggestions

    def _get_section_scores(self, section: str, evaluation_results: Dict[str, Any]) -> Dict[str, float]:
        """Get relevant scores for a specific section."""
        component_scores = evaluation_results.get("component_scores", {})

        # Map sections to relevant score components
        section_score_mapping = {
            "title": ["seo", "language_accuracy"],
            "meta_description": ["seo", "language_accuracy"],
            "h1": ["seo", "readability"],
            "description": ["readability", "fact_accuracy", "tone"],
            "key_features": ["fact_accuracy", "readability"],
            "neighborhood": ["fact_accuracy", "tone"],
            "call_to_action": ["tone", "readability"],
        }

        relevant_scores = {}
        for score_type in section_score_mapping.get(section, []):
            if score_type in component_scores:
                relevant_scores[score_type] = component_scores[score_type]

        return relevant_scores

    def _add_fallback_suggestions(
        self, suggestions: Dict[str, Dict[str, Any]], evaluation_results: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Add fallback suggestions for critical issues not covered by AI response."""

        component_scores = evaluation_results.get("component_scores", {})

        # Critical thresholds
        critical_thresholds = {"seo": 0.6, "fact_accuracy": 0.7, "spelling": 0.8, "language_accuracy": 0.6}

        # Add critical suggestions if missing
        for component, threshold in critical_thresholds.items():
            score = component_scores.get(component, 1.0)
            if score < threshold:
                self._add_critical_suggestion(suggestions, component, score)

        return suggestions

    def _add_critical_suggestion(self, suggestions: Dict[str, Dict[str, Any]], component: str, score: float):
        """Add critical suggestion for low-scoring component."""

        critical_suggestions = {
            "seo": {
                "section": "meta_description",
                "priority": "HIGH",
                "suggestion": "Optimize meta description for SEO: ensure it's under 155 characters, includes target keywords, and accurately describes the property",
                "reason": f"SEO score is critically low ({score:.2f})",
                "impact": "SEO/Visibility",
            },
            "fact_accuracy": {
                "section": "description",
                "priority": "HIGH",
                "suggestion": "Review all property details for accuracy: verify room counts, measurements, features, and location information match the property data",
                "reason": f"Fact accuracy score is low ({score:.2f})",
                "impact": "Accuracy/Trust",
            },
            "spelling": {
                "section": "description",
                "priority": "HIGH",
                "suggestion": "Perform thorough spell-check and grammar review of all content",
                "reason": f"Spelling score is low ({score:.2f})",
                "impact": "Professional quality",
            },
            "language_accuracy": {
                "section": "title",
                "priority": "HIGH",
                "suggestion": "Ensure all content is written in the correct target language with proper grammar and vocabulary",
                "reason": f"Language accuracy score is low ({score:.2f})",
                "impact": "Communication/Clarity",
            },
        }

        if component in critical_suggestions:
            critical_info = critical_suggestions[component]
            section = critical_info["section"]

            # Only add if section doesn't already have suggestions
            if section not in suggestions:
                suggestions[section] = {
                    "priority": critical_info["priority"],
                    "suggestion": critical_info["suggestion"],
                    "reason": critical_info["reason"],
                    "impact": critical_info["impact"],
                    "component_scores": {component: score},
                    "is_critical": True,
                }
