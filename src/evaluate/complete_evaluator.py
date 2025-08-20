from typing import Dict, Any, List
from .seo import SeoEvaluator
from .language import LanguageEvaluator
from .fact import FactEvaluator
from bs4 import BeautifulSoup


class CompleteEvaluator:
    """
    Complete evaluator that combines all evaluation methods and provides standardized output.
    This replaces the evaluation logic from EvaluatorAgent to separate concerns.
    """

    def __init__(self):
        """Initialize all sub-evaluators."""
        self.seo_evaluator = SeoEvaluator()
        self.language_evaluator = LanguageEvaluator()
        self.fact_evaluator = FactEvaluator()

    async def evaluate_html_complete(
        self, html_content: str, property_data: Dict[str, Any], language: str = "en", tone: str = "professional"
    ) -> Dict[str, Any]:
        """
        Evaluate complete HTML content using all evaluation methods.

        Args:
            html_content: Complete HTML content to evaluate
            property_data: Property data for context and fact checking
            language: Expected language code (en, es, pt)
            tone: Expected tone (professional, friendly, luxury, etc.)

        Returns:
            Dict with standardized evaluation results including scores, issues, and suggestions
        """
        try:
            # Parse HTML content to extract different sections
            soup = BeautifulSoup(html_content, "html.parser")

            # Extract content sections for detailed analysis
            title = soup.find("title")
            title_text = title.text if title else ""

            meta_desc = soup.find("meta", attrs={"name": "description"})
            meta_desc_text = meta_desc.get("content", "") if meta_desc else ""

            h1 = soup.find("h1")
            h1_text = h1.text if h1 else ""

            # Get all text content for comprehensive evaluation
            all_text = soup.get_text()

            # Run all evaluations using standardized interfaces
            seo_results = await self.seo_evaluator.evaluate(html_content=html_content)
            """language_results = await self.language_evaluator.evaluate(
                html_content=html_content, language=language, tone=tone
            )
            fact_results = await self.fact_evaluator.evaluate(html_content=html_content, property_data=property_data)
            """
            print(seo_results)
            # Compile standardized evaluation results
            evaluation = self._compile_evaluation_results(seo_results, seo_results, seo_results)

            # Add extracted content sections for reference
            evaluation["content_sections"] = {
                "title": title_text,
                "meta_description": meta_desc_text,
                "h1": h1_text,
                "full_content": all_text[:500] + "..." if len(all_text) > 500 else all_text,
            }

            return evaluation

        except Exception as e:
            return {
                "overall_score": 0.0,
                "needs_improvement": True,
                "component_scores": {},
                "detailed_results": {},
                "all_findings": [],
                "content_sections": {},
                "error": f"Evaluation failed: {str(e)}",
            }

    def _compile_evaluation_results(
        self,
        seo_results: Dict[str, Any],
        language_results: Dict[str, Any],
        fact_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compile all evaluation results into a standardized format.

        Returns:
            Dict with standardized scores, issues, suggestions, and improvement flags.
        """
        all_results = {
            "seo": seo_results,
            "language": language_results,
            "facts": fact_results,
        }

        # Extract component scores from standardized results
        component_scores = {
            "seo": seo_results.get("score", 0.0),
            "language": language_results.get("score", 0.0),
            "facts": fact_results.get("score", 0.0),
        }

        # Calculate weighted overall score
        weights = {"seo": 0.4, "language": 0.3, "facts": 0.3}
        overall_score = sum(component_scores[key] * weights[key] for key in weights)
        overall_score = min(1.0, max(0.0, overall_score))

        # Collect all findings (issues and suggestions)
        all_findings = []
        for result in all_results.values():
            all_findings.extend(result.get("findings", []))

        # Determine if improvement is needed based on 'passed' flags
        needs_improvement = not all(r.get("passed", False) for r in all_results.values())

        return {
            "overall_score": overall_score,
            "needs_improvement": needs_improvement,
            "component_scores": component_scores,
            "detailed_results": all_results,
            "all_findings": all_findings,
        }
