from typing import Dict, Any, List
import re
from .seo import SeoEvaluator
from .language import (
    LanguageMatchEvaluator2,
    ToneMatchEvaluator,
    SpellingEvaluator,
    ReadabilityEvaluator,
)
from .fact import FactEvaluator


class CompleteEvaluator:
    """
    Complete evaluator that combines all evaluation methods and provides standardized output.
    This replaces the evaluation logic from EvaluatorAgent to separate concerns.
    """

    def __init__(self):
        """Initialize all sub-evaluators."""
        self.seo_evaluator = SeoEvaluator()
        # self.language_match = LanguageMatchEvaluator()
        self.language_match = LanguageMatchEvaluator2()
        self.tone_match = ToneMatchEvaluator()
        self.spelling = SpellingEvaluator()
        self.readability = ReadabilityEvaluator()
        self.fact_evaluator = FactEvaluator()

    async def evaluate_html_complete(
        self,
        html_content: str,
        property_data: Dict[str, Any],
        language: str = "en",
        language_name: str = "English",
        tone: str = "professional",
    ) -> Dict[str, Any]:
        """
        Simplified: Only compile findings from sub-evaluators if relevant.
        """

        seo_results = await self.seo_evaluator.evaluate(html_content=html_content)
        # Extraer texto plano
        text = re.sub(pattern=r"<[^>]+>", repl="", string=html_content)
        text = " ".join(text.split())

        # language_results = await self.language_match.evaluate(text=text, target_language=language_name)
        language_results = self.language_match.evaluate(
            text=text, language_code=language, target_language=language_name
        )
        tone_results = await self.tone_match.evaluate(text=text, target_tone=tone)
        # spelling_results = self.spelling.evaluate(text=text, language_code=language)
        readability_results = self.readability.evaluate(text=text, language_code=language)
        # fact_results = await self.fact_evaluator.evaluate(html_content=html_content, property_data=property_data)

        findings = (
            language_results.get("findings", [])
            + tone_results.get("findings", [])
            # + spelling_results.get("findings", [])
            + readability_results.get("findings", [])
            # + fact_results.get("findings", []),
            + seo_results.get("findings", [])
        )
        needs_improvement = not (
            language_results.get("passed", True)
            and tone_results.get("passed", True)
            # and spelling_results.get("passed", True)
            and readability_results.get("passed", True)
            # and fact_results.get("passed", True)
            and seo_results.get("passed", True)
        )
        evaluation = {
            "seo": seo_results,
            "language_match": language_results,
            "tone_match": tone_results,
            # "spelling": spelling_results,
            "readability": readability_results,
            # "facts": fact_results,
            "all_findings": findings,
            "needs_improvement": needs_improvement,
        }
        return evaluation
