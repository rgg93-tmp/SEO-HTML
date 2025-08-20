from typing import Dict, List, Any
import re
from .base_evaluator import BaseEvaluator
from spellchecker import SpellChecker
from agents.evaluation.language_evaluator_agent import LanguageEvaluatorAgent
from agents.evaluation.tone_evaluator_agent import ToneEvaluatorAgent


class LanguageEvaluator(BaseEvaluator):
    """Enhanced language evaluator using both spell checking and LLM-based evaluation."""

    def __init__(self):
        """Initialize spell checkers and LLM agents."""
        # Initialize spell checkers for supported languages
        self.spell_checkers = {
            "en": SpellChecker(language="en"),
            "es": SpellChecker(language="es"),
            "pt": SpellChecker(language="pt"),
        }

        # Initialize LLM-based evaluators
        self.language_agent = LanguageEvaluatorAgent()
        self.tone_agent = ToneEvaluatorAgent()

    async def evaluate(
        self, html_content: str, language: str = "en", tone: str = "professional", **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate language quality of the HTML content using both traditional and LLM-based methods.

        Args:
            html_content: HTML content to evaluate
            language: Expected language code (en, es, pt)
            tone: Expected tone (professional, friendly, luxury, etc.)
            **kwargs: Additional parameters

        Returns:
            Standardized evaluation results dictionary
        """
        # Extract text content from HTML
        import re

        text = re.sub(r"<[^>]+>", "", html_content)
        text = " ".join(text.split())  # Clean whitespace

        # Perform individual evaluations
        spelling_results = self._evaluate_spelling(text, language)
        language_results = await self.language_agent.evaluate(text, language)
        tone_results = await self.tone_agent.evaluate(text, tone)

        # Combine scores (weighted average)
        weights = {"spelling": 0.3, "language": 0.4, "tone": 0.3}
        overall_score = (
            spelling_results["score"] * weights["spelling"]
            + language_results["score"] * weights["language"]
            + tone_results["score"] * weights["tone"]
        )

        # Collect all findings
        all_findings = []
        all_findings.extend(spelling_results.get("findings", []))
        all_findings.extend(language_results.get("findings", []))
        all_findings.extend(tone_results.get("findings", []))

        # Determine overall pass status
        passed = all(r["passed"] for r in [spelling_results, language_results, tone_results])

        return {
            "evaluator": "LanguageEvaluator",
            "score": overall_score,
            "passed": passed,
            "summary": f"Language evaluation completed with score {overall_score:.2f}",
            "findings": all_findings,
            "details": {
                "spelling": spelling_results,
                "language_accuracy": language_results,
                "tone": tone_results,
            },
        }

    def _evaluate_spelling(self, text: str, language: str) -> Dict[str, Any]:
        """Evaluate spelling and return standardized dictionary."""
        spell_checker = self.spell_checkers.get(language)
        if not spell_checker:
            return {
                "evaluator": "SpellingCheck",
                "score": 1.0,
                "passed": True,
                "summary": f"Spelling check skipped for unsupported language: {language}",
                "findings": [],
            }

        words = re.findall(r"\b\w+\b", text.lower())
        if not words:
            return {
                "evaluator": "SpellingCheck",
                "score": 1.0,
                "passed": True,
                "summary": "No words found to check for spelling.",
                "findings": [],
            }

        misspelled = list(spell_checker.unknown(words))
        error_rate = len(misspelled) / len(words)
        score = max(0.0, 1.0 - error_rate * 5)  # Penalize more for errors

        findings = []
        if misspelled:
            findings.append(
                {
                    "component": "Spelling",
                    "message": f"Found {len(misspelled)} possible spelling errors: {', '.join(misspelled[:5])}",
                    "severity": "error" if score < 0.8 else "warning",
                }
            )

        return {
            "evaluator": "SpellingCheck",
            "score": score,
            "passed": score >= 0.8,
            "summary": f"Spelling check found {len(misspelled)} errors.",
            "findings": findings,
        }

    def _evaluate_readability(self, text: str) -> float:
        """Readability score based on sentence and word analysis."""
        # This is a simplified readability metric.
        # For more accuracy, consider libraries like 'textstat'.
        sentences = [s for s in re.split(r"[.!?]+", text) if s]
        words = re.findall(r"\b\w+\b", text)

        if not sentences or not words:
            return 1.0  # Perfect score for empty or simple text

        avg_sentence_length = len(words) / len(sentences)

        # Simple score: shorter sentences are better. Target 15 words.
        readability_score = max(0.0, 1.0 - (abs(avg_sentence_length - 15) / 15))

        return readability_score

    async def _evaluate_language_accuracy(self, text: str, language: str) -> float:
        """Evaluate language accuracy using LLM agent."""
        result = await self.language_agent.evaluate(text, language)
        return result.get("score", 0.5)

    async def _evaluate_tone_match(self, text: str, tone: str) -> float:
        """Evaluate tone match using LLM agent."""
        result = await self.tone_agent.evaluate(text, tone)
        return result.get("score", 0.5)

    def _evaluate_readability(self, text: str) -> float:
        """Readability score based on sentence and word analysis."""
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 0.0

        words = text.split()
        if not words:
            return 0.0

        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)

        # Score based on optimal ranges
        # Ideal: 15-20 words per sentence, 4-6 chars per word
        sentence_score = (
            1.0 if 15 <= avg_sentence_length <= 20 else max(0.0, 1.0 - abs(avg_sentence_length - 17.5) * 0.02)
        )
        word_score = 1.0 if 4 <= avg_word_length <= 6 else max(0.0, 1.0 - abs(avg_word_length - 5) * 0.1)

        return (sentence_score + word_score) / 2

    def _evaluate_spelling(self, text: str, language: str) -> float:
        """Evaluate spelling using pyspellchecker framework."""
        if language not in self.spell_checkers:
            return 0.5  # Neutral score for unsupported languages

        spell_checker = self.spell_checkers[language]

        # Clean text and get words
        words = re.findall(r"\b[a-zA-ZáéíóúàèìòùâêîôûãõçÁÉÍÓÚÀÈÌÒÙÂÊÎÔÛÃÕÇ]+\b", text.lower())

        if not words:
            return 1.0

        # Find misspelled words
        misspelled = spell_checker.unknown(words)

        # Calculate score
        error_rate = len(misspelled) / len(words)
        return max(0.0, 1.0 - error_rate * 2)  # Penalize more heavily

    async def _evaluate_language_accuracy(self, text: str, expected_language: str) -> float:
        """Evaluate if content matches expected language using LLM agent."""
        try:
            result = await self.language_agent.evaluate(text, expected_language)
            return result.get("score", 0.5)
        except Exception as e:
            print(f"Error in language accuracy evaluation: {e}")
            return 0.5  # Neutral score on error

    async def _evaluate_tone_match(self, text: str, expected_tone: str) -> float:
        """Evaluate if content matches expected tone using LLM agent."""
        try:
            result = await self.tone_agent.evaluate(text, expected_tone)
            return result.get("score", 0.5)
        except Exception as e:
            print(f"Error in tone evaluation: {e}")
            return 0.5  # Neutral score on error

    def _add_language_suggestions(self, result: Dict[str, Any], text: str, expected_language: str, expected_tone: str):
        """Legacy method - no longer used in standardized evaluation."""
        pass
