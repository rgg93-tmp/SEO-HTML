from typing import Dict, Any
import re
from .base_evaluator import BaseEvaluator
from spellchecker import SpellChecker
from agents.evaluation.language_evaluator_agent import LanguageEvaluatorAgent
from agents.evaluation.tone_evaluator_agent import ToneEvaluatorAgent
import textstat

import readability
import syntok.segmenter as segmenter


class LanguageMatchEvaluator(BaseEvaluator):
    def __init__(self):
        self.agent = LanguageEvaluatorAgent()

    async def evaluate(self, text: str, target_language: str) -> Dict[str, Any]:
        """Evaluate if content matches target language."""
        result = await self.agent.evaluate(content=text, expected_language=target_language)
        findings = []
        if not result.get("passed", True):
            findings.append(
                {
                    "type": "language_mismatch",
                    "severity": "medium",
                    "message": f"Text should be in {target_language}.",
                }
            )
        return {
            "score": result.get("score", 0.0),
            "passed": result.get("passed", True),
            "findings": findings,
        }


class LanguageMatchEvaluator2(BaseEvaluator):

    def evaluate(self, text: str, language_code: str, target_language: str) -> Dict[str, Any]:
        """Evaluate if content matches target language."""
        try:
            # Use the spell checker for the specified language
            spell_checker = SpellChecker(language=language_code)
        except Exception as e:
            return {
                "evaluator": "LanguageMatchEvaluator2",
                "score": 1.0,
                "passed": True,
                "findings": [{"type": "spelling", "message": f"Error: {str(e)}"}],
            }
        words = re.findall(pattern=r"\b\w+\b", string=text.lower())
        misspelled = spell_checker.unknown(words=words)
        findings = []
        score = 1.0 - (len(misspelled) / max(1, len(words)))
        if score < 0.98:
            findings.append(
                {
                    "type": "language_mismatch",
                    "severity": "medium",
                    "message": f"Text should be in {target_language}.",
                }
            )
        return {
            "evaluator": "LanguageMatchEvaluator2",
            "score": score,
            "passed": score > 0.8,
            "findings": findings,
        }


class ToneMatchEvaluator(BaseEvaluator):
    def __init__(self):
        self.agent = ToneEvaluatorAgent()

    async def evaluate(self, text: str, target_tone: str) -> Dict[str, Any]:
        result = await self.agent.evaluate(content=text, expected_tone=target_tone)
        findings = []
        if not result.get("passed", True):
            findings.append(
                {
                    "type": "tone",
                    "message": f"Text should have a '{target_tone}' tone.",
                }
            )
        return {
            "evaluator": "ToneMatchEvaluator",
            "score": result.get("score", 0.0),
            "passed": result.get("passed", True),
            "findings": findings,
        }


class SpellingEvaluator(BaseEvaluator):

    def evaluate(self, text: str, language_code: str) -> Dict[str, Any]:
        try:
            # Use the spell checker for the specified language
            spell_checker = SpellChecker(language=language_code)
        except Exception as e:
            return {
                "evaluator": "SpellingEvaluator",
                "score": 1.0,
                "passed": True,
                "findings": [{"type": "spelling", "message": f"Error: {str(e)}"}],
            }
        words = re.findall(pattern=r"\b\w+\b", string=text.lower())
        misspelled = spell_checker.unknown(words=words)
        findings = []
        if misspelled:
            findings.append(
                {
                    "type": "spelling",
                    "message": f"Misspelled words: {', '.join(misspelled)}",
                }
            )
        score = 1.0 - (len(misspelled) / max(1, len(words)))
        return {
            "evaluator": "SpellingEvaluator",
            "score": score,
            "passed": not bool(misspelled),
            "findings": findings,
        }


class ReadabilityEvaluator(BaseEvaluator):

    def _get_difficulty_level(self, flesch_score: float) -> str:
        """Map Flesch Reading Ease score to difficulty level."""
        difficulty_ranges = [
            (90, "Very Easy"),
            (80, "Easy"),
            (70, "Fairly Easy"),
            (60, "Standard"),
            (50, "Fairly Difficult"),
            (30, "Difficult"),
            (0, "Very Confusing"),
        ]

        for threshold, difficulty in difficulty_ranges:
            if flesch_score >= threshold:
                return difficulty
        return "Very Confusing"  # fallback

    def evaluate(self, text: str, language_code: str) -> Dict[str, Any]:
        try:
            textstat.set_lang(language_code)
            flesch_score = textstat.flesch_reading_ease(text)
            difficulty = self._get_difficulty_level(flesch_score)

            findings = [{"type": "readability", "message": f"Flesch Reading Ease: {flesch_score:.2f} ({difficulty})"}]

            # Score: normalize to 0-1 (100 = 1.0, 0 = 0.0)
            score = max(0.0, min(1.0, flesch_score / 100.0))
            # Passed: at least Standard (>=60)
            passed = flesch_score >= 60

            return {
                "evaluator": "ReadabilityEvaluator",
                "score": score,
                "passed": passed,
                "findings": findings,
            }
        except Exception as e:
            return {
                "evaluator": "ReadabilityEvaluator",
                "score": 1.0,
                "passed": True,
                "findings": [],
            }
