from typing import Dict, Any
import re
from .base_evaluator import BaseEvaluator
from spellchecker import SpellChecker
from agents.evaluation.language_evaluator_agent import LanguageEvaluatorAgent
from agents.evaluation.tone_evaluator_agent import ToneEvaluatorAgent

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
                "score": 1.0,
                "passed": True,
                "findings": [{"type": "spelling", "message": f"Error: {str(e)}"}],
            }
        words = re.findall(pattern=r"\b\w+\b", string=text.lower())
        misspelled = spell_checker.unknown(words=words)
        findings = []
        score = 1.0 - (len(misspelled) / max(1, len(words)))
        if score < 0.8:
            findings.append(
                {
                    "type": "language_mismatch",
                    "severity": "medium",
                    "message": f"Text should be in {target_language}.",
                }
            )
        return {
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
            "score": score,
            "passed": not bool(misspelled),
            "findings": findings,
        }


class ReadabilityEvaluator(BaseEvaluator):

    def evaluate(self, text: str, language_code: str) -> Dict[str, Any]:
        # Tokenize text as in example
        tokenized = "\n\n".join(
            "\n".join(" ".join(token.value for token in sentence) for sentence in paragraph)
            for paragraph in segmenter.analyze(text)
        )
        try:
            results = readability.getmeasures(text=tokenized, lang=language_code)
            flesch = results["readability grades"]["FleschReadingEase"]
            findings = []
            if flesch < 60:
                findings.append(
                    {
                        "type": "readability",
                        "message": f"Low readability score: {flesch:.2f}",
                    }
                )
            return {
                "score": min(1.0, max(0.0, flesch / 100)),
                "passed": flesch >= 60,
                "findings": findings,
            }
        except Exception as e:
            return {
                "score": 0.0,
                "passed": False,
                "findings": [{"type": "readability", "message": f"Error: {str(e)}"}],
            }
