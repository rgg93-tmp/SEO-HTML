import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluate.language import LanguageMatchEvaluator2, SpellingEvaluator, ReadabilityEvaluator


class TestLanguageEvaluators:
    """Test language evaluation functionality"""

    def test_language_match_evaluator_initialization(self):
        """Test LanguageMatchEvaluator2 can be initialized"""
        evaluator = LanguageMatchEvaluator2()
        assert evaluator is not None

    def test_language_match_evaluator_english(self):
        """Test LanguageMatchEvaluator2 with English text"""
        evaluator = LanguageMatchEvaluator2()

        english_text = "This is a beautiful modern apartment in downtown San Francisco."
        result = evaluator.evaluate(text=english_text, language_code="en", target_language="English")

        assert "evaluator" in result
        assert result["evaluator"] == "LanguageMatchEvaluator2"
        assert "score" in result
        assert "passed" in result
        assert "findings" in result
        assert isinstance(result["score"], float)
        assert isinstance(result["passed"], bool)
        assert isinstance(result["findings"], list)

    def test_language_match_evaluator_spanish(self):
        """Test LanguageMatchEvaluator2 with Spanish text"""
        evaluator = LanguageMatchEvaluator2()

        spanish_text = "Este es un hermoso apartamento moderno en el centro de San Francisco."
        result = evaluator.evaluate(text=spanish_text, language_code="es", target_language="Español")

        assert result["evaluator"] == "LanguageMatchEvaluator2"
        assert isinstance(result["score"], float)
        assert isinstance(result["passed"], bool)

    def test_spelling_evaluator_initialization(self):
        """Test SpellingEvaluator can be initialized"""
        evaluator = SpellingEvaluator()
        assert evaluator is not None

    def test_spelling_evaluator_correct_text(self):
        """Test SpellingEvaluator with correctly spelled text"""
        evaluator = SpellingEvaluator()

        correct_text = "This beautiful apartment has modern features."
        result = evaluator.evaluate(text=correct_text, language_code="en")

        assert "evaluator" in result
        assert result["evaluator"] == "SpellingEvaluator"
        assert "score" in result
        assert "passed" in result
        assert "findings" in result
        # Should pass with high score for correct text
        assert result["score"] > 0.8

    def test_spelling_evaluator_misspelled_text(self):
        """Test SpellingEvaluator with misspelled text"""
        evaluator = SpellingEvaluator()

        misspelled_text = "This beautifull apartmnt has modren featurez."
        result = evaluator.evaluate(text=misspelled_text, language_code="en")

        assert result["evaluator"] == "SpellingEvaluator"
        # Should fail or have lower score due to misspellings
        assert result["score"] < 1.0
        # Should have findings about misspelled words
        assert len(result["findings"]) > 0

    def test_readability_evaluator_initialization(self):
        """Test ReadabilityEvaluator can be initialized"""
        evaluator = ReadabilityEvaluator()
        assert evaluator is not None

    def test_readability_evaluator_simple_text(self):
        """Test ReadabilityEvaluator with simple text"""
        evaluator = ReadabilityEvaluator()

        simple_text = "This home is nice. It has two rooms. The kitchen is big. The price is good."
        result = evaluator.evaluate(text=simple_text, language_code="en")

        assert "evaluator" in result
        assert result["evaluator"] == "ReadabilityEvaluator"
        assert "score" in result
        assert "passed" in result
        assert "findings" in result
        assert isinstance(result["score"], float)
        assert 0.0 <= result["score"] <= 1.0

    def test_readability_evaluator_complex_text(self):
        """Test ReadabilityEvaluator with more complex text"""
        evaluator = ReadabilityEvaluator()

        complex_text = "This exceptional residential property exemplifies contemporary architectural sophistication while maintaining unprecedented accessibility to metropolitan commercial establishments and educational institutions."
        result = evaluator.evaluate(text=complex_text, language_code="en")

        assert result["evaluator"] == "ReadabilityEvaluator"
        assert isinstance(result["score"], float)
        # Complex text should have findings about readability
        assert len(result["findings"]) > 0

    def test_readability_difficulty_levels(self):
        """Test ReadabilityEvaluator difficulty level mapping"""
        evaluator = ReadabilityEvaluator()

        # Test the difficulty level function directly
        assert evaluator._get_difficulty_level(95) == "Very Easy"
        assert evaluator._get_difficulty_level(85) == "Easy"
        assert evaluator._get_difficulty_level(75) == "Fairly Easy"
        assert evaluator._get_difficulty_level(65) == "Standard"
        assert evaluator._get_difficulty_level(55) == "Fairly Difficult"
        assert evaluator._get_difficulty_level(35) == "Difficult"
        assert evaluator._get_difficulty_level(5) == "Very Confusing"
