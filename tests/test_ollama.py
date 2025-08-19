from unittest import TestCase
import sys

sys.path.append("../src")
from ..src.models.ollama_model import OllamaModel


class TestOllama(TestCase):

    def setUp(self) -> None:
        self.model = OllamaModel(
            model_name="gemma3n:e2b",
            max_tokens=40,
            temperature=0.01,
        )

    def test_generate_single_response(self):
        response = self.model.generate("Test prompt?")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_generate_multiple_responses(self):
        count = 3
        responses = self.model.generate_multiple("Test prompt?", num_responses=count)
        assert isinstance(responses, list)
        assert len(responses) == count
        for response in responses:
            assert isinstance(response, str)

    def test_generate_city_response(self):
        response = self.model.generate("Respond only the city. The capital of France is: ")
        assert isinstance(response, str)
        # Note: The actual response may vary, so we just check it's a string
        assert len(response) > 0

    def test_model_info(self):
        info = self.model.get_model_info()
        assert "model_name" in info
        assert "model_type" in info
        assert info["model_type"] == "ollama"
        assert info["model_name"] == "gemma3n:e2b"
