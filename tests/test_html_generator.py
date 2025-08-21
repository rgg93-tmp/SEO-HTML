import pytest
import sys
import os
import asyncio

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.html_generator import HTMLGenerator


class TestHTMLGenerator:
    """Test HTMLGenerator main functionality"""

    def setup_method(self):
        """Setup test data"""
        self.sample_property_data = {
            "title": "Modern apartment in downtown",
            "location": {"city": "San Francisco", "neighborhood": "Mission Bay"},
            "features": {
                "bedrooms": 2,
                "bathrooms": 1,
                "area_sqm": 80,
                "balcony": True,
                "parking": False,
                "elevator": True,
                "floor": 3,
                "year_built": 2020,
            },
            "price": 650000,
            "listing_type": "sale",
        }

    def test_html_generator_initialization(self):
        """Test HTMLGenerator can be initialized with default parameters"""
        generator = HTMLGenerator()
        assert generator.max_iterations == 1
        assert generator.model == "gemma3n:e2b"
        assert hasattr(generator, "agents")
        assert hasattr(generator, "complete_evaluator")
        assert hasattr(generator, "improvement_agent")

    def test_html_generator_initialization_with_params(self):
        """Test HTMLGenerator can be initialized with custom parameters"""
        generator = HTMLGenerator(max_iterations=5, model="gemma3:1b-it-qat")
        assert generator.max_iterations == 5
        assert generator.model == "gemma3:1b-it-qat"

    def test_html_generator_invalid_model(self):
        """Test HTMLGenerator raises error with invalid model"""
        with pytest.raises(ValueError, match="Unsupported model"):
            HTMLGenerator(model="invalid-model")

    def test_agents_initialization(self):
        """Test all required agents are initialized"""
        generator = HTMLGenerator()
        required_agents = ["title", "meta", "h1", "description", "key_features", "neighborhood", "call_to_action"]

        for agent_name in required_agents:
            assert agent_name in generator.agents
            assert generator.agents[agent_name] is not None

    @pytest.mark.asyncio
    async def test_generate_html_invalid_language(self):
        """Test HTMLGenerator raises error with invalid language"""
        generator = HTMLGenerator()

        with pytest.raises(ValueError, match="Unsupported language"):
            await generator.generate_html(
                property_data=self.sample_property_data, language="invalid", tone="professional"
            )

    @pytest.mark.asyncio
    async def test_generate_html_invalid_tone(self):
        """Test HTMLGenerator raises error with invalid tone"""
        generator = HTMLGenerator()

        with pytest.raises(ValueError, match="Unsupported tone"):
            await generator.generate_html(property_data=self.sample_property_data, language="en", tone="invalid")

    def test_html_generator_sections_initialization(self):
        """Test sections are properly initialized as empty"""
        generator = HTMLGenerator()
        # Should not have sections before generation
        assert not hasattr(generator, "sections") or generator.sections is None
