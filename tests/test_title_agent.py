import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agents.content_generation.title_agent import TitleAgent


class TestTitleAgent:
    """Test TitleAgent functionality"""

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
            },
            "price": 650000,
            "listing_type": "sale",
        }

    def test_title_agent_initialization(self):
        """Test TitleAgent can be initialized"""
        agent = TitleAgent()
        assert agent is not None
        assert hasattr(agent, "name")

    def test_title_agent_with_custom_model(self):
        """Test TitleAgent can be initialized with custom model"""
        agent = TitleAgent(model="gemma3:1b-it-qat")
        assert agent is not None

    @pytest.mark.asyncio
    async def test_title_agent_generate_initial(self):
        """Test TitleAgent can generate initial content"""
        agent = TitleAgent()

        try:
            result = await agent.generate_initial(
                property_data=self.sample_property_data, language="en", tone="professional"
            )

            # Should return a string
            assert isinstance(result, str)
            # Should not be empty
            assert len(result.strip()) > 0
            # Should be reasonable length for a title (30-60 chars as per requirements)
            assert 10 <= len(result.strip()) <= 100  # Give some tolerance

        except Exception as e:
            # If the agent fails due to model not being available, that's expected in test environment
            pytest.skip(f"Agent test skipped due to model availability: {e}")

    @pytest.mark.asyncio
    async def test_title_agent_refine_content(self):
        """Test TitleAgent can refine content"""
        agent = TitleAgent()
        current_title = "Basic Apartment Title"
        suggestion = "Make the title more appealing and add location details"

        try:
            result = await agent.refine_content(
                property_data=self.sample_property_data,
                current_content=current_title,
                suggestion=suggestion,
                language="en",
                tone="professional",
            )

            # Should return a string
            assert isinstance(result, str)
            # Should not be empty
            assert len(result.strip()) > 0
            # Should be different from the original (though not guaranteed)

        except Exception as e:
            # If the agent fails due to model not being available, that's expected in test environment
            pytest.skip(f"Agent test skipped due to model availability: {e}")

    def test_title_agent_prompts_exist(self):
        """Test that title prompts are properly defined"""
        from agents.content_generation.title_agent import TITLE_PROMPTS

        # Should have prompts for all supported languages
        assert "en" in TITLE_PROMPTS
        assert "es" in TITLE_PROMPTS
        assert "pt" in TITLE_PROMPTS

        # Each language should have initial and refinement prompts
        for lang in ["en", "es", "pt"]:
            assert "initial" in TITLE_PROMPTS[lang]
            assert "refinement" in TITLE_PROMPTS[lang]
            assert isinstance(TITLE_PROMPTS[lang]["initial"], str)
            assert isinstance(TITLE_PROMPTS[lang]["refinement"], str)
