import logging
import json
import os
import asyncio
from core.html_generator import HTMLGenerator
from models.ollama_model import OllamaModel

# Set logging level to INFO
logging.basicConfig(level=logging.INFO)


async def main():
    """Main function to generate sample HTML content from property data."""

    # Initialize the HTML generator with optional language model
    # You can set language_model=None to disable AI enhancement
    try:
        language_model = OllamaModel(model_name="gemma3:1b-it-qat", temperature=0.7, max_tokens=512)
        html_generator = HTMLGenerator(
            language_model=language_model,
            template_style="modern",
            include_seo=True,
            multilingual=False,
            default_language="en",
        )
        logging.info("HTML generator initialized with language model")
    except Exception as e:
        logging.warning(f"Could not initialize language model: {e}")
        html_generator = HTMLGenerator(language_model=None, template_style="modern", include_seo=True)
        logging.info("HTML generator initialized without language model")

    # Sample property data
    sample_properties = [
        {
            "title": "Modern home in San Francisco",
            "location": {"city": "San Francisco", "neighborhood": "Nob Hill"},
            "features": {
                "bedrooms": 3,
                "bathrooms": 2,
                "area_sqm": 167,
                "balcony": True,
                "parking": True,
                "elevator": False,
                "floor": 1,
                "year_built": 2010,
            },
            "price": 850000,
            "listing_type": "sale",
            "language": "en",
        },
        {
            "title": "T3 apartment in Lisbon",
            "location": {"city": "Lisbon", "neighborhood": "Campo de Ourique"},
            "features": {
                "bedrooms": 3,
                "bathrooms": 2,
                "area_sqm": 120,
                "balcony": True,
                "parking": False,
                "elevator": True,
                "floor": 2,
                "year_built": 2005,
            },
            "price": 650000,
            "listing_type": "sale",
            "language": "en",
        },
    ]

    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)

    # Generate HTML files for each property
    for i, property_data in enumerate(sample_properties, 1):
        # Save property data as JSON
        json_filename = f"data/property_{i}_data.json"
        with open(json_filename, "w") as f:
            json.dump(property_data, f, indent=2)

        # Generate and save HTML content using the HTML generator
        html_content = await html_generator.generate_html(property_data)
        html_filename = f"data/property_{i}_listing.html"
        with open(html_filename, "w") as f:
            f.write(html_content)

        logging.info(f"Generated {json_filename} and {html_filename}")

    logging.info("Sample property listings generated successfully!")


if __name__ == "__main__":
    asyncio.run(main())
