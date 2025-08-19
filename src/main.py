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
            "address": "123 Main Street",
            "city": "San Francisco, CA",
            "price": 850000,
            "beds": 3,
            "baths": 2,
            "sqft": 1800,
            "description": "Beautiful modern home in a prime location. This stunning property features an open floor plan, updated kitchen with granite countertops, and a spacious backyard perfect for entertaining. Located in a highly sought-after neighborhood with excellent schools and convenient access to shopping and transportation.",
            "features": [
                "Open floor plan",
                "Updated kitchen with granite countertops",
                "Hardwood floors throughout",
                "Spacious backyard",
                "2-car garage",
                "Central air conditioning",
                "Fresh paint",
                "New appliances",
            ],
        },
        {
            "address": "456 Oak Avenue",
            "city": "Los Angeles, CA",
            "price": 1200000,
            "beds": 4,
            "baths": 3,
            "sqft": 2200,
            "description": "Luxurious family home with stunning views. This exceptional property offers the perfect blend of comfort and elegance. Features include a gourmet kitchen, master suite with spa-like bathroom, and a beautiful garden with outdoor entertainment area. Located in a prestigious neighborhood with top-rated schools.",
            "features": [
                "Gourmet kitchen with island",
                "Master suite with spa bathroom",
                "Beautiful garden with outdoor entertainment area",
                "3-car garage",
                "Smart home features",
                "Solar panels",
                "Pool and spa",
                "Wine cellar",
            ],
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
