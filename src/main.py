import logging
import json
import os
import asyncio
from core.html_generator import HTMLGenerator

logging.basicConfig(level=logging.WARNING)


async def main():
    """Main function to generate sample HTML content from property data."""

    # Initialize the HTML generator
    html_generator = HTMLGenerator()
    logging.info("HTML generator initialized.")

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
        },
    ]

    # Ensure data directory exists
    os.makedirs(name="data", exist_ok=True)

    # Generate HTML files for each property
    for i, property_data in enumerate(iterable=sample_properties, start=1):
        # Save property data as JSON
        json_filename = f"data/property_{i}_data.json"
        with open(file=json_filename, mode="w") as f:
            json.dump(obj=property_data, fp=f, indent=2)

        # Generate and save HTML content using the HTML generator
        # For demo purposes, we'll use Spanish and luxury tone for property 1, English professional for property 2
        if i == 1:
            html_content = await html_generator.generate_html(property_data=property_data, language="es", tone="luxury")
        else:
            html_content = await html_generator.generate_html(
                property_data=property_data, language="en", tone="family-oriented"
            )

        html_filename = f"data/property_{i}_listing.html"
        with open(file=html_filename, mode="w") as f:
            f.write(html_content)

        logging.info(f"Generated {json_filename} and {html_filename}")

    logging.info("Sample property listings generated successfully!")


if __name__ == "__main__":
    asyncio.run(main=main())
