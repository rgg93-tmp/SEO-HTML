import gradio as gr
import logging
import json
from fastapi import FastAPI
from fastapi.responses import FileResponse
from typing import Tuple

from core.html_generator import HTMLGenerator
from models.ollama_model import OllamaModel


# Initialize the HTML generator (will be created per request to handle errors gracefully)
def create_html_generator(use_language_model: bool = True):
    """Create an HTML generator instance with optional language model."""
    try:
        if use_language_model:
            language_model = OllamaModel(model_name="gemma3:1b-it-qat", temperature=0.7, max_tokens=512)
            return HTMLGenerator(
                language_model=language_model,
                template_style="modern",
                include_seo=True,
                multilingual=False,
                default_language="en",
            )
        else:
            return HTMLGenerator(language_model=None, template_style="modern", include_seo=True)
    except Exception as e:
        logging.warning(f"Could not initialize language model: {e}")
        return HTMLGenerator(language_model=None, template_style="modern", include_seo=True)


async def favicon() -> FileResponse:
    """
    Serves the favicon image for the application.
    Returns:
        FileResponse: A response object containing the favicon image file.
    """
    favicon_path = "./src/icon.png"
    return FileResponse(favicon_path)


import asyncio


async def generate_html_content(property_data: str, template_style: str = "modern", use_ai: bool = True) -> str:
    """
    Generate HTML content for a real estate listing based on property data.

    Args:
        property_data: JSON string containing property information
        template_style: Style template to use ("modern", "classic", "minimal")
        use_ai: Whether to use AI enhancement for descriptions

    Returns:
        str: Generated HTML content
    """
    try:
        data = json.loads(property_data)

        # Create HTML generator with specified settings
        html_generator = create_html_generator(use_ai)
        html_generator.template_style = template_style

        # Generate HTML content
        html_content = await html_generator.generate_html(data)

        return html_content

    except json.JSONDecodeError:
        return "<p>Error: Invalid JSON data provided</p>"
    except Exception as e:
        return f"<p>Error generating content: {str(e)}"


# Build Gradio interface
with gr.Blocks(
    title="Real Estate Content Generator",
    head='<link rel="icon" href="/realestate/favicon.ico">',
) as re_app:

    with gr.Row():
        with gr.Column():
            gr.Markdown("# Real Estate Website Content Generator")
            gr.Markdown(
                "Create high-quality, SEO-optimized HTML content for property listing pages based on structured data."
            )

            # Sample property data
            sample_data = {
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
            }

            gr.Markdown("**Property Data (JSON):**")
            property_input = gr.Textbox(
                label="Property Data",
                placeholder="Enter property data in JSON format...",
                lines=8,
                value=json.dumps(sample_data, indent=2),
            )

            # Settings panel
            with gr.Accordion(label="Generation Settings", open=False):
                template_style = gr.Radio(
                    choices=["modern", "classic", "minimal"], label="Template Style", value="modern"
                )
                use_ai = gr.Checkbox(
                    label="Use AI Enhancement", value=True, info="Enhance descriptions using language model"
                )

            run_btn = gr.Button("Generate HTML Content", variant="primary")

        with gr.Column():
            gr.Markdown("**Generated HTML Content:**")
            output_html = gr.HTML(label="Generated Content")

    run_btn.click(
        fn=generate_html_content,
        inputs=[property_input, template_style, use_ai],
        outputs=[output_html],
    )


# Mount gradio interface on top of FastAPI app
def mount_realestate_app(app: FastAPI) -> FastAPI:
    """
    Mounts the Real Estate Content Generator on top of a FastAPI app.

    Args:
        app: The FastAPI application instance to which routes and Gradio app will be added.

    Returns:
        The FastAPI application instance with the Real Estate UI mounted.
    """
    logging.info("Gradio Real Estate app initialized.")

    app.add_api_route("/realestate/favicon.ico", favicon, methods=["GET"])

    # Mount the Gradio app
    app = gr.mount_gradio_app(app, re_app, path="/realestate")

    logging.info("Gradio Real Estate app mounted.")

    return app
