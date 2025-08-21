import gradio as gr
import logging
import json
from fastapi import FastAPI
from fastapi.responses import FileResponse

from core.html_generator import HTMLGenerator
from config.options import LANGUAGE_OPTIONS, TONE_OPTIONS, MODEL_OPTIONS


async def favicon() -> FileResponse:
    """
    Serves the favicon image for the application.
    Returns:
        FileResponse: A response object containing the favicon image file.
    """
    favicon_path = "./src/icon.png"
    return FileResponse(favicon_path)


async def generate_html_content(
    property_data: str,
    language: str = "en",
    tone: str = "professional",
    model: str = "gemma3n:e2b",
    max_iterations: int = 3,
) -> str:
    """
    Generate HTML content for a real estate listing based on property data.

    Args:
        property_data: JSON string containing property information
        language: Language for content generation (en, es, pt)
        tone: Tone for content generation (professional, friendly, luxury, etc.)
        model: Model to use for content generation
        max_iterations: Maximum number of refinement iterations

    Returns:
        str: Generated HTML content
    """
    try:
        data = json.loads(s=property_data)

        # Remove language and tone from property data if they exist
        if "language" in data:
            del data["language"]
        if "tone" in data:
            del data["tone"]

        # Create HTML generator with specified model and max_iterations
        html_generator = HTMLGenerator(model=model, max_iterations=max_iterations)

        # Generate HTML content with language and tone parameters
        html_content = await html_generator.generate_html(property_data=data, language=language, tone=tone)

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

            # Sample property data (without language/tone - those come from UI)
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
            }

            gr.Markdown("**Property Data (JSON):**")
            property_input = gr.Textbox(
                label="Property Data",
                placeholder="Enter property data in JSON format...",
                lines=8,
                value=json.dumps(obj=sample_data, indent=2),
            )

            # Language and Tone selection
            with gr.Row():
                language_dropdown = gr.Dropdown(
                    choices=list(LANGUAGE_OPTIONS.keys()),
                    value="en",
                    label="Language",
                    info="Select the language for content generation",
                )
                tone_dropdown = gr.Dropdown(
                    choices=list(TONE_OPTIONS.keys()),
                    value="family-oriented",
                    label="Tone",
                    info="Select the tone/style for the content",
                )

            # Settings panel (currently empty but could be expanded)
            with gr.Accordion(label="Advanced Settings", open=False):
                model_dropdown = gr.Dropdown(
                    choices=list(MODEL_OPTIONS.keys()),
                    value="gemma3n:e2b",
                    label="Model",
                    info="Select the AI model for content generation",
                )
                max_iterations_slider = gr.Slider(
                    minimum=0,
                    maximum=10,
                    value=1,
                    step=1,
                    label="Max Iterations",
                    info="Maximum number of refinement iterations (higher values = better quality but slower)",
                )

            run_btn = gr.Button("Generate HTML Content", variant="primary")

        with gr.Column():
            gr.Markdown("**Generated HTML Content:**")
            output_html = gr.HTML(label="Generated Content")

    run_btn.click(
        fn=generate_html_content,
        inputs=[property_input, language_dropdown, tone_dropdown, model_dropdown, max_iterations_slider],
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
