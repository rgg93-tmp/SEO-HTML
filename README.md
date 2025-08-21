# 🏠 Real Estate: Website Content Generator

This project is a **Gradio web app** that generates **high‑quality, SEO‑optimized, multilingual content** for real property listing pages from structured property data.

It outputs JSON sections tagged by specific HTML elements so they can be integrated dynamically into a website template.

---

## 📘 Problem Statement

You have been hired by a real estate company that manages hundreds of property listings across different cities and regions. The company wants to automate the creation of high‑quality, SEO‑optimized, multilingual content for property listing pages on their website.

Your task is to build an AI‑powered system that generates all written content for these listing pages based on structured property data and a fixed content structure, with each section tagged by specific HTML elements so it can later be integrated dynamically into a website template.

---

## 🔄 Application Flow

The content generation follows a sophisticated multi-stage pipeline designed to produce high-quality, SEO-optimized content:

### 1. Input Processing
- **Property Data**: Structured JSON containing property details (location, features, price, etc.)
- **Configuration**: Language selection (English, Spanish, Portuguese), tone preference, AI model choice, and refinement iterations
- **Validation**: Input data is validated and preprocessed for content generation

### 2. Initial Content Generation
The **HTMLGenerator** orchestrates multiple specialized agents that work in parallel:
- **TitleAgent**: Generates SEO-optimized titles (30-60 characters)
- **MetaDescriptionAgent**: Creates compelling meta descriptions (max 155 characters)
- **H1Agent**: Produces main headlines
- **DescriptionAgent**: Writes detailed property descriptions (500-700 characters)
- **KeyFeaturesAgent**: Extracts and formats property features
- **NeighborhoodAgent**: Generates neighborhood descriptions
- **CallToActionAgent**: Creates engaging call-to-action content

### 3. Quality Assessment & Refinement Loop
After initial generation, the system enters an iterative refinement process:

#### Evaluation Phase
- **CompleteEvaluator**: Comprehensive analysis using specialized evaluators:
  - **LanguageMatchEvaluator**: Ensures content matches target language
  - **SpellingEvaluator**: Checks for spelling errors
  - **ReadabilityEvaluator**: Analyzes text readability using Flesch Reading Ease
  - **ToneMatchEvaluator**: Verifies tone consistency
  - **FactCheckerAgent**: Validates factual accuracy
  - **SEOEvaluator**: Ensures SEO best practices

#### Improvement Phase
- **ImprovementSuggestionAgent**: Analyzes evaluation findings and provides specific, actionable improvement suggestions for each content section
- **Multilingual Support**: Suggestions provided in the target language (English, Spanish, Portuguese)
- **Section-Specific**: Problems and solutions are isolated to their respective content sections

#### Refinement Phase
- Agents regenerate content based on improvement suggestions
- Process repeats for the configured number of iterations (1-10, default: 3)
- Each iteration improves content quality, SEO optimization, and linguistic accuracy

### 4. Final Output
- **Structured HTML**: Complete property listing page with semantic HTML tags
- **SEO-Optimized**: Title tags, meta descriptions, headers, and content optimized for search engines
- **Multilingual**: Content generated in the selected target language
- **Quality-Assured**: Multiple evaluation rounds ensure high-quality output

### Architecture Benefits
- **Modular Design**: Each agent specializes in specific content types
- **Parallel Processing**: Initial generation happens concurrently for efficiency
- **Iterative Improvement**: Multiple refinement cycles enhance quality
- **Language Flexibility**: Native support for multiple languages
- **Customizable**: Configurable models, iterations, and content parameters

---


## 📁 Project Structure

```bash
.
├── data/                           # Sample data and outputs
│   ├── property_1_data.json       # Example property data
│   ├── property_1_listing.html    # Generated HTML output
│   ├── property_2_data.json       # Another example property
│   └── property_2_listing.html    # Generated HTML output
├── src/                            # Main application code
│   ├── agents/                     # AI agents for content generation and evaluation
│   │   ├── content_generation/     # Content generation agents
│   │   │   ├── title_agent.py      # SEO-optimized title generation
│   │   │   ├── meta_description_agent.py # Meta description generation
│   │   │   ├── h1_agent.py         # H1 headline generation
│   │   │   ├── description_agent.py # Property description generation
│   │   │   ├── key_features_agent.py # Feature extraction and formatting
│   │   │   ├── neighborhood_agent.py # Neighborhood description
│   │   │   └── call_to_action_agent.py # CTA generation
│   │   └── evaluation/             # Content evaluation agents
│   │       ├── improvement_suggestion_agent.py # Improvement suggestions
│   │       ├── language_evaluator_agent.py # Language validation
│   │       ├── tone_evaluator_agent.py # Tone consistency
│   │       └── fact_checker_agent.py # Fact validation
│   ├── config/                     # Configuration settings
│   │   └── options.py              # Language, tone, and model options
│   ├── core/                       # Core application logic
│   │   └── html_generator.py       # Main HTML generation orchestrator
│   ├── evaluate/                   # Evaluation system
│   │   ├── base_evaluator.py       # Base evaluation class
│   │   ├── complete_evaluator.py   # Complete evaluation orchestrator
│   │   ├── language.py             # Language-related evaluators
│   │   ├── seo.py                  # SEO evaluation
│   │   └── fact.py                 # Fact checking evaluation
│   ├── models/                     # Model utilities (legacy)
│   ├── app.py                      # FastAPI application entry point
│   ├── real_estate_app.py          # Gradio UI application
│   ├── main.py                     # Example script for testing
│   └── icon.png                    # Application icon
├── tests/                          # Test suite
│   ├── test_html_generator.py      # HTMLGenerator tests
│   ├── test_title_agent.py         # TitleAgent tests
│   ├── test_evaluators.py          # Evaluator tests
│   ├── test_config_options.py      # Configuration tests
│   └── test_utilities.py           # Utility function tests
├── notebooks/                      # Jupyter notebooks for analysis
├── .flake8                         # Python linting configuration
├── mypy.ini                        # Type checking configuration
├── pytest.ini                     # Test configuration
├── pyproject.toml                  # Project dependencies and metadata
├── uv.lock                         # Dependency lock file
├── Dockerfile.Base                 # Base Docker image
├── Dockerfile.Code                 # Application Docker image
└── README.md                       # Project documentation
```

---

## 🛠️ Setup Instructions

This project uses [`uv`](https://github.com/astral-sh/uv) for Python dependency and environment management.

### Install `uv` and create venv (if not already)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
```

From there, UV will handle dependencies in the venv, according to pyproject.toml definition.

### Deploy models

The whole project uses models running locally with [`Ollama`](https://ollama.com/).

```bash
brew install ollama
ollama run **model_id**
```

Recommended models are:

- gemma3n:e2b
- gemma3:1b-it-qat


---

## ▶️ Run code
 
### main.py

There is a `main.py` example that triggers a sample run and writes outputs into `data/` for quick inspection.

```bash
uv run src/main.py
```

---

## 🖥️ Run app (Gradio UI)
 
### app.py

But the main goal should be to run the interactive app. The app is deployed locally with uvicorn on the port 5000. For that just run:

```bash
uv run src/app.py
```


After that, you should be able to navigate to:

[http://0.0.0.0:5000/realestate/](http://0.0.0.0:5000/realestate/)

You should see there an app like this:

![App example](image.png)

**Advanced Settings**: The UI includes configurable parameters:
- **Model Selection**: Choose between `gemma3n:e2b` (recommended) or `gemma3:1b-it-qat` (lightweight)
- **Max Iterations**: Control refinement cycles (1-10, higher values = better quality but slower processing)
- **Language & Tone**: Select target language and content tone for optimal results

## 🧹 Linters

The whole code follows PEP8, checks cyclomatic complexity and incorporates type hinting. It is highly recommended to check linters before deploying code or creating pull requests.

```bash
flake8 src
uv run mypy .
```

## 🧪 Tests

The project includes comprehensive unit tests for core functionality including:
- HTMLGenerator and content generation agents
- Evaluation system and language validators
- Configuration options and utility functions

Tests can be run with:

```bash
uv run pytest tests
```

For verbose output:
```bash
uv run pytest tests -v
```

---

## 🔗 References
[Gradio](https://www.gradio.app/)

---

## 🔒 License
This project is proprietary and intended for internal use only.
Unauthorized copying, modification, or distribution of any part of this project is strictly prohibited.
