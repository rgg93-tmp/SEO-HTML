"""
Configuration settings for language and tone options.
"""

# Language options with their codes and full names
LANGUAGE_OPTIONS = {
    "en": {"name": "English", "code": "en", "spell_check_code": "en"},
    "es": {"name": "Español", "code": "es", "spell_check_code": "es"},
    "pt": {"name": "Português", "code": "pt", "spell_check_code": "pt"},
}

# Tone options with descriptions
TONE_OPTIONS = {
    "luxury": {
        "name": "Luxury",
        "description": "Sophisticated, exclusive, and premium. Highlight high-end finishes, amenities, and prestige.",
    },
    "investor-focused": {
        "name": "Investor-Focused",
        "description": "Data-driven, analytical, and ROI-oriented. Emphasize investment potential, rental yield, and market growth.",
    },
    "family-oriented": {
        "name": "Family-Oriented",
        "description": "Welcoming, safe, and community-focused. Highlight schools, parks, and family-friendly features.",
    },
    "modern": {
        "name": "Modern",
        "description": "Contemporary, trendy, and innovative. Emphasize new technology, design, and urban lifestyle.",
    },
    "classic": {
        "name": "Classic",
        "description": "Timeless, elegant, and traditional. Highlight architectural details, heritage, and enduring value.",
    },
    "youthful": {
        "name": "Youthful",
        "description": "Joyful, vibrant, and affordable. Highlight fun activities, social life, and low price for young buyers.",
    },
}

# Model options with descriptions
MODEL_OPTIONS = {
    "gemma3n:e2b": {
        "name": "Gemma3n E2B",
        "description": "Recommended high-performance model with optimized efficiency",
    },
    "gemma3:1b-it-qat": {
        "name": "Gemma3 1B IT QAT",
        "description": "Lightweight quantized model for faster inference",
    },
}


def get_language_options():
    """Get list of language options for dropdowns."""
    return [(lang["name"], code) for code, lang in LANGUAGE_OPTIONS.items()]


def get_tone_options():
    """Get list of tone options for dropdowns."""
    return [(f"{tone['name']} - {tone['description']}", code) for code, tone in TONE_OPTIONS.items()]


def get_model_options():
    """Get list of model options for dropdowns."""
    return [(f"{model['name']} - {model['description']}", code) for code, model in MODEL_OPTIONS.items()]


def get_language_name(code: str) -> str:
    """Get language name from code."""
    return LANGUAGE_OPTIONS.get(code, {}).get("name", "English")


def get_tone_name(code: str) -> str:
    """Get tone name from code."""
    return TONE_OPTIONS.get(code, {}).get("name", "Professional")
