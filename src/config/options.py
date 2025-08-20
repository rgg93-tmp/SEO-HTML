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
    "professional": {"name": "Professional", "description": "Formal, trustworthy, and business-oriented"},
    "friendly": {"name": "Friendly", "description": "Warm, approachable, and conversational"},
    "luxury": {"name": "Luxury", "description": "Sophisticated, exclusive, and premium"},
    "investor-focused": {"name": "Investor-Focused", "description": "Data-driven, analytical, and ROI-oriented"},
    "family-oriented": {"name": "Family-Oriented", "description": "Welcoming, safe, and community-focused"},
    "modern": {"name": "Modern", "description": "Contemporary, trendy, and innovative"},
    "classic": {"name": "Classic", "description": "Timeless, elegant, and traditional"},
    "casual": {"name": "Casual", "description": "Relaxed, informal, and easy-going"},
}


def get_language_options():
    """Get list of language options for dropdowns."""
    return [(lang["name"], code) for code, lang in LANGUAGE_OPTIONS.items()]


def get_tone_options():
    """Get list of tone options for dropdowns."""
    return [(f"{tone['name']} - {tone['description']}", code) for code, tone in TONE_OPTIONS.items()]


def get_language_name(code: str) -> str:
    """Get language name from code."""
    return LANGUAGE_OPTIONS.get(code, {}).get("name", "English")


def get_tone_name(code: str) -> str:
    """Get tone name from code."""
    return TONE_OPTIONS.get(code, {}).get("name", "Professional")
