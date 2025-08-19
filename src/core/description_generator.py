from typing import Dict, Any


class DescriptionGenerator:
    """
    Generator for the <section id="description"> section of the property page.
    """

    def __init__(self, agent):
        self.agent = agent

    async def generate(self, property_data: Dict[str, Any]) -> str:
        desc = await self.agent.run(task=self.agent.build_user_prompt(property_data))
        desc_text = desc.messages[-1].content.strip()
        # Validate length
        if len(desc_text) > 700:
            desc_text = desc_text[:697] + "..."
        return f'<section id="description">\n<p>{desc_text}</p>\n</section>'
