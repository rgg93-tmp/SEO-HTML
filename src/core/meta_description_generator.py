from typing import Dict, Any


class MetaDescriptionGenerator:
    """
    Generator for the <meta name="description"> section of the property page.
    """

    def __init__(self, agent):
        self.agent = agent

    async def generate(self, property_data: Dict[str, Any]) -> str:
        meta = await self.agent.run(task=self.agent.build_user_prompt(property_data))
        meta_text = meta.messages[-1].content.strip()
        if len(meta_text) > 155:
            meta_text = meta_text[:152] + "..."
        return f'<meta name="description" content="{meta_text}">'
