from typing import Dict, Any


class TitleGenerator:
    """
    Generator for the <title> section of the property page.
    """

    def __init__(self, agent):
        self.agent = agent

    async def generate(self, property_data: Dict[str, Any]) -> str:
        # Call the agent to generate the text
        prompt = self.agent.build_user_prompt(property_data)
        title = await self.agent.run(task=prompt)
        # Validate max length 60 characters
        title_text = title.messages[-1].content.strip()
        if len(title_text) > 60:
            title_text = title_text[:57] + "..."
        return f"<title>{title_text}</title>"
