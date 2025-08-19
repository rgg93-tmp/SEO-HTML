from typing import Dict, Any


class H1Generator:
    """
    Generator for the <h1> section of the property page.
    """

    def __init__(self, agent):
        self.agent = agent

    async def generate(self, property_data: Dict[str, Any]) -> str:
        prompt = self.agent.build_user_prompt(property_data)
        h1 = await self.agent.run(task=prompt)
        h1_text = h1.messages[-1].content.strip()
        return f"<h1>{h1_text}</h1>"
