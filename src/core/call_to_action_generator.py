from typing import Dict, Any


class CallToActionGenerator:
    """
    Generator for the <p class="call-to-action"> section of the property page.
    """

    def __init__(self, agent):
        self.agent = agent

    async def generate(self, property_data: Dict[str, Any]) -> str:
        prompt = self.agent.build_user_prompt(property_data)
        cta = await self.agent.run(task=prompt)
        cta_text = cta.messages[-1].content.strip()
        return f'<p class="call-to-action">{cta_text}</p>'
