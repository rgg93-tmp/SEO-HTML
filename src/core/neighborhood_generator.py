from typing import Dict, Any


class NeighborhoodGenerator:
    """
    Generator for the <section id="neighborhood"> section of the property page.
    """

    def __init__(self, agent):
        self.agent = agent

    async def generate(self, property_data: Dict[str, Any]) -> str:
        nb = await self.agent.run(
            task=f"Generate a lifestyle paragraph about the neighborhood (<p>) for the property with this data: {property_data}"
        )
        nb_text = nb.messages[-1].content.strip()
        prompt = self.agent.build_user_prompt(property_data)
        neighborhood = await self.agent.run(task=prompt)
        neighborhood_text = neighborhood.messages[-1].content.strip()
        return f"<p>{neighborhood_text}</p>"
