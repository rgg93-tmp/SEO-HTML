from typing import Dict, Any


class KeyFeaturesGenerator:
    """
    Generator for the <ul id="key-features"> section of the property page.
    """

    def __init__(self, agent):
        self.agent = agent

    async def generate(self, property_data: Dict[str, Any]) -> str:
        prompt = self.agent.build_user_prompt(property_data)
        features = await self.agent.run(task=prompt)
        features_text = features.messages[-1].content.strip()
        return f"<ul>{features_text}</ul>"
        features_list = [f for f in features_list if f.strip()][:5]
        lis = "".join([f"<li>{f.strip()}</li>" for f in features_list])
        return f'<ul id="key-features">\n{lis}\n</ul>'
