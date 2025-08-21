import openai

# Point to Ollama's OpenAI-compatible endpoint
client = openai.OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")  # The key is ignored by Ollama

response = client.chat.completions.create(
    model="gemma3n:e2b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Respond only the city. The capital of France is: "},
    ],
)

print(response.choices)
print(response.choices[0].message.content)
