# after server is hosted on port 8000
from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "ai4ce"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_response = client.chat.completions.create(
    model="Qwen/Qwen3-VL-8B-Instruct",
    messages=[
        {"role": "user", "content": "Give me a short introduction to large language models."},
    ],
    temperature=0.6,
    top_p=0.95,
    extra_body={
        "top_k": 20,
    },
)
print("Chat response:", chat_response)
print("--------------------------------------------------------")
print("Response text:", chat_response.choices[0].message.content)
