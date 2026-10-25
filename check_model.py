"""Check which chat models your Vocareum key is allowed to use."""

import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1"),
)

MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4.1",
    "gpt-4-turbo",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-3.5-turbo",
]

for model in MODELS:
    try:
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say OK"}],
            max_completion_tokens=20,
        )
        print(f"OK      {model}")
    except Exception as exc:
        print(f"FAILED  {model}  ->  {str(exc)[:110]}")