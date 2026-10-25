import os
from typing import Dict, List
from openai import OpenAI


def generate_response(openai_key: str, user_message: str, context: str,
                      conversation_history: List[Dict],
                      model: str = "gpt-4o-mini") -> str:
    """Generate response using OpenAI with context"""

    # Define system prompt
    system_prompt = """
You are a NASA mission expert. Answer questions accurately using the
provided NASA mission context.

Use the provided context as your primary source of information.
Cite the retrieved source documents when providing your answer.

Do not make claims that are not supported by the provided context.
If the context does not contain enough information to answer the question,
clearly state that you do not have enough information.
"""
    # Set context in messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}"}
    ]

    # Add conversation history
    messages.extend(conversation_history)

    # Add current user question
    messages.append({"role": "user", "content": user_message})

    # Create OpenAI client (Vocareum endpoint, same as the other scripts)
    client = OpenAI(
        api_key=openai_key,
        base_url=os.getenv(
            "OPENAI_BASE_URL",
            "https://openai.vocareum.com/v1"
        )
    )

    # Send request to OpenAI
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )

    # Return response
    return response.choices[0].message.content