import os
from typing import Dict, List
from openai import OpenAI


# Only the most recent turns are sent to the model, so long chats stay
# within the context window and keep costs predictable.
# 10 messages = the last 5 user/assistant exchanges.
MAX_HISTORY_MESSAGES = 10


def generate_response(openai_key: str, user_message: str, context: str,
                      conversation_history: List[Dict],
                      model: str = "gpt-4o-mini") -> str:
    """Generate response using OpenAI with context"""

    if not isinstance(user_message, str) or not user_message.strip():
        raise ValueError("user_message must be a non-empty string")

    # Define system prompt
    system_prompt = """
    You are a NASA mission expert. Answer questions accurately using the
    provided NASA mission context.

    Use the provided context as your primary source of information.
    If the context does not contain enough information to answer the question,
    clearly state that you do not have enough information.
    """

    # Set context in messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}"}
    ]

    # Add the most recent conversation history only
    recent_history = [
        {"role": m["role"], "content": m["content"]}
        for m in (conversation_history or [])[-MAX_HISTORY_MESSAGES:]
        if isinstance(m, dict)
        and m.get("role") in {"user", "assistant"}
        and m.get("content")
    ]
    messages.extend(recent_history)

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