def build_prompt(user_text, emotion, history=None):
    history = history or []
    history_lines = []
    for msg in history[-8:]:
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = msg.get("text", "").strip()
        if content:
            history_lines.append(f"{role}: {content}")
    history_block = "\n".join(history_lines)

    return f"""
You are a warm, friendly AI companion chatting with a human.

Conversation style rules:
- Keep replies SHORT (2-4 sentences max)
- Sound natural, casual, and human
- Do NOT sound like a therapist or textbook
- React first, then ask ONE simple question
- Avoid long explanations or emotional analysis
- Be curious, not preachy
- Ask friend-like follow-ups (what/why/how)

User emotion: {emotion}

Conversation so far:
{history_block}

User message:
"{user_text}"

Respond like a caring friend in a real chat.
### Response:
"""
