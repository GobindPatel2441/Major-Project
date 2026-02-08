def wants_affirmation_only(text: str, history=None) -> bool:
    text = text.lower()
    triggers = [
        "stop asking",
        "stop asking questions",
        "no questions",
        "no question",
        "just tell me",
        "just reassure me",
        "just need affirmation",
        "i just want affirmation",
        "just say something",
        "don't ask",
        "dont ask",
        "don't ask questions",
        "dont ask questions",
        "wish me luck",
        "just support me"
    ]
    if any(t in text for t in triggers):
        return True

    history = history or []
    # If the user recently asked for no questions, keep affirm-only for a bit.
    recent = history[-6:]
    for item in reversed(recent):
        if item.get("role") != "user":
            continue
        content = item.get("content", "").lower()
        if any(t in content for t in triggers):
            return True
    return False


def build_prompt(user_text, emotion_info, history=None, affirm_only=False):
    primary = emotion_info.get("primary", "neutral")
    severity = emotion_info.get("severity", "low")

    tone = "Be calm, validating, and kind."
    if severity in {"medium", "high"}:
        tone = "Be extra gentle, grounding, and validating."

    history = history or []
    history_block = ""
    if history:
        # Expecting list of {"role": "...", "content": "..."} objects
        lines = []
        for item in history:
            role = item.get("role", "user")
            content = item.get("content", "")
            lines.append(f"{role}: {content}")
        history_block = "\n".join(lines)

    if affirm_only:
        question_rule = "Do NOT ask any questions. End with a firm, supportive statement."
    else:
        question_rule = "React first, then ask ONE gentle follow-up question."

    return f"""
You are an emotionally intelligent AI companion.

Detected user emotion:
- Emotion: {primary}
- Severity: {severity}

Response guidelines:
- {tone}
- Keep replies SHORT (2-4 sentences)
- Sound natural and human
- {question_rule}
- NEVER argue with the user
- NEVER over-explain

Conversation so far:
{history_block}

User message:
"{user_text}"

Respond like a caring friend.
"""
