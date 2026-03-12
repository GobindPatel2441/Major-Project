import re

# -----------------------------
# Affirmation Detection
# -----------------------------

AFFIRM_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"stop asking",
        r"stop asking questions",
        r"no questions?",
        r"dont ask",
        r"don't ask",
        r"dont ask questions",
        r"don't ask questions",
        r"just reassure",
        r"just support",
        r"just tell me",
        r"only support",
        r"wish me luck",
        r"no advice",
        r"i just need (support|reassurance|encouragement)",
        r"just need affirmation",
        r"i just want affirmation",
    ]
]


def wants_affirmation_only(text: str, history=None) -> bool:
    text = text.lower().strip()

    # Check current message
    if any(pattern.search(text) for pattern in AFFIRM_PATTERNS):
        return True

    history = history or []

    # Check recent user messages
    for item in reversed(history[-6:]):
        if item.get("role") != "user":
            continue

        content = (item.get("content") or item.get("text", "")).lower().strip()

        if any(pattern.search(content) for pattern in AFFIRM_PATTERNS):
            return True

    return False


# -----------------------------
# Prompt Builder
# -----------------------------

def build_prompt(user_text, emotion_info, history=None, affirm_only=False):

    user_text = user_text.strip()
    emotion_info = emotion_info or {}

    primary = (emotion_info.get("primary") or "neutral")
    severity = (emotion_info.get("severity") or "low")

    # Tone selection
    tone = "Be calm, validating, and kind."
    if severity in {"medium", "high"}:
        tone = "Be extra gentle, grounding, and validating."

    # Emotion-specific guidance
    emotion_guidance = ""

    if primary == "anxiety":
        emotion_guidance = """
If the user feels anxious:
- Offer reassurance
- Normalize uncertainty
- Encourage calm perspective
"""

    elif primary == "sadness":
        emotion_guidance = """
If the user feels sad:
- Validate their feelings
- Avoid jumping straight to solutions
- Show empathy
"""

    elif primary == "frustration":
        emotion_guidance = """
If the user feels frustrated:
- Acknowledge the difficulty
- Validate their effort
- Respond calmly
"""

    elif primary == "anger":
        emotion_guidance = """
If the user feels angry:
- Stay calm and non-judgmental
- Acknowledge the emotion
- Avoid escalating
"""

    history = history or []

    # Limit history for token efficiency
    recent_history = history[-4:]

    history_lines = []
    for item in recent_history:
        role = item.get("role", "user")
        content = item.get("content") or item.get("text", "")
        history_lines.append(f"{role.upper()}: {content}")

    history_block = "\n".join(history_lines) or "None"

    # Question behavior
    if affirm_only:
        question_rule = """
Conversation mode: affirmation_only = True

Rules:
- NEVER ask questions
- NEVER request clarification
- Provide reassurance and encouragement
- End with a supportive statement
"""
    else:
        question_rule = """
Conversation mode: affirmation_only = False

Rules:
- You may ask ONE gentle follow-up question
- Only if it helps the user continue talking
- Do NOT ask multiple questions
- Do NOT interrogate the user
"""

    prompt = f"""
You are a calm, emotionally intelligent AI companion.

User emotional state:
Emotion: {primary}
Severity: {severity}

Tone:
{tone}

Emotion guidance:
{emotion_guidance}

Response rules:
- Keep replies SHORT (2–4 sentences)
- Sound natural and human
- Be supportive and empathetic
- Never argue with the user
- Never lecture
- Never over-explain

{question_rule}

Conversation history:
{history_block}

User message:
"{user_text}"

Respond like a caring friend.
"""

    return prompt