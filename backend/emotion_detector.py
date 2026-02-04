positive_words = [
    "happy", "good", "great", "awesome", "excited",
    "joy", "love", "relaxed", "calm", "peaceful"
]

negative_words = [
    "sad", "angry", "upset", "depressed", "tired",
    "lonely", "anxious", "stressed", "worried"
]

def detect_emotion(text):
    text = text.lower()
    pos = sum(word in text for word in positive_words)
    neg = sum(word in text for word in negative_words)

    if pos > neg:
        return "positive", pos
    elif neg > pos:
        return "negative", neg
    else:
        return "neutral", 0
