def safety_check(text):
    keywords = [
        "suicide", "kill myself", "self harm",
        "die", "depressed", "worthless"
    ]

    for word in keywords:
        if word in text.lower():
            return True

    return False


def safety_response():
    return (
        "I'm really sorry you're feeling this way.\n\n"
        "You are not alone. Talking to someone you trust can help.\n\n"
        "United States: Call or text 988 for the Suicide & Crisis Lifeline.\n"
        "If you are in immediate danger, call your local emergency number.\n\n"
        "If you want, I can stay here and talk with you."
    )
