from flask import Flask, request, jsonify
from flask_cors import CORS
from .utils import wants_affirmation_only


from .emotion_detector import detect_emotion
try:
    from .prompt import build_prompt, wants_affirmation_only
except ImportError:
    from .prompt import build_prompt

    def wants_affirmation_only(_text: str) -> bool:
        return False
from .safety import safety_level, safety_response
from .local_model import generate_clean_response

app = Flask(__name__)
CORS(app)

@app.route("/chat", methods=["POST"])
def chat():
    user_text = request.json.get("message", "")
    history = request.json.get("history", [])

    #SAFETY CHECK (FIRST)
    level = safety_level(user_text)

    if level == "high":
        return jsonify({
            "emotion": "distress",
            "response": safety_response()
        })

    #EMOTION DETECTION
    emotion_info = detect_emotion(user_text)

    #PROMPT BUILDING
    affirm_only = wants_affirmation_only(user_text, history)
    prompt = build_prompt(
        user_text,
        emotion_info,
        history,
        affirm_only=affirm_only
    )

    app.logger.info("Generating response...")

    #OLLAMA RESPONSE
    return jsonify({
        "emotion": emotion_info,
        "response": generate_clean_response(prompt)
    })

    

if __name__ == "__main__":
    app.run(debug=True)
