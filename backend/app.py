from flask import Flask, request, jsonify
from flask_cors import CORS

from emotion_detector import detect_emotion
from safety import safety_check, safety_response
from prompt import build_prompt
from local_model import generate_clean_response

app = Flask(__name__)
CORS(app)

@app.route("/chat", methods=["POST"])
def chat():
    user_text = request.json.get("message", "")
    history = request.json.get("history", [])

    if safety_check(user_text):
        return jsonify({
            "emotion": "distress",
            "response": safety_response()
        })

    emotion, score = detect_emotion(user_text)
    prompt = build_prompt(user_text, emotion, history)

    app.logger.info("Generating response...")
    return jsonify({
        "emotion": emotion,
        "response": generate_clean_response(prompt, user_text, emotion, history)
    })

if __name__ == "__main__":
    app.run(debug=True)
