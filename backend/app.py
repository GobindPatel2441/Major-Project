from flask import Flask, request, jsonify, Response
import json
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
from .local_model import generate_clean_response, generate_clean_response_stream, OLLAMA_URL
import requests

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

    #OLLAMA RESPONSE STREAM
    def generate():
        yield json.dumps({"type": "emotion", "data": emotion_info}) + "\n"
        for chunk in generate_clean_response_stream(prompt):
            yield json.dumps({"type": "chunk", "data": chunk}) + "\n"
            
    return Response(generate(), mimetype='application/x-ndjson')


@app.route("/status", methods=["GET"])
def status():
    """
    Lightweight health check for the Ollama backend.
    Tries to reach the Ollama HTTP API; if it responds at all,
    we treat the AI server as online.
    """
    health_url = OLLAMA_URL.replace("/api/generate", "/api/tags")
    try:
        response = requests.get(health_url, timeout=3)
        online = response.ok
    except requests.exceptions.RequestException:
        online = False

    return jsonify({"status": "online" if online else "offline"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
