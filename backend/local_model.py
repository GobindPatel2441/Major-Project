from pathlib import Path
import sys

import torch
from tokenizers import Tokenizer

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "Model"
DATA_DIR = MODEL_DIR / "data"
CHECKPOINT = MODEL_DIR / "checkpoints" / "latest.pt"

# Allow importing the training model definition
sys.path.append(str(MODEL_DIR))
from train import GPT, GPTConfig  # noqa: E402

_model = None
_tokenizer = None
_device = "cuda" if torch.cuda.is_available() else "cpu"
_last_response = ""
_last_topic = None
_last_user_text = ""


def _load_model():
    global _model, _tokenizer
    if _model is not None and _tokenizer is not None:
        return

    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    meta_path = DATA_DIR / "meta.json"
    tokenizer_path = DATA_DIR / "tokenizer.json"
    if not meta_path.exists() or not tokenizer_path.exists():
        raise FileNotFoundError("Tokenizer/meta not found. Run prepare_data.py first.")

    _tokenizer = Tokenizer.from_file(str(tokenizer_path))

    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    config = GPTConfig(**ckpt["config"])
    _model = GPT(config)
    _model.load_state_dict(ckpt["model"])
    _model.eval()
    _model.to(_device)


@torch.no_grad()
def _generate(idx, max_new_tokens=80, temperature=1.0, top_k=40):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -_model.config.block_size :]
        logits, _ = _model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    return idx


def generate_response(prompt: str, max_new_tokens: int = 40) -> str:
    _load_model()
    ids = _tokenizer.encode(prompt).ids
    if not ids:
        ids = [0]
    # Keep only the last block_size tokens for speed
    if len(ids) > _model.config.block_size:
        ids = ids[-_model.config.block_size :]
    idx = torch.tensor([ids], dtype=torch.long, device=_device)
    out = _generate(idx, max_new_tokens=max_new_tokens)
    text = _tokenizer.decode(out[0].tolist())
    # Basic cleanup for byte-level artifacts and prompt echo
    if "### Response:" in text:
        text = text.split("### Response:", 1)[-1]
    if prompt in text:
        text = text.split(prompt, 1)[-1]
    text = (
        text.replace("Ċ", "\n")
        .replace("Ġ", " ")
        .replace("Â", " ")
        .replace("â€“", "-")
        .strip()
    )
    text = " ".join(text.split())

    # Keep response short (first 2 sentences if possible)
    for sep in [". ", "? ", "! "]:
        parts = text.split(sep)
        if len(parts) > 2:
            text = sep.join(parts[:2]) + sep.strip()
            break
    return text


def _pick_response(candidates, last_bot=None):
    global _last_response
    for text in candidates:
        if text != _last_response and text != (last_bot or ""):
            _last_response = text
            return text
    return candidates[0]


def safe_fallback_response(user_text: str, emotion: str, history=None) -> str:
    # Simple, clean conversational fallback with light variety
    text = user_text.lower().strip()
    global _last_topic, _last_user_text
    history = history or []
    last_bot = ""
    for msg in reversed(history):
        if msg.get("role") == "bot":
            last_bot = msg.get("text", "")
            break

    def set_topic(topic):
        global _last_topic, _last_user_text
        _last_topic = topic
        _last_user_text = text

    # If the user asks a question, answer briefly then ask one follow-up
    if text.endswith("?") or any(q in text for q in ["what", "why", "how", "when", "where", "should i"]):
        return _pick_response([
            "That’s a good question. What made you ask?",
            "Hmm, that depends. What’s most important to you here?",
        ], last_bot)

    # Direct emotion cues in user text
    if any(word in text for word in ["sad", "down", "lonely", "tired", "upset", "overwhelmed"]):
        set_topic("low")
        return _pick_response([
            "I’m sorry you’re feeling that way. Want to tell me what’s been weighing on you?",
            "That sounds rough. Do you want to talk about what happened?",
        ], last_bot)

    if any(word in text for word in ["happy", "excited", "great", "good", "awesome", "amazing"]):
        set_topic("positive")
        return _pick_response([
            "Love that for you. What made you feel that way?",
            "That’s great to hear. What was the best part?",
        ], last_bot)

    if any(word in text for word in ["hi", "hello", "hey"]):
        set_topic("greeting")
        return _pick_response([
            "Hey! How’s your day going so far?",
            "Hi there! What’s on your mind today?",
            "Hello! What’s been the highlight of your day?",
        ], last_bot)

    if any(word in text for word in ["reward", "won", "prize", "award"]):
        set_topic("reward")
        return _pick_response([
            "That’s awesome—congrats! What did you get it for?",
            "Nice! How did you feel when you got the reward?",
            "That’s great news. Why do you think it meant so much to you?",
        ], last_bot)

    if "hackathon" in text:
        set_topic("hackathon")
        return _pick_response([
            "That’s huge—congrats on the hackathon win! What did you build?",
            "Nice! What was your project about?",
            "That’s impressive. How did your team come up with the idea?",
        ], last_bot)

    if any(phrase in text for phrase in ["it was", "felt", "really awesome", "so good", "amazing"]):
        if _last_topic == "hackathon":
            return _pick_response([
                "Love that. What part of the hackathon was your favorite?",
                "That’s a great feeling. Who were you on a team with?",
                "Why do you think it clicked so well?",
            ], last_bot)
        if _last_topic == "reward":
            return _pick_response([
                "You deserve it. Want to celebrate somehow?",
                "That’s a proud moment. What was the best part?",
                "How did you celebrate afterward?",
            ], last_bot)

    if emotion.lower() in {"sad", "sadness", "depression", "fear", "anger", "disgust"}:
        set_topic("low")
        return _pick_response([
            "I’m here for you. Want to share a little more about what’s been tough?",
            "That sounds heavy. What’s been the hardest part?",
            "What’s been weighing on you the most lately?",
        ], last_bot)

    if emotion.lower() in {"happy", "joy", "positive", "love"}:
        if _last_topic in {"reward", "hackathon"}:
            return _pick_response([
                "That’s awesome. What happened next?",
                "So cool. Did you celebrate after?",
                "Why do you think that moment felt so good?",
            ], last_bot)
        set_topic("positive")
        return _pick_response([
            "That sounds nice. What made your day feel good?",
            "Love that for you. What’s been the best part so far?",
            "How did it all start?",
        ], last_bot)

    if _last_topic == "hackathon":
        return _pick_response([
            "What was the coolest challenge you solved?",
            "How long did you work on the project?",
            "Why did you choose that problem to solve?",
        ], last_bot)

    return _pick_response([
        "Got it. What’s been on your mind lately?",
        "Thanks for sharing. Want to tell me more about it?",
        "What’s the one thing you want to talk about most right now?",
    ], last_bot)


def generate_clean_response(prompt: str, user_text: str, emotion: str, history=None) -> str:
    try:
        text = generate_response(prompt)
    except Exception:
        return safe_fallback_response(user_text, emotion, history)

    # If model echoes prompt or is too long, fallback to clean template
    if "conversation style rules" in text.lower() or len(text) > 320:
        return safe_fallback_response(user_text, emotion, history)
    return text
