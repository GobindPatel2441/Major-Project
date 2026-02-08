import os
import joblib
from pathlib import Path

from .utils import (
    intensity_to_severity,
    handle_negation,
    get_polarity,
    polarity_gate,
    suppress_secondary_emotions
)


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.getenv("EMOTION_MODEL_DIR", BASE_DIR / "Model")).resolve()

EMOTIONS = ["anger", "fear", "joy", "sadness"]

_models = {}
_vectorizers = {}


def _load_models():
    if _models:
        return

    for emo in EMOTIONS:
        _models[emo] = joblib.load(MODEL_DIR / f"{emo}_svr_model.joblib")
        _vectorizers[emo] = joblib.load(MODEL_DIR / f"{emo}_svr_vectorizer.joblib")


def detect_emotion(text: str):
    _load_models()

    # 1. Negation handling
    text_proc = handle_negation(text)

    # 2. Raw SVR predictions
    raw = {}
    for emo in EMOTIONS:
        vec = _vectorizers[emo].transform([text_proc])
        raw[emo] = float(_models[emo].predict(vec)[0])

    # 3. Polarity gating
    polarity = get_polarity(text_proc)
    gated = polarity_gate(raw, polarity)

    # 4. Dominant suppression
    final_scores = suppress_secondary_emotions(gated)

    # 5. Pick dominant emotion
    dominant = max(final_scores, key=final_scores.get)
    intensity = final_scores[dominant]
    severity = intensity_to_severity(intensity)

    return {
        "primary": dominant,
        "severity": severity,
        "intensity": round(intensity, 3),
        "all": {k: round(v, 3) for k, v in final_scores.items()}
    }
