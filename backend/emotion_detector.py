import os
import joblib
from pathlib import Path

from .utils import (
    intensity_to_severity,
    handle_negation,
    get_polarity,
    polarity_gate,
    suppress_secondary_emotions,
)


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.getenv("EMOTION_MODEL_DIR", BASE_DIR / "Model")).resolve()

EMOTIONS = ["anger", "fear", "joy", "sadness"]

_models = {}
_vectorizers = {}


def _load_models():
    """
    Lazily load SVR models if they are present on disk.
    If model files are missing, we leave the caches empty so that
    downstream logic can fall back to a lightweight heuristic-based
    emotion detector instead of crashing.
    """
    if _models:
        return

    for emo in EMOTIONS:
        model_path = MODEL_DIR / f"{emo}_svr_model.joblib"
        vec_path = MODEL_DIR / f"{emo}_svr_vectorizer.joblib"

        # If any required file is missing, or if unpickling fails
        # (for example because sklearn is not installed in this
        # environment), we clear any partially loaded state and
        # return so that the heuristic fallback path is used.
        if not model_path.exists() or not vec_path.exists():
            _models.clear()
            _vectorizers.clear()
            return

        try:
            _models[emo] = joblib.load(model_path)
            _vectorizers[emo] = joblib.load(vec_path)
        except Exception:
            _models.clear()
            _vectorizers.clear()
            return


def detect_emotion(text: str):
    _load_models()

    # If the SVR models are not available, fall back to a simple
    # polarity-based heuristic so the API still returns a valid
    # emotion payload instead of a 500 error.
    if not _models or not _vectorizers:
        polarity = get_polarity(text)

        # Start from a small baseline for all emotions.
        scores = {emo: 0.1 for emo in EMOTIONS}

        if polarity == "positive":
            scores["joy"] = 0.9
        elif polarity == "negative":
            scores["sadness"] = 0.9
        else:
            scores["joy"] = 0.4
            scores["sadness"] = 0.4

        final_scores = suppress_secondary_emotions(scores)
        dominant = max(final_scores, key=final_scores.get)
        intensity = final_scores[dominant]
        severity = intensity_to_severity(intensity)

        return {
            "primary": dominant,
            "severity": severity,
            "intensity": round(intensity, 3),
            "all": {k: round(v, 3) for k, v in final_scores.items()},
        }

    # Full model-based path
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
        "all": {k: round(v, 3) for k, v in final_scores.items()},
    }
