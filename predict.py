import sys
import time
import os

import joblib
from features import extract_features

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_final.joblib")
_model = None


def _get_model():
    global _model
    if _model is None:
        _model = joblib.load(_MODEL_PATH)
    return _model


def predict(image_path: str) -> float:
    feats = extract_features(image_path).reshape(1, -1)
    return float(_get_model().predict_proba(feats)[0, 1])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py some_image.jpg", file=sys.stderr)
        sys.exit(1)

    t0 = time.perf_counter()
    score = predict(sys.argv[1])
    elapsed_ms = (time.perf_counter() - t0) * 1000

    print(round(score, 4))
    print(f"[latency: {elapsed_ms:.0f} ms]", file=sys.stderr)
