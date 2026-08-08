"""
predict.py
----------
Loads the exported best model + scaler and exposes a simple
predict() function used by app.py.
"""

import numpy as np
from utils import load_artifacts

_model, _scaler, _feature_names = None, None, None


def _ensure_loaded():
    global _model, _scaler, _feature_names
    if _model is None:
        _model, _scaler, _feature_names = load_artifacts()
    return _model, _scaler, _feature_names


def get_feature_names():
    _, _, feature_names = _ensure_loaded()
    return feature_names


def predict(features: dict) -> dict:
    """
    features: dict mapping feature name -> value, matching the
              dataset's feature columns.
    """
    model, scaler, feature_names = _ensure_loaded()

    missing = [f for f in feature_names if f not in features]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    ordered_values = [features[f] for f in feature_names]
    X = np.array(ordered_values).reshape(1, -1)
    X_scaled = scaler.transform(X)

    pred = int(model.predict(X_scaled)[0])
    result = {"prediction": pred, "label": "benign" if pred == 1 else "malignant"}

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)[0]
        result["probability"] = {
            "malignant": float(proba[0]),
            "benign": float(proba[1]),
        }
    return result
