"""
test_api.py
-----------
Tests for the FastAPI service. These require trained model artifacts
to exist in models/ (produced by `python src/train.py`). The CI
workflow runs data prep + training before pytest for this reason.
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

MODEL_PRESENT = os.path.exists(os.path.join("models", "best_model.pkl"))

if MODEL_PRESENT:
    from app import app

    client = TestClient(app)


def test_health():
    if not MODEL_PRESENT:
        pytest.skip("model artifacts not present; run src/train.py first")
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_features():
    if not MODEL_PRESENT:
        pytest.skip("model artifacts not present; run src/train.py first")
    resp = client.get("/features")
    assert resp.status_code == 200
    assert len(resp.json()["required_features"]) == 30


def test_predict_valid_input():
    if not MODEL_PRESENT:
        pytest.skip("model artifacts not present; run src/train.py first")
    from predict import get_feature_names

    sample = {name: 1.0 for name in get_feature_names()}
    resp = client.post("/predict", json={"features": sample})
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] in (0, 1)
    assert body["label"] in ("malignant", "benign")


def test_predict_missing_feature():
    if not MODEL_PRESENT:
        pytest.skip("model artifacts not present; run src/train.py first")
    resp = client.post("/predict", json={"features": {"mean radius": 14.0}})
    assert resp.status_code == 400
