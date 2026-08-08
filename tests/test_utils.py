"""
test_utils.py
-------------
Basic sanity tests for the data pipeline that don't require a trained
model to be present. These are the tests GitHub Actions runs on every
push, independent of whether model artifacts exist in the runner.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sklearn.datasets import load_breast_cancer
from utils import split_features_target


def test_dataset_loads():
    df = load_breast_cancer(as_frame=True).frame
    assert df.shape[0] > 0
    assert "target" in df.columns


def test_split_features_target():
    df = load_breast_cancer(as_frame=True).frame
    X, y = split_features_target(df)
    assert "target" not in X.columns
    assert len(X) == len(y)
    assert set(y.unique()).issubset({0, 1})
