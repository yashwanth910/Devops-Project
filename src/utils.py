"""
utils.py
--------
Shared helper functions used by train.py, predict.py, and app.py.
"""

import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Resolve paths relative to the project root (parent of this file's
# directory) so these work regardless of the current working directory
# the script/app is launched from (e.g. `python src/train.py` from repo
# root, or `uvicorn app:app` from inside src/, or inside Docker).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(_PROJECT_ROOT, "data", "raw", "data.csv")
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "feature_names.pkl")


def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the dataset produced by prepare_data.py"""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `python src/prepare_data.py` "
            "or `dvc repro` first."
        )
    return pd.read_csv(path)


def split_features_target(df: pd.DataFrame, target_col: str = "target"):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def save_artifacts(model, scaler: StandardScaler, feature_names) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(list(feature_names), FEATURES_PATH)


def load_artifacts():
    """Load the locally saved model, scaler, and feature name list.

    These are the artifacts copied into the Docker image so the API
    doesn't need a live connection to the MLflow tracking server at
    inference time.
    """
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
        raise FileNotFoundError(
            "Model artifacts not found. Run `python src/train.py` first "
            "to train and export the best model."
        )
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    return model, scaler, feature_names
