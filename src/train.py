"""
train.py
--------
Complete training pipeline for the Breast Cancer classification task.

Steps:
  1. Load data (data loading)
  2. Preprocess (train/test split + feature scaling)
  3. Train three different models
  4. Evaluate each on held-out test data
  5. Log params/metrics/artifacts to MLflow for every run
  6. Pick the best model (by F1 score) and register it in the
     MLflow Model Registry
  7. Export the best model + scaler locally to models/ so the
     FastAPI service / Docker image can load them without needing
     a live MLflow server at inference time.

Run:
    python src/train.py
"""

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from utils import load_dataset, split_features_target, save_artifacts

EXPERIMENT_NAME = "breast-cancer-classification"
REGISTERED_MODEL_NAME = "breast-cancer-best-model"
RANDOM_STATE = 42

MODELS = {
    "logistic_regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
}


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1_score": f1_score(y_test, preds),
    }
    if probs is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, probs)
    return metrics


def main():
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_dataset()
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    best_run = {"name": None, "model": None, "f1": -1.0, "metrics": None}

    for name, model in MODELS.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train_scaled, y_train)
            metrics = evaluate(model, X_test_scaled, y_test)

            # Log params
            mlflow.log_params(model.get_params())
            mlflow.log_param("model_type", name)

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log the model artifact for this run
            mlflow.sklearn.log_model(model, artifact_path="model")

            print(f"[{name}] " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

            if metrics["f1_score"] > best_run["f1"]:
                best_run.update(
                    name=name, model=model, f1=metrics["f1_score"], metrics=metrics
                )

    print(f"\nBest model: {best_run['name']} (f1_score={best_run['f1']:.4f})")

    # Register the best model in a dedicated MLflow run so it's clearly
    # tagged as the production candidate in the Model Registry.
    with mlflow.start_run(run_name=f"best-{best_run['name']}"):
        mlflow.log_param("model_type", best_run["name"])
        mlflow.log_metrics(best_run["metrics"])
        model_info = mlflow.sklearn.log_model(
            best_run["model"],
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )
        print(f"Registered model URI: {model_info.model_uri}")

    # Export locally for the FastAPI service / Docker image.
    save_artifacts(best_run["model"], scaler, X.columns)
    print("Saved best model + scaler to models/")


if __name__ == "__main__":
    main()
