# Breast Cancer Prediction — MLOps Pipeline

An end-to-end MLOps capstone project: data versioning (DVC) → experiment
tracking & model registry (MLflow) → prediction API (FastAPI) →
containerization (Docker) → CI/CD (GitHub Actions).

**Dataset:** Breast Cancer Wisconsin (Diagnostic) — a built-in scikit-learn
classification dataset (30 numeric features, binary target: malignant/benign).
No external download needed, which keeps the pipeline fully reproducible.

## Project Structure

```
project/
├── data/raw/              # DVC-tracked dataset (generated, not committed to git)
├── models/                # Exported best model + scaler (generated)
├── src/
│   ├── prepare_data.py    # Fetches & saves the dataset
│   ├── utils.py           # Shared data/model helper functions
│   ├── train.py           # Trains 3 models, logs to MLflow, registers best
│   ├── predict.py         # Inference helper used by the API
│   └── app.py              # FastAPI service
├── tests/                 # pytest test suite
├── .github/workflows/ci.yml
├── Dockerfile
├── requirements.txt
├── dvc.yaml
└── README.md
```

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Data Versioning with DVC

```bash
git init
dvc init

python src/prepare_data.py          # creates data/raw/data.csv
dvc add data/raw/data.csv           # start tracking the dataset with DVC
git add data/raw/data.csv.dvc .gitignore .dvc
git commit -m "Track dataset with DVC"
```

To reproduce the full pipeline (data prep + training) as DVC stages instead
of running scripts manually:

```bash
dvc repro
```

This uses `dvc.yaml`, which defines `prepare_data` and `train` as connected,
cacheable stages — re-running `dvc repro` only re-executes stages whose
dependencies changed.

(Optional) Add a remote to push versioned data:
```bash
dvc remote add -d storage <your-remote-url>
dvc push
```

## 3. Train Models & Track with MLflow

```bash
mlflow ui   # in a separate terminal, then open http://127.0.0.1:5000
```

```bash
python src/train.py
```

This trains three models — **Logistic Regression**, **Random Forest**, and
**Gradient Boosting** — and for each run logs:
- Parameters (all hyperparameters via `model.get_params()`)
- Metrics: accuracy, precision, recall, F1 score, ROC-AUC
- The trained model artifact

It then compares all runs by F1 score, logs the winner in a dedicated
`best-<model>` run, and **registers it in the MLflow Model Registry** under
the name `breast-cancer-best-model`. The same best model + scaler are also
exported to `models/` so the API can load them directly (no live MLflow
server required at inference time — this is what gets packaged into Docker).

Open the MLflow UI to compare experiments and confirm the registered model.

## 4. Run the Prediction API

```bash
cd src
uvicorn app:app --host 0.0.0.0 --port 8000
```

- `GET /health` — liveness check
- `GET /features` — lists the 30 required feature names
- `POST /predict` — run inference

Example request:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"mean radius": 14.0, "mean texture": 20.0, ...}}'
```
(Use `GET /features` to get the full list of 30 keys required.)

Interactive docs: `http://localhost:8000/docs`

## 5. Docker

```bash
docker build -t breast-cancer-api .
docker run -p 8000:8000 breast-cancer-api
```

The image bundles the FastAPI app, dependencies, and the exported
model/scaler from `models/`, so it serves predictions immediately on start
with no external dependencies.

## 6. Tests

```bash
pytest tests/ -v
```

`tests/test_utils.py` checks the data pipeline; `tests/test_api.py` checks
the API endpoints (these require `models/*.pkl` to exist — run
`python src/train.py` first, or let CI do it automatically).

## 7. CI/CD — GitHub Actions

`.github/workflows/ci.yml` runs automatically on every push/PR to `main`:
1. Checkout repository
2. Set up Python 3.11
3. Install dependencies
4. Prepare data + train model (so tests have real artifacts)
5. Run `pytest`
6. Build the Docker image

Check the **Actions** tab in GitHub after pushing to see the run.

## Submission Checklist

- [ ] GitHub repository link (push this project)
- [ ] Screenshot: MLflow experiments comparison (`mlflow ui`)
- [ ] Screenshot: registered model in MLflow Model Registry
- [ ] Screenshot: `dvc status` / `dvc add` showing successful tracking
- [ ] Screenshot: successful GitHub Actions run (green checkmarks)
- [ ] Screenshot: `POST /predict` returning a prediction (e.g. via `/docs` or curl)
