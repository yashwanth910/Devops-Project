"""
prepare_data.py
----------------
Fetches the Breast Cancer Wisconsin (Diagnostic) dataset from scikit-learn
and writes it to data/raw/data.csv so it can be tracked by DVC.

Run:
    python src/prepare_data.py
"""

import os
from sklearn.datasets import load_breast_cancer
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(_PROJECT_ROOT, "data", "raw", "data.csv")


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    dataset = load_breast_cancer(as_frame=True)
    df = dataset.frame  # includes feature columns + 'target' column

    # target: 0 = malignant, 1 = benign (as defined by sklearn)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved dataset with shape {df.shape} to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
