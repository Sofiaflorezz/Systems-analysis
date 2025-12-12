# data_ingestion.py
"""
Data Ingestion Module
Corresponde al módulo 1 (Data Ingestion) del Workshop 2:
carga los datos crudos de la competencia MoA.
"""

from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")


def load_raw_data():
    """
    Carga los tres archivos principales de la competencia MoA:
      - train_features.csv
      - train_targets_scored.csv
      - test_features.csv

    Retorna:
        train_features (DataFrame)
        train_targets (DataFrame)
        test_features (DataFrame)
    """
    train_features = pd.read_csv(DATA_DIR / "train_features.csv")
    train_targets = pd.read_csv(DATA_DIR / "train_targets_scored.csv")
    test_features = pd.read_csv(DATA_DIR / "test_features.csv")

    # Validación básica
    assert "sig_id" in train_features.columns
    assert "sig_id" in train_targets.columns
    assert "sig_id" in test_features.columns

    return train_features, train_targets, test_features
