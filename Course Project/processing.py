# processing.py
"""
Processing & Feature Engineering Module
- Filtra controles
- Prepara X, y y test
- Construye el preprocesador (StandardScaler + PCA para numéricas, OneHot para categóricas)
"""

from typing import Tuple, List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.2

CAT_COLS = ["cp_type", "cp_time", "cp_dose"]


def filter_controls(
    train_features: pd.DataFrame, train_targets: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Elimina las filas con cp_type = 'ctl_vehicle' (controles),
    tal como se menciona en Workshop 4 (data cleaning).
    """
    mask = train_features["cp_type"] != "ctl_vehicle"
    train_features_f = train_features.loc[mask].reset_index(drop=True)
    train_targets_f = train_targets.loc[mask].reset_index(drop=True)
    return train_features_f, train_targets_f


def prepare_features_and_targets(
    train_features: pd.DataFrame,
    train_targets: pd.DataFrame,
    test_features: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, List[str]]:
    """
    - Ordena por sig_id
    - Quita sig_id de features
    - Construye X, y y X_test
    - Retorna lista de columnas objetivo y sig_id de test
    """
    train_features = train_features.sort_values("sig_id").reset_index(drop=True)
    train_targets = train_targets.sort_values("sig_id").reset_index(drop=True)
    test_features = test_features.sort_values("sig_id").reset_index(drop=True)

    sig_id_test = test_features["sig_id"].copy()

    X = train_features.drop(columns=["sig_id"])
    X_test = test_features.drop(columns=["sig_id"])

    y = train_targets.drop(columns=["sig_id"])
    target_cols = list(y.columns)

    return X, y, X_test, sig_id_test, target_cols


def split_train_valid(
    X: pd.DataFrame, y: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split sencillo train/valid. Para el escenario 1 se realizan varias corridas.
    """
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    return X_train, X_valid, y_train, y_valid


def build_preprocessor(X_sample: pd.DataFrame) -> ColumnTransformer:
    """
    Construye el preprocesador:
      - Numéricas (g-*, c-*): StandardScaler + PCA (reducción a ~280 componentes)
      - Categóricas: OneHotEncoder (cp_type, cp_time, cp_dose)
    Esto sigue la idea de Workshop 4 (estandarización + PCA).
    """
    num_cols = [c for c in X_sample.columns if c not in CAT_COLS]

    numeric_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=280)),
        ]
    )

    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_transformer, CAT_COLS),
        ]
    )

    return preprocessor
