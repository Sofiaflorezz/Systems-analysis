# modeling.py
"""
Model Construction Module
Define el pipeline de ML (preprocesador + RandomForest multi-etiqueta),
inspirado en notebooks de Kaggle para MoA.

"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

from processing import build_preprocessor

MODEL_PATH = Path("moa_random_forest.joblib")
RANDOM_STATE = 42


def build_model(X_sample: pd.DataFrame) -> Pipeline:
    """
    Construye el pipeline completo:
      preprocessor (ColumnTransformer) +
      classifier (MultiOutputClassifier(RandomForestClassifier))
    """
    preprocessor = build_preprocessor(X_sample)

    base_rf = RandomForestClassifier(
        n_estimators=10,  
        max_depth=4,      
        n_jobs=-1,
        random_state=RANDOM_STATE,
        oob_score=False,
    )


    classifier = MultiOutputClassifier(base_rf, n_jobs=-1)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    return model


def train_model(model: Pipeline, X_train: pd.DataFrame, y_train: pd.DataFrame) -> Pipeline:
    """
    Entrena el modelo con los datos de entrenamiento.
    """
    model.fit(X_train, y_train)
    return model


def save_model(model: Pipeline, path: Path = MODEL_PATH) -> None:
    """
    Guarda el modelo entrenado en disco.
    """
    joblib.dump(model, path)


def load_model(path: Path = MODEL_PATH) -> Pipeline:
    """
    Carga un modelo entrenado desde disco.
    """
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el modelo en {path}")
    return joblib.load(path)
