# prediction.py
"""
Result Generation Module
genera el archivo submission.csv con las probabilidades
para cada MoA, usando el modelo entrenado.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from data_ingestion import load_raw_data
from modeling import build_model, train_model, load_model, MODEL_PATH
from processing import filter_controls, prepare_features_and_targets

SUBMISSION_PATH = Path("submission_moa_rf.csv")


def generate_submission() -> Path:
    """
    Entrena el modelo con todos los datos (sin controles) y
    genera un archivo submission_moa_rf.csv.
    """
    print("Cargando datos...")
    train_features, train_targets, test_features = load_raw_data()
    train_features, train_targets = filter_controls(train_features, train_targets)

    X, y, X_test, sig_id_test, target_cols = prepare_features_and_targets(
        train_features, train_targets, test_features
    )

    print("Cargando o construyendo modelo...")
    if MODEL_PATH.exists():
        model: Pipeline = load_model(MODEL_PATH)
        print("Modelo cargado desde disco.")
    else:
        model = build_model(X)
        model = train_model(model, X, y)

    print("Entrenando modelo con todos los datos disponibles...")
    model.fit(X, y)

    print("Generando predicciones para test...")
    probas_list = model.predict_proba(X_test)
    probas_test = np.stack([p[:, 1] for p in probas_list], axis=1)

    submission = pd.DataFrame(probas_test, columns=target_cols)
    submission.insert(0, "sig_id", sig_id_test)

    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission guardada en {SUBMISSION_PATH.resolve()}")

    return SUBMISSION_PATH
