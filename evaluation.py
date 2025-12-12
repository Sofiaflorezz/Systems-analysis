# evaluation.py
"""
Evaluation & Simulation Module

Incluye:
- evaluación del modelo (log_loss)
- simulación de ruido
- Escenario 1: 30 corridas (15 baseline, 15 perturbadas sobre cp_time),
  con cálculo de Media, Desviación Estándar y Coefficient of Variation.
"""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from modeling import build_model, train_model
from processing import CAT_COLS

RANDOM_STATE = 42
TEST_SIZE = 0.2


from sklearn.metrics import log_loss
import numpy as np


def evaluate_model(model, X_valid, y_valid):
    """
    Evalúa el modelo MultiOutputClassifier.

    Maneja el caso en que alguna de las salidas solo tenga una clase
    (predict_proba devuelve shape (n_samples, 1) en vez de (n_samples, 2)).
    En ese caso, se asume probabilidad 0 para la clase positiva.
    """
    probas_list = model.predict_proba(X_valid)  # lista de arrays (n_samples, n_classes)

    cols = []
    for i, p in enumerate(probas_list):
        # p: (n_samples, n_classes)
        if p.shape[1] == 2:
            # Caso normal: [P(clase0), P(clase1)]
            cols.append(p[:, 1])
        else:
            # Solo hay una clase en el entrenamiento de esa etiqueta.
            # Asumimos probabilidad 0 de clase positiva.
            cols.append(np.zeros(p.shape[0], dtype=float))

    probas = np.stack(cols, axis=1)

    # y_valid suele ser un DataFrame; usamos los valores como matriz
    y_true = y_valid.values if hasattr(y_valid, "values") else y_valid
    loss = log_loss(y_true, probas)

    return probas, loss



def simulate_noise_scenario(
    model: Pipeline,
    X_valid: pd.DataFrame,
    y_valid: pd.DataFrame,
    base_loss: float,
    noise_level: float = 0.05,
) -> Tuple[float, float]:
    """
    Escenario sencillo: añade ruido a las features numéricas y compara el log_loss.
    (Versión simple de "caos" local en las features).
    """
    print("\n--- Escenario ruido gaussiano (simple) ---")
    print(f"Error Normal: {base_loss:.5f}")

    X_noisy = X_valid.copy()
    num_cols = [c for c in X_noisy.columns if c not in CAT_COLS]

    rng = np.random.default_rng(RANDOM_STATE)
    noise = rng.normal(loc=0.0, scale=noise_level, size=X_noisy[num_cols].shape)
    X_noisy[num_cols] = X_noisy[num_cols] + noise

    probas_list = model.predict_proba(X_noisy)
    probas_caos = np.stack([p[:, 1] for p in probas_list], axis=1)

    loss_caos = log_loss(y_valid.values.ravel(), probas_caos.ravel(), labels=[0, 1])
    aumento = (loss_caos - base_loss) / base_loss * 100

    print(f"Error Caos : {loss_caos:.5f}")
    print(f"AUMENTO DEL ERROR: {aumento:.2f}%")

    return loss_caos, aumento


# -------------------------------------------------------------------
# Scenario 1 - 30 corridas con perturbación de cp_time (Workshop 4)
# -------------------------------------------------------------------


def _build_dataset_perturbed_cp_time(
    X: pd.DataFrame, y: pd.DataFrame, rng: np.random.Generator
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construye un dataset "perturbado" alterando la distribución de cp_time:
      - Aumenta en ~15% la proporción de cp_time = 24
      - Disminuye en ~10% la proporción de cp_time = 72
    Esto simula el 'cp_time distribution shift' descrito en el Workshop 4.
    """

    # Índices por cp_time
    idx_24 = X.index[X["cp_time"] == 24]
    idx_48 = X.index[X["cp_time"] == 48]
    idx_72 = X.index[X["cp_time"] == 72]
    idx_other = X.index[
        ~X.index.isin(idx_24.union(idx_48).union(idx_72))
    ] 

    # Oversample 24h (+15%)
    n_24 = len(idx_24)
    n_24_new = int(n_24 * 1.15)
    if n_24 > 0:
        sampled_24 = rng.choice(idx_24, size=n_24_new, replace=True)
    else:
        sampled_24 = np.array([], dtype=int)

    # Downsample 72h (-10%)
    n_72 = len(idx_72)
    n_72_new = int(n_72 * 0.9)
    if n_72_new > 0:
        sampled_72 = rng.choice(idx_72, size=n_72_new, replace=False)
    else:
        sampled_72 = np.array([], dtype=int)

    # 48h + otros quedan igual
    sampled_48 = idx_48.to_numpy()
    sampled_other = idx_other.to_numpy()

    final_idx = np.concatenate([sampled_24, sampled_48, sampled_72, sampled_other])
    rng.shuffle(final_idx)

    Xp = X.loc[final_idx].reset_index(drop=True)
    yp = y.loc[final_idx].reset_index(drop=True)
    return Xp, yp


def run_scenario1_simulation(
    X: pd.DataFrame,
    y: pd.DataFrame,
    n_baseline: int = 15,
    n_perturbed: int = 15,
) -> pd.DataFrame:
    """
    Implementa el Scenario 1 del Workshop 4:

    - 15 corridas "baseline" usando la distribución original de cp_time.
    - 15 corridas "perturbadas" con shift en cp_time (según _build_dataset_perturbed_cp_time).
    - En cada corrida:
        * hace un split train/valid
        * entrena un modelo RandomForest multi-etiqueta
        * calcula el log_loss en valid
    - Devuelve una tabla con Media, Desviación Estándar y COV por grupo.

    """

    rng = np.random.default_rng(RANDOM_STATE)

    results = []  # lista de dicts {group, run, log_loss}

    # --- Grupo baseline ---
    print("\n=== Scenario 1: Grupo BASELINE (distribución original cp_time) ===")
    for r in range(n_baseline):
        # Split distinto por corrida
        X_train, X_valid, y_train, y_valid = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=int(rng.integers(0, 1_000_000)),
        )

        model = build_model(X)
        model = train_model(model, X_train, y_train)
        _, loss = evaluate_model(model, X_valid, y_valid)
        print(f"Baseline run {r+1:02d} - log_loss: {loss:.5f}")

        results.append({"group": "baseline", "run": r + 1, "log_loss": loss})

    # --- Grupo perturbado ---
    print("\n=== Scenario 1: Grupo PERTURBADO (shift cp_time) ===")
    for r in range(n_perturbed):
        Xp, yp = _build_dataset_perturbed_cp_time(X, y, rng)

        X_train, X_valid, y_train, y_valid = train_test_split(
            Xp,
            yp,
            test_size=TEST_SIZE,
            random_state=int(rng.integers(0, 1_000_000)),
        )

        model = build_model(Xp)
        model = train_model(model, X_train, y_train)
        _, loss = evaluate_model(model, X_valid, y_valid)
        print(f"Perturbed run {r+1:02d} - log_loss: {loss:.5f}")

        results.append({"group": "perturbed", "run": r + 1, "log_loss": loss})

    df_results = pd.DataFrame(results)

    
    summary = (
        df_results.groupby("group")["log_loss"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "mu", "std": "sigma"})
    )
    summary["cov_percent"] = (summary["sigma"] / summary["mu"]) * 100.0

    print("\n=== Scenario 1 - Resumen (Media, Sigma, COV%) ===")
    print(summary)

    
    summary.to_csv("scenario1_logloss_summary.csv")

    return df_results
