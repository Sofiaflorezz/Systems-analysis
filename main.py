# main.py
"""
Main Orchestrator

Integra:
- Workshop 2: Arquitectura modular (Data Ingestion, Processing, Modeling, Evaluation, Result Generation)
- Workshop 4: Scenario 1 (30 corridas con shift en cp_time) y Scenario 2 (Cellular Automata)
"""

from data_ingestion import load_raw_data
from processing import (
    filter_controls,
    prepare_features_and_targets,
    split_train_valid,
)
from modeling import build_model, train_model, save_model
from evaluation import evaluate_model, simulate_noise_scenario, run_scenario1_simulation
from prediction import generate_submission
from ca_simulation import run_ca_experiments
from ca_simulation import save_ca_snapshot



def run_pipeline():
    # 1. Data Ingestion
    print("=== 1. DATA INGESTION ===")
    train_features, train_targets, test_features = load_raw_data()

    # Filtrado de controles (cp_type = 'ctl_vehicle')
    train_features, train_targets = filter_controls(train_features, train_targets)

    # 2. Processing & Feature Engineering
    print("=== 2. PROCESSING & FEATURE ENGINEERING ===")
    X, y, X_test, sig_id_test, target_cols = prepare_features_and_targets(
        train_features, train_targets, test_features
    )
    X_train, X_valid, y_train, y_valid = split_train_valid(X, y)

    # 3. Model Construction & Training
    print("=== 3. MODEL CONSTRUCTION & TRAINING ===")
    model = build_model(X)
    model = train_model(model, X_train, y_train)
    save_model(model)

    # 4. Evaluation (single run) + ruido simple
    print("=== 4. EVALUATION (single run) ===")
    _, loss = evaluate_model(model, X_valid, y_valid)
    print(f"Log loss en validación (single run): {loss:.5f}")

    simulate_noise_scenario(model, X_valid, y_valid, loss, noise_level=0.05)

    # 4b. Scenario 1 - 30 corridas (baseline vs perturbed cp_time)
    print("=== 4b. SCENARIO 1 (30 corridas, cp_time perturbado) ===")
    run_scenario1_simulation(X, y, n_baseline=2, n_perturbed=2)

    # 5. Result Generation (submission para Kaggle)
    print("=== 5. RESULT GENERATION (SUBMISSION) ===")
    generate_submission()

    # 6. Scenario 2 - Cellular Automata
    print("=== 6. SCENARIO 2 (Cellular Automata) ===")
    run_ca_experiments()

    save_ca_snapshot(0.9)

    print("=== PIPELINE COMPLETO FINALIZADO ===")



if __name__ == "__main__":
    run_pipeline()
