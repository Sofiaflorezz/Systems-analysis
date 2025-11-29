# Workshop 4 – Computational Simulation Report  
**Analysis of Chaos and Non-Linear Dynamics in the MoA Prediction System**

This part of the project focuses on the computational simulation stage, where we analyze chaotic and non-linear behavior in a system for predicting Mechanisms of Action (MoA) from high-dimensional biological data (Kaggle challenge).



## Project Overview

The main goal of this workshop is to validate a previously proposed architecture by:

- Studying the sensitivity of the system to distribution shifts in the input data, especially the metadata parameter `cp_time`.
- Identifying potential non-linear and chaotic behaviors that affect model performance and stability.
- Proposing architectural improvements to make the MoA prediction system more robust and trustworthy.



## Data and Preprocessing

The pipeline includes:

- Filtering control samples and preparing the training dataset.
- **Standardizing features** by type (gene expression, cell features, metadata, etc.).
- **Dimensionality reduction** with PCA to mitigate noise and strong correlations.
- Designing controlled perturbation scenarios that explicitly modify the distribution of `cp_time`.



## Simulation Scenarios

### Scenario 1 – Machine Learning Model (Random Forest)

- A multi-output Random Forest model is trained on the preprocessed dataset.
- The distribution of `cp_time` is systematically perturbed.
- Performance stability is evaluated using:
  - **Mean log-loss**
  - **Standard deviation**
  - **Coefficient of Variation (CoV)**

This scenario quantifies how stable the model is when exposed to subtle but structured changes in the input distribution.

### Scenario 2 – Cellular Automaton (CA)

- A 2D cellular automaton is implemented to model the propagation of local biological signals.
- The system is analyzed for the presence of a bifurcation point, where the dynamics transition:
  - From stable and predictable  
  - To a regime of spatiotemporal chaos, highly sensitive to control parameters derived from metadata.



## Key Findings

- The MoA prediction system exhibits high sensitivity to distribution shifts in `cp_time`, reflected in increased CoV and log-loss variability under perturbations.
- The cellular automaton experiments reveal a parameter threshold where the dynamics become chaotic, supporting the idea that the overall system can enter regions of complex, hard-to-control behavior.



## Architectural Recommendations

To make the MoA prediction system more resilient, the report proposes:

1. **Active Distribution-Validation Module**  
   - Continuously monitor input distributions.
   - Detect significant shifts before prediction and trigger alerts or recalibration strategies.

2. **Stratified Mixture-of-Experts Architecture (by `cp_time`)**  
   - Train specialized models for different `cp_time` ranges.
   - Combine their outputs using a mixture-of-experts approach driven by metadata.

3. **Bayesian-style Uncertainty Estimation (e.g., MC Dropout)**  
   - Attach an explicit uncertainty estimate to each prediction.
   - Enable more informed decision-making in regions where the model may behave chaotically.

