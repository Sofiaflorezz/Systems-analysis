# MoA Prediction - Workshop 3

This project consolidates the design of a Mechanisms of Action (MoA) prediction system by applying **robust engineering** principles and **project management** practices. The goal is to refine the architecture and improve **reliability, scalability, maintainability, and usability**, aligning decisions with quality standards such as **ISO 9000**, **CMMI**, and **Six Sigma**.

### Architecture and Components
The system follows a **modular architecture** with five independent, executable modules—simplifying testing, replacements, and pipeline evolution:
- **Data Loader:** dataset loading and validation (column checks, missing values).
- **Preprocessing:** data cleaning, transformation, and normalization.
- **Model Trainer:** model training and metric logging (Random Forest / simple NN).
- **Prediction:** prediction generation and submission file export.
- **Report Generator:** metric visualization and result summaries.  
Modularity enables swapping models/datasets without impacting the rest of the system, while **versioning** and **reproducibility** are supported via GitHub.

### Quality, Risks, and Mitigation
Key risks were identified and matched with control strategies:
- **Data quality:** EDA, detection of missing values/outliers, normalization, and early alerts.
- **Overfitting:** cross-validation, hyperparameter tuning, early stopping, and model versioning.
- **Interpretability:** **SHAP/LIME** to explain predictions and support biological consistency checks.
- **Pipeline flow:** per-module input/output checks and verification of result persistence.

### Project Management
Clear **roles**, **milestones**, and **deliverables** were defined on a five-week schedule. **Kanban** (Trello/Miro) is used to orchestrate work—improving task visibility, adaptability to changes, blocker detection, and transparent progress tracking. Documentation and **GitHub** version control ensure traceability and collaboration.

### Improvements and Current Status
This phase **refined the architecture**, detailed **mitigation and monitoring** strategies, and strengthened **documentation**. The system is now prepared for subsequent implementation phases and continuous optimization, while maintaining result quality and sustainable development practices.
