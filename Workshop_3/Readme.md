# MoA Prediction - Workshop 3

## Project overview
This repository contains the code, data processing pipelines and documentation
for the Mechanism of Action (MoA) prediction project (Kaggle challenge).

## Requirements
- Python
- PyTorch or TensorFlow for deep models
- Docker

## Quickstart
1. Clone the repository
2. Prepare data: place the competition files in /data
3. Run preprocessing: `python src/preprocess.py`
4. Train baseline model: `python src/train_baseline.py`
5. Evaluate: `python src/evaluate.py`
6. Generate submission: `python src/generate_submission.py`

## Structure
- /data : raw and processed datasets (not included)
- /src : preprocessing, modeling, and evaluation scripts
- /notebooks : exploratory analysis and experiments
- /reports : final LaTeX report and figures

## Reproducibility
Use Docker to replicate the environment:
`docker build -t moa-env .`
`docker run --rm -v $(pwd):/work moa-env bash -c "python src/preprocess.py"`
