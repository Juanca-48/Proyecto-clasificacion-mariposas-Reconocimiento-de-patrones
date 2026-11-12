"""Rutas del proyecto, calculadas desde la ubicación de este archivo (no dependen de dónde se clone el repo).

    dataset/                       fotos originales (Papilionidae/, Pieridae/) y sus CSV de etiquetas
    scripts/                       todos los scripts
    resultados/<etapa>/            la salida de cada script, una subcarpeta por etapa
"""
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_DIR / "dataset"
RESULTS_DIR = REPO_DIR / "resultados"

SEGMENTATION_DIR = RESULTS_DIR / "segmentacion"    # Segmentacion.py, texturas.py, color_features.py, formas.py, prepare_dataset.py
SPLIT_DIR = RESULTS_DIR / "dataset_split"          # Split_dataset.py, Standardize_features.py (standardized/)
RF_DIR = RESULTS_DIR / "rf"                        # Train_random_forest.py
MODELS_DIR = RESULTS_DIR / "modelos"               # Train_models.py
FINAL_DIR = RESULTS_DIR / "final"                  # Train_final.py
