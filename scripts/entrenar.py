#!/usr/bin/env python3
"""
entrenar.py
-----------
Ejecuta en orden lo que sigue al preprocesamiento (preprocesar.py): la división, el entrenamiento
y el modelo final, para el vector sin forma y para el vector con forma.

  Vector sin forma (resultados/segmentacion/features_dataset_ready.csv):
  1. split               Split_dataset.py          división 70/15/15 por especie  -> resultados/dataset_split/
  2. rf                  Train_random_forest.py    línea base                     -> resultados/rf/
  3. estandarizar        Standardize_features.py   z-score ajustado con train     -> resultados/dataset_split/standardized/
  4. modelos             Train_models.py           compara 4 modelos              -> resultados/modelos/
  5. final               Train_final.py            modelo final                   -> resultados/final/

  Vector con forma (resultados/features_forma/features_dataset_ready.csv):
  6. split_forma         misma división sobre el vector con forma                 -> resultados/dataset_split_forma/
  7. estandarizar_forma
  8. modelos_forma                                                                -> resultados/modelos_forma/
  9. final_forma                                                                  -> resultados/final_forma/

La etapa `split` usa --overwrite: vuelve a copiar las imágenes de resultados/dataset_split/segmented/
(la división es determinista, con semilla 42, y da las mismas particiones).
Las etapas sobre el vector con forma usan la misma partición que el vector sin forma, sin copiar imágenes.

Antes de empezar comprueba las dependencias (numpy, pandas, scikit-learn, joblib, matplotlib) e
instala con pip las que falten. Si una etapa falla se detiene. Antes de ejecutar valida que existan
los datos que cada etapa necesita; si falta algo, dice qué etapa lo produce. Los datos de entrada se
generan con preprocesar.py.

Uso:
    python scripts/entrenar.py --estado                   (qué etapas ya tienen sus resultados)
    python scripts/entrenar.py                            (todas las etapas)
    python scripts/entrenar.py --hasta final              (solo el vector sin forma)
    python scripts/entrenar.py --desde estandarizar       (sin repetir la división ni el Random Forest)
    python scripts/entrenar.py --solo modelos final
    python scripts/entrenar.py --omitir rf
    python scripts/entrenar.py --dry-run                  (solo muestra los comandos)
"""
from orquestador import correr
from rutas import FINAL_DIR, MODELS_DIR, RESULTS_DIR, RF_DIR, SEGMENTATION_DIR, SPLIT_DIR

REQUISITOS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "joblib": "joblib",
    "matplotlib": "matplotlib",
}

VECTOR = SEGMENTATION_DIR / "features_dataset_ready.csv"
VECTOR_FORMA = RESULTS_DIR / "features_forma" / "features_dataset_ready.csv"
SPLIT_FORMA = RESULTS_DIR / "dataset_split_forma"
MODELOS_FORMA = RESULTS_DIR / "modelos_forma"
FINAL_FORMA = RESULTS_DIR / "final_forma"


def particion(d):                       # lo que produce Split_dataset.py
    return [d / f"features_{s}.csv" for s in ("train", "val", "test")] + [d / "splits.csv"]


def estandarizados(d):                  # lo que produce Standardize_features.py
    return [d / "standardized" / f"features_{s}_std.csv" for s in ("train", "val", "test")] + [d / "standardized" / "scaler.joblib"]


FOTOS_SEGMENTADAS = [SEGMENTATION_DIR / d / "segmented" for d in ("Dataset_papilionidae", "Dataset_pieridae")]

# nombre, script, argumentos, datos que necesita, datos que produce
STAGES = [
    ("split", "Split_dataset.py", ["--overwrite"], [VECTOR] + FOTOS_SEGMENTADAS, particion(SPLIT_DIR)),
    ("rf", "Train_random_forest.py", [], particion(SPLIT_DIR), [RF_DIR / "metrics_report.txt"]),
    ("estandarizar", "Standardize_features.py", [], particion(SPLIT_DIR), estandarizados(SPLIT_DIR)),
    ("modelos", "Train_models.py", [], estandarizados(SPLIT_DIR), [MODELS_DIR / "metrics_report.txt"]),
    ("final", "Train_final.py", [], particion(SPLIT_DIR), [FINAL_DIR / "final_model.joblib"]),

    ("split_forma", "Split_dataset.py",
     ["--features-csv", str(VECTOR_FORMA), "--output", str(SPLIT_FORMA), "--no-copy"],
     [VECTOR_FORMA] + FOTOS_SEGMENTADAS, particion(SPLIT_FORMA)),
    ("estandarizar_forma", "Standardize_features.py", ["--split-dir", str(SPLIT_FORMA)],
     particion(SPLIT_FORMA), estandarizados(SPLIT_FORMA)),
    ("modelos_forma", "Train_models.py",
     ["--std-dir", str(SPLIT_FORMA / "standardized"), "--output", str(MODELOS_FORMA)],
     estandarizados(SPLIT_FORMA), [MODELOS_FORMA / "metrics_report.txt"]),
    ("final_forma", "Train_final.py", ["--split-dir", str(SPLIT_FORMA), "--output", str(FINAL_FORMA)],
     particion(SPLIT_FORMA), [FINAL_FORMA / "final_model.joblib"]),
]

if __name__ == "__main__":
    correr(__doc__, STAGES, REQUISITOS, "entrenar.py",
           cierre=f"Modelo final (sin forma): {FINAL_DIR / 'final_model.joblib'}\n"
                  f"Modelo final (con forma): {FINAL_FORMA / 'final_model.joblib'}")
