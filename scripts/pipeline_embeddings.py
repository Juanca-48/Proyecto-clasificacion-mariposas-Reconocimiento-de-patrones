#!/usr/bin/env python3
"""
pipeline_embeddings.py
----------------------
Orquesta el segundo modelo de la app: ConvNeXt-Tiny (red preentrenada con ImageNet) sobre los recortes de la
mariposa, con una regresión logística encima. Cada etapa se ejecuta al terminar la anterior:

  1. extraer   redes_preentrenadas/embeddings.py          embeddings de los 812 recortes   -> resultados/embeddings/
  2. evaluar   redes_preentrenadas/evaluar_embeddings.py  CV agrupada por especie y test, contra el modelo clásico
  3. modelo    redes_preentrenadas/entrenar_embeddings.py modelo final y calibración del rechazo
                                                                                          -> resultados/modelo_embeddings/
  4. afinar    fine-tuning completo de ConvNeXt-Tiny (3 semillas)                          -> resultados/modelo_afinado/
               Es lo más lento: en CPU, unas 2 min por época, del orden de 1 hora en total. Para saltarlo: --hasta modelo

Necesita los recortes (preprocesar.py) y la partición train/val/test (entrenar.py, etapa split). Las etapas 1 a 3 usan los
pesos de ImageNet sin cambiarlos; la etapa 4 los ajusta con tus datos y genera un tercer modelo, independiente de ellas.
Este pipeline termina en el modelo; copiarlo a la app es un paso aparte (despliegue/exportar_app.py).

Antes de empezar comprueba las dependencias (torch, torchvision, timm, pillow, opencv-python, numpy, pandas,
scikit-learn, joblib) e instala con pip las que falten. La primera vez descarga los pesos de ConvNeXt-Tiny (~110 MB).

Uso:
    python scripts/pipeline_embeddings.py --estado               (qué etapas ya tienen sus resultados)
    python scripts/pipeline_embeddings.py                        (todas las etapas)
    python scripts/pipeline_embeddings.py --hasta modelo         (sin el fine-tuning, que tarda ~1 hora)
    python scripts/pipeline_embeddings.py --solo afinar          (solo el fine-tuning)
    python scripts/pipeline_embeddings.py --dry-run
"""
from orquestador import correr
from rutas import RESULTS_DIR, SEGMENTATION_DIR, SPLIT_DIR

REQUISITOS = {
    "torch": "torch",
    "torchvision": "torchvision",
    "timm": "timm",
    "PIL": "pillow",
    "cv2": "opencv-python",
    "numpy": "numpy",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "joblib": "joblib",
}

RED, ENTRADA = "convnext", "recorte"
EMB = RESULTS_DIR / "embeddings"
VECTOR_CLASICO = SEGMENTATION_DIR / "features_dataset_ready.csv"   # referencia que usa evaluar_embeddings.py
RECORTES = [SEGMENTATION_DIR / d / "segmented" for d in ("Dataset_papilionidae", "Dataset_pieridae")]
EMBEDDINGS = EMB / f"{RED}__{ENTRADA}.npy"
PARTICION = SPLIT_DIR / "splits.csv"

# nombre, script, argumentos, datos que necesita, datos que produce
STAGES = [
    ("extraer", "redes_preentrenadas/embeddings.py", ["--modelos", RED, "--entradas", ENTRADA],
     RECORTES, [EMBEDDINGS, EMB / "indice.csv"]),
    ("evaluar", "redes_preentrenadas/evaluar_embeddings.py", [],
     [EMBEDDINGS, EMB / "indice.csv", PARTICION, VECTOR_CLASICO], [EMB / "evaluacion_cv.csv"]),
    ("modelo", "redes_preentrenadas/entrenar_embeddings.py", ["--red", RED, "--entrada", ENTRADA],
     [EMBEDDINGS, EMB / "indice.csv", PARTICION], [RESULTS_DIR / "modelo_embeddings" / "modelo.joblib"]),
    ("afinar", "redes_preentrenadas/afinar.py", [], RECORTES + [PARTICION], [RESULTS_DIR / "modelo_afinado" / "modelo_afinado.joblib"]),
]

if __name__ == "__main__":
    correr(__doc__, STAGES, REQUISITOS, "pipeline_embeddings.py",
           cierre=f"Modelo: {RESULTS_DIR / 'modelo_embeddings' / 'modelo.joblib'}\n"
                  "Siguiente: python scripts/despliegue/exportar_app.py --sin-clasico")
