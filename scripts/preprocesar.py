#!/usr/bin/env python3
"""
preprocesar.py
--------------
Ejecuta el preprocesamiento en orden, cada etapa al terminar la anterior:

  1. segmentacion    Segmentacion_rembg.py  máscaras y recortes (rembg)  -> resultados/segmentacion/
  2. texturas        texturas.py          GLCM, Haralick, Gabor
  3. color           color_features.py    histogramas y momentos HSV/BGR
  4. formas          formas.py            descriptores de forma
  5. prepare         prepare_dataset.py   vector sin forma              -> resultados/segmentacion/features_dataset_ready.csv
  6. prepare_forma   prepare_dataset.py   vector con forma              -> resultados/features_forma/

El método anterior (GrabCut, Segmentacion.py) ya no está en este flujo; sigue disponible aparte, pero escribe en las
mismas carpetas. La primera vez, rembg descarga su modelo (176 MB).

Antes de empezar comprueba las dependencias (opencv-python, numpy, pandas, scikit-image,
mahotas, matplotlib, rembg, onnxruntime) e instala con pip las que falten. Con --dry-run solo las lista.

Cada etapa usa los valores por defecto de su script. Si una falla, se detiene y no ejecuta las
siguientes (así no se mezclan features de una corrida con máscaras de otra).
Antes de ejecutar valida que existan los datos que cada etapa necesita: lo que no se vaya a
generar en esta misma corrida debe estar ya en disco, y si falta se avisa qué etapa lo produce.
Termina en el vector de features: la división, el entrenamiento y lo que sigue están en entrenar.py.

La segmentación (812 imágenes) y las texturas tardan varios minutos. Para no repetirlas:

Uso:
    python scripts/preprocesar.py --estado                (qué etapas ya tienen sus resultados)
    python scripts/preprocesar.py                         (todas las etapas)
    python scripts/preprocesar.py --desde color           (de color en adelante)
    python scripts/preprocesar.py --desde color --hasta formas
    python scripts/preprocesar.py --solo formas prepare   (solo esas etapas, en su orden)
    python scripts/preprocesar.py --omitir segmentacion texturas
    python scripts/preprocesar.py --dry-run               (solo muestra los comandos)
"""
from orquestador import correr
from rutas import DATASET_DIR, RESULTS_DIR, SEGMENTATION_DIR

# Dependencias: nombre con el que se importa -> paquete de pip. matplotlib solo lo usa
# prepare_dataset.py para el gráfico de distribución de clases (con --labels).
REQUISITOS = {
    "cv2": "opencv-python",
    "numpy": "numpy",
    "pandas": "pandas",
    "skimage": "scikit-image",
    "mahotas": "mahotas",
    "matplotlib": "matplotlib",
    "rembg": "rembg",
    "onnxruntime": "onnxruntime",
}

DATASETS = ["Dataset_papilionidae", "Dataset_pieridae"]
FOTOS = [DATASET_DIR / "Papilionidae", DATASET_DIR / "Pieridae"]
SEGMENTADAS = [SEGMENTATION_DIR / d / "segmented" for d in DATASETS]
MASCARAS = [SEGMENTATION_DIR / d / "masks" for d in DATASETS]
TEXTURA = SEGMENTATION_DIR / "texture_features_all.csv"
COLOR = SEGMENTATION_DIR / "color_features_all.csv"
FORMA = SEGMENTATION_DIR / "shape_features_all.csv"
VECTOR = SEGMENTATION_DIR / "features_dataset_ready.csv"
VECTOR_FORMA = RESULTS_DIR / "features_forma" / "features_dataset_ready.csv"

# nombre, script, argumentos, datos que necesita, datos que produce
STAGES = [
    ("segmentacion", "segmentacion/Segmentacion_rembg.py", [], FOTOS, SEGMENTADAS + MASCARAS),
    ("texturas", "caracteristicas/texturas.py", [], SEGMENTADAS + MASCARAS, [TEXTURA]),
    ("color", "caracteristicas/color_features.py", [], SEGMENTADAS + MASCARAS, [COLOR]),
    ("formas", "caracteristicas/formas.py", [], MASCARAS, [FORMA]),
    ("prepare", "caracteristicas/prepare_dataset.py", [], [TEXTURA, COLOR], [VECTOR]),
    ("prepare_forma", "caracteristicas/prepare_dataset.py", ["--shape-csv", str(FORMA), "--output", str(VECTOR_FORMA.parent)],
     [TEXTURA, COLOR, FORMA], [VECTOR_FORMA]),
]

if __name__ == "__main__":
    correr(__doc__, STAGES, REQUISITOS, "preprocesar.py",
           cierre=f"Vector de features: {VECTOR}\nSiguiente: python scripts/entrenar.py")
