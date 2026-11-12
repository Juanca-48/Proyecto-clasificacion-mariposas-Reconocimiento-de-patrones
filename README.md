# Clasificación de Mariposas Colombianas — Reconocimiento de Patrones

Proyecto académico (Universidad Militar Nueva Granada — Reconocimiento de Patrones) para
clasificar mariposas colombianas por **familia** (con expansión planeada a **género**),
a partir de fotografías de catálogo, usando características clásicas de **textura** y
**color** como entrada a un modelo de machine learning.

## Tabla de contenido

- [Pipeline general](#pipeline-general)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Requisitos](#requisitos)
- [Uso — paso a paso](#uso--paso-a-paso)
  1. [Segmentación](#1-segmentación)
  2. [Extracción de características de textura](#2-extracción-de-características-de-textura)
  3. [Extracción de características de color](#3-extracción-de-características-de-color)
  4. [Preparación del dataset final](#4-preparación-del-dataset-final)
  5. [División train / val / test](#5-división-train--val--test)
  6. [Entrenamiento y evaluación (Random Forest)](#6-entrenamiento-y-evaluación-random-forest)
- [Fuente de datos](#fuente-de-datos)
- [Limitaciones conocidas](#limitaciones-conocidas)
- [Estado actual / próximos pasos](#estado-actual--próximos-pasos)

## Pipeline general

```
Papilionidae/ , Pieridae/   (fotos originales, nombradas NNN_Genero_especie.png)
        │
        ▼
segmentacionV2.py         → máscara binaria + recorte sobre fondo blanco (GrabCut, 2 pasadas)
        │
        ├──► texturas.py         → GLCM + Haralick + Gabor  (~81 features/imagen)
        │
        └──► color_features.py  → histogramas + momentos HSV/BGR (~66 features/imagen)
        │
        ▼
prepare_dataset.py        → une textura+color, audita calidad, distribución de clases
        │                    → segmentacion/features_dataset_ready.csv
        ▼
Split_dataset.py          → división 70/15/15 agrupada por especie → dataset_split/
        │
        ▼
Train_random_forest.py    → Random Forest por familia, métricas e importancia → resultados_rf/
```

Cada etapa escribe su salida a disco (imágenes, máscaras, CSVs) de forma independiente,
para poder revisar la calidad en cada paso antes de avanzar al siguiente.

## Estructura del repositorio

```
Proyecto-clasificacion-mariposas-Reconocimiento-de-patrones/
├── Papilionidae/                      # fotos originales, familia Papilionidae
├── Pieridae/                          # fotos originales, familia Pieridae
├── Papilionidae.csv                   # etiquetas: photo_number, filename, label, genus
├── Pieridae.csv
├── segmentacion/
│   ├── Dataset_papilionidae/          # salida de segmentación v1 (Segmentacion.py)
│   ├── Dataset_pieridae/
│   ├── Dataset_papilionidae_v2/       # salida de segmentación v2 (segmentacionV2.py)
│   │   ├── masks/
│   │   ├── segmented/
│   │   └── preview/
│   ├── Dataset_pieridae_v2/
│   ├── texture_features_all.csv       # combinado, ambas familias
│   ├── color_features_all.csv         # combinado, ambas familias
│   ├── features_dataset_ready.csv     # dataset final, listo para ML
│   └── quality_report.txt
├── dataset_split/
│   ├── segmented/                     # imágenes copiadas por partición
│   ├── features_train.csv / features_val.csv / features_test.csv
│   ├── splits.csv                     # manifiesto imagen → partición
│   └── split_report.txt
├── mariposas_analisis.xlsx            # análisis exploratorio del dataset
├── Segmentacion.py                    # segmentación v1 (GrabCut simple)
├── segmentacionV2.py                  # segmentación v2 (GrabCut, dos pasadas)
├── texturas.py
├── color_features.py
├── prepare_dataset.py
├── Split_dataset.py
├── Train_random_forest.py
└── README.md
```

## Requisitos

```bash
pip install opencv-python numpy scikit-image mahotas pandas matplotlib scikit-learn joblib
```

> En Windows, si `mahotas` falla al instalar (requiere compilar), probar con
> `conda install -c conda-forge mahotas` o `pip install mahotas --only-binary :all:`.

Todos los scripts se pueden correr **sin argumentos** o pasando rutas específicas por
línea de comandos — cada uno documenta sus opciones con `--help`.

> **Importante:** las rutas por defecto están escritas como rutas absolutas de Windows
> (`REPO_DIR` / `PROYECTO_DIR` al inicio de cada script). Si clonas el repo en otra
> ubicación, edita esa constante o pasa las rutas por argumento.

## Uso — paso a paso

### 1. Segmentación

```bash
python segmentacionV2.py
```

Separa la mariposa del fondo con **GrabCut** (OpenCV): inicializa con un rectángulo que
cubre casi toda la imagen, itera para refinar los modelos de color de fondo/frente, y
limpia el resultado (morfología, relleno de huecos, componente conexo principal).
Por defecto lee `Papilionidae/` y `Pieridae/` y escribe en
`segmentacion/Dataset_papilionidae_v2` y `segmentacion/Dataset_pieridae_v2`.

La v2 incluye un **refinamiento de dos pasadas**: se detectan posibles intrusos (piel
de dedos, flores de fondo muy saturadas) y, en vez de restarlos directamente de la
máscara, se pasan como semilla de "fondo probable" a una segunda pasada de GrabCut — así
el algoritmo reconsidera esos píxeles con el contexto completo (su propio modelo de
color + bordes reales), en vez de cortar a ciegas por color. Esto evita tanto incluir
dedos/flores como, el error opuesto, morder el interior real del ala en especies de
colores claros.

`Segmentacion.py` es la versión anterior (v1, una sola pasada, opción `--method otsu`);
se conserva como referencia y generó `segmentacion/Dataset_papilionidae` y
`segmentacion/Dataset_pieridae`.

Salida por dataset: `masks/`, `segmented/` (recorte sobre fondo blanco) y `preview/`
(collage original|máscara|recorte de las primeras imágenes, para revisar calidad rápido).

### 2. Extracción de características de textura

```bash
python texturas.py
```

Autodetecta los datasets dentro de `segmentacion/` y extrae, por imagen (excluyendo el
fondo mediante la máscara):

- **GLCM** (scikit-image): contraste, disimilitud, homogeneidad, energía, correlación,
  ASM — a 3 distancias × 4 ángulos, promediados (36 columnas).
- **Haralick** (mahotas): los 13 descriptores clásicos (Haralick, 1979).
- **Gabor** (scikit-image): banco de filtros a 4 frecuencias × 4 orientaciones,
  media/std de la magnitud de respuesta (32 columnas).

Genera un CSV por dataset más `texture_features_all.csv` combinado.

### 3. Extracción de características de color

```bash
python color_features.py
```

Mismo patrón que `texturas.py`, pero en espacio **HSV**: histogramas normalizados
(16 bins × 3 canales) + momentos de color (media, desviación estándar, asimetría, en
HSV y BGR). Genera `color_features_all.csv`.

### 4. Preparación del dataset final

```bash
python prepare_dataset.py --labels Papilionidae.csv Pieridae.csv
```

Une textura + color por `(filename, dataset)`, audita calidad (filas con NaN o con muy
pocos píxeles de mariposa detectados — segmentación fallida), reporta la distribución de
clases **por especie y por género** (con un umbral configurable de mínimo de ejemplos
por clase), descarta columnas sin información (casi constantes o con demasiado NaN), y
guarda `segmentacion/features_dataset_ready.csv`.

El escalado de features (`StandardScaler`, etc.) se deja fuera de este script a
propósito — debe ajustarse *después* de separar train/test, para no filtrar información
del conjunto de prueba.

### 5. División train / val / test

```bash
python Split_dataset.py
```

Divide las imágenes de los datasets v2 en **70 % train / 15 % val / 15 % test** por
familia (semilla 42). La división es **agrupada por especie** (`--split-mode group`, por
defecto): todas las fotos de una misma especie quedan en la misma partición, para que el
modelo se evalúe sobre especies que nunca vio y no memorice ejemplares. La especie se lee
del nombre de archivo (`001_Battus_crassus_crassus.png` → `Battus_crassus_crassus`).

Salida en `dataset_split/`: las imágenes segmentadas copiadas por partición, los CSV de
features por partición (`features_train.csv`, `features_val.csv`, `features_test.csv`),
el manifiesto `splits.csv` y `split_report.txt`, que verifica que ninguna especie se
repita entre particiones.

Resultado actual: 812 imágenes → train 568, val 122, test 122
(Papilionidae 285, Pieridae 527).

Opciones útiles: `--also masks original` (copiar también máscaras/originales),
`--no-copy` (solo manifiesto y reporte), `--overwrite`.

### 6. Entrenamiento y evaluación (Random Forest)

```bash
python Train_random_forest.py
```

Entrena un **Random Forest** para distinguir la familia (Papilionidae vs Pieridae); la
familia se deduce de la columna `dataset`. Si existe `dataset_split/`, usa el
**modo dividido**:

1. Verifica que no haya imágenes ni especies repetidas entre particiones.
2. Ajusta hiperparámetros con una malla pequeña y validación cruzada agrupada por
   especie sobre train + val, eligiendo la configuración más simple dentro de un error
   estándar de la mejor.
3. Evalúa en val un modelo entrenado solo con train (control).
4. Ablación con validación cruzada: solo textura, solo color, ambas.
5. Evalúa **una sola vez en test** el modelo final (entrenado con train + val), con
   intervalos de confianza al 95 % por bootstrap por especie y comparación contra un
   clasificador que siempre predice la familia mayoritaria.
6. Importancia de features por permutación (medida en val) y agregada por grupo.

Con `--csv segmentacion/features_dataset_ready.csv` usa el **modo un solo CSV**
(exploratorio, con hold-out interno). Otras opciones: `--no-tune`, `--skip-ablation`,
`--n-estimators`, `--cv-folds`, `--seed`.

Salida en `resultados_rf/`: `model_family_rf.joblib`, `metrics_report.txt`,
`cv_results.csv`, `cv_grid_results.csv`, `test_predictions.csv`,
`feature_importance.csv` / `.png` y `confusion_matrix.png`.

Notas: no se escalan features (innecesario en Random Forest), `fg_pixels` se excluye
para no dar un atajo ligado al tamaño de imagen, los NaN se imputan con la mediana de
entrenamiento, y se usa `class_weight="balanced"` reportando balanced accuracy y F1 macro.

## Fuente de datos

Catálogos fotográficos de **BioButterfly Database** —
[butterflycatalogs.com/colombia.html](https://www.butterflycatalogs.com/colombia.html) —,
un proyecto colaborativo de documentación fotográfica de mariposas colombianas y
neotropicales, con identificación taxonómica revisada por especialistas. Cada catálogo
(uno por familia) contiene fotografías numeradas en cuadrícula con el nombre científico,
rango de distribución y localidad de cada ejemplar.

## Limitaciones conocidas

- **Desbalance de clases severo**: varias especies del catálogo original tienen una sola
  fotografía. La clasificación por especie individual no es viable con este dataset sin
  agregación taxonómica (familia/género) o técnicas de balanceo.
- **Segmentación automática, no perfecta**: GrabCut no tiene noción semántica de "qué es
  una mariposa" — en casos extremos (objetos de color/enfoque muy similar al insecto)
  puede seguir cometiendo errores. Revisar siempre la carpeta `preview/` de cada dataset
  tras segmentar.
- **Features de textura/color dependen de la máscara**: una segmentación fallida se
  propaga silenciosamente a los CSV de características — de ahí la auditoría de calidad
  en `prepare_dataset.py` (columna `low_quality`, `fg_pixels` bajo).

## Estado actual / próximos pasos

- [x] Segmentación (GrabCut, dos pasadas)
- [x] Extracción de características de textura (GLCM, Haralick, Gabor)
- [x] Extracción de características de color (histogramas y momentos HSV/BGR)
- [x] Unión, auditoría de calidad y reporte de distribución de clases
- [x] División train / val / test agrupada por especie
- [ ] Entrenamiento y evaluación del clasificador (familia) — script listo, falta correrlo y reportar resultados
- [ ] Expansión a clasificación por género
- [ ] Análisis de resultados (métricas, matriz de confusión, importancia de features)
