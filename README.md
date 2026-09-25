# Clasificación de Mariposas Colombianas — Reconocimiento de Patrones

Proyecto académico (Universidad Militar Nueva Granada — Reconocimiento de Patrones) para
clasificar mariposas colombianas por **familia** (con expansión planeada a **género**),
a partir de fotografías de catálogo, usando características clásicas de **textura** y
**color** como entrada a un modelo de machine learning.

El enfoque es deliberadamente clásico (no deep learning end-to-end): el dataset es
pequeño y muy desbalanceado por clase, así que se prioriza un pipeline de
características diseñadas a mano (*hand-crafted features*) + clasificador clásico, en
vez de una red neuronal entrenada desde cero.

## Tabla de contenido

- [Pipeline general](#pipeline-general)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Requisitos](#requisitos)
- [Uso — paso a paso](#uso--paso-a-paso)
  1. [Extracción del dataset desde los catálogos PDF](#1-extracción-del-dataset-desde-los-catálogos-pdf)
  2. [Segmentación](#2-segmentación)
  3. [Extracción de características de textura](#3-extracción-de-características-de-textura)
  4. [Extracción de características de color](#4-extracción-de-características-de-color)
  5. [Preparación del dataset final](#5-preparación-del-dataset-final)
- [Fuente de datos](#fuente-de-datos)
- [Limitaciones conocidas](#limitaciones-conocidas)
- [Estado actual / próximos pasos](#estado-actual--próximos-pasos)

## Pipeline general

```
Catálogos PDF (ButterflyCatalogs.com)
        │
        ▼
extract_dataset.py        → imágenes + nombre científico (label), por familia
        │
        ▼
segment_butterflies.py    → máscara binaria + recorte sobre fondo blanco (GrabCut)
        │
        ├──► texturas.py         → GLCM + Haralick + Gabor  (~81 features/imagen)
        │
        └──► color_features.py  → histogramas + momentos HSV/BGR (~66 features/imagen)
        │
        ▼
prepare_dataset.py        → une textura+color, audita calidad, distribución de clases
        │
        ▼
features_dataset_ready.csv   → listo para entrenar el clasificador
```

Cada etapa escribe su salida a disco (imágenes, máscaras, CSVs) de forma independiente,
para poder revisar la calidad en cada paso antes de avanzar al siguiente.

## Estructura del repositorio

```
Proyecto-clasificacion-mariposas-Reconocimiento-de-patrones/
├── Papilionidae/                      # fotos originales, familia Papilionidae
├── Pieridae/                          # fotos originales, familia Pieridae
├── segmentacion/
│   ├── Dataset_papilionidae/          # salida de segmentación (v1)
│   │   ├── masks/
│   │   ├── segmented/
│   │   ├── preview/
│   │   └── texture_features.csv / color_features.csv
│   ├── Dataset_pieridae/
│   ├── Dataset_papilionidae_v2/       # salida de segmentación (v2, refinada)
│   ├── Dataset_pieridae_v2/
│   ├── texture_features_all.csv       # combinado, ambas familias
│   ├── color_features_all.csv         # combinado, ambas familias
│   ├── features_dataset_ready.csv     # dataset final, listo para ML
│   ├── class_distribution.png
│   └── quality_report.txt
├── extract_dataset.py
├── rename_by_label.py
├── segment_butterflies.py
├── texturas.py
├── color_features.py
├── prepare_dataset.py
└── README.md
```

## Requisitos

```bash
pip install pymupdf opencv-python numpy scikit-image mahotas pandas matplotlib
```

> En Windows, si `mahotas` falla al instalar (requiere compilar), probar con
> `conda install -c conda-forge mahotas` o `pip install mahotas --only-binary :all:`.

Todos los scripts se pueden correr **sin argumentos** (las rutas por defecto ya están
configuradas para la estructura de este repo) o pasando rutas específicas por línea de
comandos — cada uno documenta sus opciones con `--help`.

## Uso — paso a paso

### 1. Extracción del dataset desde los catálogos PDF

```bash
python extract_dataset.py catalogo_papilionidae.pdf catalogo_pieridae.pdf -o dataset_salida/
```

Recorre cada PDF con PyMuPDF, localiza cada imagen embebida y la asocia con el texto en
**cursiva** más cercano por debajo (el nombre científico). Las imágenes sin un nombre
científico cerca (portadas, mapas, logos) se descartan automáticamente. Salida:
`images/<catalogo>/NNN_Genero_especie.png` + `labels.csv` / `labels.json`.

`rename_by_label.py` es un utilitario aparte para renombrar imágenes ya descargadas a
partir de un CSV externo `(filename, label)` (por ejemplo un dataset ya etiquetado tipo
Kaggle), incrustando el label en el nombre de archivo sin sobrescribir nada.

### 2. Segmentación

```bash
python segment_butterflies.py
```

Separa la mariposa del fondo con **GrabCut** (OpenCV): inicializa con un rectángulo que
cubre casi toda la imagen, itera para refinar los modelos de color de fondo/frente, y
limpia el resultado (morfología, relleno de huecos, componente conexo principal).

La versión actual incluye un **refinamiento de dos pasadas**: se detectan posibles
intrusos (piel de dedos, flores de fondo muy saturadas) y, en vez de restarlos
directamente de la máscara, se pasan como semilla de "fondo probable" a una segunda
pasada de GrabCut — así el algoritmo reconsidera esos píxeles con el contexto completo
(su propio modelo de color + bordes reales), en vez de cortar a ciegas por color. Esto
evita tanto incluir dedos/flores como, el error opuesto, morder el interior real del ala
en especies de colores claros.

Salida por dataset: `masks/`, `segmented/` (recorte sobre fondo blanco) y `preview/`
(collage original|máscara|recorte de las primeras imágenes, para revisar calidad rápido).

### 3. Extracción de características de textura

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

### 4. Extracción de características de color

```bash
python color_features.py
```

Mismo patrón que `texturas.py`, pero en espacio **HSV**: histogramas normalizados
(16 bins × 3 canales) + momentos de color (media, desviación estándar, asimetría, en
HSV y BGR). Genera `color_features_all.csv`.

### 5. Preparación del dataset final

```bash
python prepare_dataset.py --labels ruta/a/labels.csv
```

Une textura + color por `(filename, dataset)`, audita calidad (filas con NaN o con muy
pocos píxeles de mariposa detectados — segmentación fallida), reporta la distribución de
clases **por especie y por género** (con un umbral configurable de mínimo de ejemplos
por clase), descarta columnas sin información (casi constantes o con demasiado NaN), y
guarda `features_dataset_ready.csv` listo para el split de entrenamiento/prueba.

El escalado de features (`StandardScaler`, etc.) se deja fuera de este script a
propósito — debe ajustarse *después* de separar train/test, para no filtrar información
del conjunto de prueba.

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

- [x] Extracción automatizada del dataset desde catálogos PDF
- [x] Segmentación (GrabCut, dos pasadas)
- [x] Extracción de características de textura (GLCM, Haralick, Gabor)
- [x] Extracción de características de color (histogramas y momentos HSV/BGR)
- [x] Unión, auditoría de calidad y reporte de distribución de clases
- [ ] Entrenamiento y evaluación del clasificador (familia)
- [ ] Expansión a clasificación por género
- [ ] Análisis de resultados (métricas, matriz de confusión, importancia de features)
