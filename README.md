# Clasificación de Mariposas Colombianas — Reconocimiento de Patrones

Proyecto académico (Universidad Militar Nueva Granada — Reconocimiento de Patrones) para
clasificar mariposas colombianas por **familia** (Papilionidae vs Pieridae),
a partir de fotografías de catálogo, usando características clásicas de **textura**,
**color** y **forma** como entrada a un modelo de machine learning.

## Tabla de contenido

- [Estructura del repositorio](#estructura-del-repositorio)
- [Pipeline general](#pipeline-general)
- [Requisitos](#requisitos)
- [Uso — paso a paso](#uso--paso-a-paso)
  1. [Segmentación](#1-segmentación)
  2. [Extracción de características de textura](#2-extracción-de-características-de-textura)
  3. [Extracción de características de color](#3-extracción-de-características-de-color)
  4. [Preparación del dataset final](#4-preparación-del-dataset-final)
  5. [División train / val / test](#5-división-train--val--test)
  6. [Entrenamiento y evaluación (Random Forest)](#6-entrenamiento-y-evaluación-random-forest)
  7. [Estandarización de features](#7-estandarización-de-features)
  8. [Comparación de modelos](#8-comparación-de-modelos)
  9. [Modelo final](#9-modelo-final)
  10. [Variante con descriptores de forma](#10-variante-con-descriptores-de-forma)
- [Fuente de datos](#fuente-de-datos)
- [Limitaciones conocidas](#limitaciones-conocidas)
- [Estado actual / próximos pasos](#estado-actual--próximos-pasos)

## Estructura del repositorio

```
Proyecto-clasificacion-mariposas-Reconocimiento-de-patrones/
├── dataset/                           # datos de entrada
│   ├── Papilionidae/                  # fotos originales, familia Papilionidae
│   ├── Pieridae/                      # fotos originales, familia Pieridae
│   ├── Papilionidae.csv               # etiquetas: photo_number, filename, label, genus
│   └── Pieridae.csv
├── scripts/                           # todos los scripts
│   ├── rutas.py                       # rutas del proyecto (las usan todos los scripts)
│   ├── orquestador.py                 # motor común de preprocesar.py y entrenar.py
│   ├── preprocesar.py                 # ejecuta todo el preprocesamiento en orden (pasos 1 a 4)
│   ├── entrenar.py                    # ejecuta división, entrenamiento y modelo final (pasos 5 a 10)
│   ├── Segmentacion.py
│   ├── texturas.py
│   ├── color_features.py
│   ├── formas.py
│   ├── prepare_dataset.py
│   ├── Split_dataset.py
│   ├── Train_random_forest.py
│   ├── Standardize_features.py
│   ├── Train_models.py
│   ├── Train_final.py
│   ├── viabilidad_genero.py           # estudio de viabilidad de clasificar por género
│   ├── inferencia.py                  # clasifica una imagen y rechaza lo que no parece una mariposa
│   ├── calibrar_rechazo.py            # calibra los umbrales de rechazo (resultados/rechazo/)
│   ├── probar_carpeta.py              # pasa las fotos de una carpeta por el clasificador de la app
│   ├── validar_escala.py              # valida la normalización de escala de las fotos (resultados/escala/)
│   ├── validar_segmentador.py         # compara GrabCut con rembg en la inferencia (resultados/segmentador/)
│   └── exportar_app.py                # copia a app/ lo que necesita el Space
├── app/                               # Space de Hugging Face (Gradio): app.py, requirements.txt, README.md, modelo/
├── resultados/                        # salida de los scripts, una subcarpeta por etapa
│   ├── segmentacion/                  # Segmentacion.py, texturas.py, color_features.py, formas.py, prepare_dataset.py
│   │   ├── Dataset_papilionidae/      #   masks/, segmented/, preview/, texture_features.csv, color_features.csv
│   │   ├── Dataset_pieridae/
│   │   ├── texture_features_all.csv / color_features_all.csv / shape_features_all.csv
│   │   ├── features_dataset_ready.csv #   dataset final, listo para ML
│   │   └── quality_report.txt
│   ├── features_forma/                # features_dataset_ready.csv con descriptores de forma (paso 10)
│   ├── dataset_split/                 # Split_dataset.py: segmented/, features_{train,val,test}.csv, splits.csv
│   │   └── standardized/              #   Standardize_features.py (ignorado en git)
│   ├── dataset_split_forma/           # misma partición con el vector que incluye forma
│   ├── rf/                            # Train_random_forest.py
│   ├── modelos/                       # Train_models.py (ignorado en git)
│   ├── modelos_forma/                 # (ignorado en git)
│   ├── final/                         # Train_final.py: final_model.joblib
│   ├── final_forma/
│   ├── genero_viabilidad/             # viabilidad_genero.py: tabla de géneros, cobertura, piloto
│   ├── rechazo/                       # calibrar_rechazo.py: rechazo.joblib y resumen.txt
│   ├── pruebas_app/                   # probar_carpeta.py: resultado por foto y segmentaciones
│   ├── escala/                        # validar_escala.py: comparación de políticas de escala
│   └── segmentador/                   # validar_segmentador.py: GrabCut vs rembg, con montaje de segmentaciones
├── mariposas_analisis.xlsx            # análisis exploratorio del dataset
└── README.md
```

## Pipeline general

El preprocesamiento completo se lanza con `python scripts/preprocesar.py`. Los scripts se
ejecutan desde cualquier directorio (`python scripts/<nombre>.py`): las rutas
se calculan desde la ubicación del propio script en `scripts/rutas.py`, así que el repo se
puede clonar en cualquier carpeta.

```
dataset/Papilionidae , dataset/Pieridae   (fotos originales, nombradas NNN_Genero_especie.png)
        │
        ▼
Segmentacion.py           → máscara binaria + recorte sobre fondo blanco (GrabCut)
        │
        ├──► texturas.py         → GLCM + Haralick + Gabor  (~81 features/imagen)
        ├──► color_features.py  → histogramas + momentos HSV/BGR (~66 features/imagen)
        └──► formas.py          → 16 descriptores de forma de la silueta (opcional, paso 10)
        │
        ▼
prepare_dataset.py        → une las features, audita calidad, distribución de clases
        │                    → resultados/segmentacion/features_dataset_ready.csv
        ▼
Split_dataset.py          → división 70/15/15 agrupada por especie → resultados/dataset_split/
        │
        ├──► Train_random_forest.py → Random Forest por familia (línea base) → resultados/rf/
        │
        ▼
Standardize_features.py   → z-score ajustado con train → resultados/dataset_split/standardized/
        │
        ▼
Train_models.py           → compara regresión logística, SVM, kNN y Random Forest → resultados/modelos/
        │
        ▼
Train_final.py            → modelo final (pipeline completo, sin estandarizar a mano) → resultados/final/
```

Cada etapa escribe su salida a disco (imágenes, máscaras, CSVs) de forma independiente,
para poder revisar la calidad en cada paso antes de avanzar al siguiente.

> **Segmentación usada:** el proyecto trabaja con la segmentación de `Segmentacion.py`, que a
> veces deja dedos, pétalos o sombras dentro de la máscara. Se probó una variante con una
> segunda pasada de GrabCut para quitarlos, pero recortaba partes del ala (el detector de
> color confundía alas amarillas, crema o con manchas rojas con intrusos), así que se
> descartó y se eliminó del repo.

## Requisitos

```bash
pip install opencv-python numpy scikit-image mahotas pandas matplotlib scikit-learn joblib
```

> En Windows, si `mahotas` falla al instalar (requiere compilar), probar con
> `conda install -c conda-forge mahotas` o `pip install mahotas --only-binary :all:`.

Todos los scripts se pueden correr **sin argumentos** o pasando rutas específicas por
línea de comandos — cada uno documenta sus opciones con `--help`.

## Uso — paso a paso

**Preprocesamiento de una sola vez** (pasos 1 a 4, más los descriptores de forma del paso 10):

```bash
python scripts/preprocesar.py                        # segmentación → texturas → color → formas → prepare
python scripts/preprocesar.py --estado               # qué etapas ya tienen sus resultados
python scripts/preprocesar.py --desde color          # de una etapa en adelante
python scripts/preprocesar.py --desde color --hasta formas
python scripts/preprocesar.py --solo formas prepare  # solo esas etapas
python scripts/preprocesar.py --omitir segmentacion texturas
python scripts/preprocesar.py --dry-run              # solo muestra los comandos
```

Las etapas son `segmentacion`, `texturas`, `color`, `formas`, `prepare` y `prepare_forma`; se
ejecutan siempre en ese orden. Antes de correr, el script verifica que existan los datos que
cada etapa necesita (los que no se generen en la misma corrida ya deben estar en disco) y, si
falta algo, dice qué etapa lo produce.

**División, entrenamiento y modelo final de una sola vez** (pasos 5 a 10), con las mismas
opciones (`--estado`, `--desde`, `--hasta`, `--solo`, `--omitir`, `--dry-run`):

```bash
python scripts/entrenar.py                            # las 9 etapas
python scripts/entrenar.py --hasta final              # solo el vector sin forma
python scripts/entrenar.py --desde estandarizar       # sin repetir la división ni el Random Forest
python scripts/entrenar.py --solo modelos final
```

Etapas: `split`, `rf`, `estandarizar`, `modelos`, `final` (vector sin forma) y `split_forma`,
`estandarizar_forma`, `modelos_forma`, `final_forma` (vector con forma, con la misma partición).
`split` se ejecuta con `--overwrite`: vuelve a copiar las imágenes de `dataset_split/segmented/`
y da las mismas particiones (semilla 42). Instala con `pip` las dependencias que falten
(numpy, pandas, scikit-learn, joblib, matplotlib). Los datos de entrada se generan con
`preprocesar.py`.

Antes de empezar comprueba las dependencias del preprocesamiento (opencv-python, numpy,
pandas, scikit-image, mahotas, matplotlib) e instala con `pip` las que falten, así que no
hace falta instalar nada a mano. Cada etapa se ejecuta al terminar la anterior con los valores
por defecto de su script; si una falla, se detiene y muestra cómo reanudar. Etapas, en orden: `segmentacion`, `texturas`,
`color`, `formas`, `prepare` (vector sin forma, en `resultados/segmentacion/`) y
`prepare_forma` (vector con forma, en `resultados/features_forma/`). Termina en el vector de
features; la división (paso 5) y el entrenamiento se ejecutan aparte. La segmentación y las
texturas tardan varios minutos; el resto, menos de un minuto. Cada paso también se puede
correr por separado, como se describe a continuación.

### 1. Segmentación

```bash
python scripts/Segmentacion.py
```

Lee `dataset/Papilionidae` y `dataset/Pieridae` y separa la mariposa del fondo con
**GrabCut** (OpenCV): inicializa con un rectángulo que cubre casi toda la imagen, itera
para refinar los modelos de color de fondo/frente, y limpia el resultado (morfología,
relleno de huecos, componente conexo principal). Admite también `--method otsu`.

Salida en `resultados/segmentacion/Dataset_papilionidae` y `Dataset_pieridae`: `masks/`,
`segmented/` (recorte sobre fondo blanco) y `preview/` (collage original|máscara|recorte
de las primeras imágenes, para revisar calidad rápido). Los nombres `Dataset_*` son el valor
de la columna `dataset` en todos los CSV de features, por lo que no deben cambiarse.

### 2. Extracción de características de textura

```bash
python scripts/texturas.py
```

Autodetecta los datasets dentro de `resultados/segmentacion/` y extrae, por imagen
(excluyendo el fondo mediante la máscara):

- **GLCM** (scikit-image): contraste, disimilitud, homogeneidad, energía, correlación,
  ASM — a 3 distancias × 4 ángulos, promediados (36 columnas).
- **Haralick** (mahotas): los 13 descriptores clásicos (Haralick, 1979).
- **Gabor** (scikit-image): banco de filtros a 4 frecuencias × 4 orientaciones,
  media/std de la magnitud de respuesta (32 columnas).

Genera un CSV por dataset más `texture_features_all.csv` combinado.

### 3. Extracción de características de color

```bash
python scripts/color_features.py
```

Mismo patrón que `texturas.py`, pero en espacio **HSV**: histogramas normalizados
(16 bins × 3 canales) + momentos de color (media, desviación estándar, asimetría, en
HSV y BGR). Genera `color_features_all.csv`.

### 4. Preparación del dataset final

```bash
python scripts/prepare_dataset.py
```

Une textura + color por `(filename, dataset)`, audita calidad (filas con NaN o con muy
pocos píxeles de mariposa detectados — segmentación fallida), reporta la distribución de
clases **por especie y por género** (con un umbral configurable de mínimo de ejemplos
por clase), descarta columnas sin información (casi constantes o con demasiado NaN), y
guarda `resultados/segmentacion/features_dataset_ready.csv`. Con `--labels
dataset/Papilionidae.csv dataset/Pieridae.csv` agrega la etiqueta de especie y género.

El escalado de features (`StandardScaler`, etc.) se deja fuera de este script a
propósito — debe ajustarse *después* de separar train/test, para no filtrar información
del conjunto de prueba.

### 5. División train / val / test

```bash
python scripts/Split_dataset.py
```

Divide las imágenes de `Dataset_papilionidae` y `Dataset_pieridae` en
**70 % train / 15 % val / 15 % test** por familia (semilla 42). La división es **agrupada
por especie** (`--split-mode group`, por defecto): todas las fotos de una misma especie
quedan en la misma partición, para que el modelo se evalúe sobre especies que nunca vio y
no memorice ejemplares. La especie se lee del nombre de archivo
(`001_Battus_crassus_crassus.png` → `Battus_crassus_crassus`).

Salida en `resultados/dataset_split/`: las imágenes segmentadas copiadas por partición, los
CSV de features por partición (`features_train.csv`, `features_val.csv`, `features_test.csv`),
el manifiesto `splits.csv` y `split_report.txt`, que verifica que ninguna especie se
repita entre particiones. Los valores de las features no se usan para decidir la
partición; solo se reparten las filas.

Resultado actual: 812 imágenes → train 568, val 122, test 122
(Papilionidae 285, Pieridae 527).

Opciones útiles: `--also masks original` (copiar también máscaras/originales),
`--no-copy` (solo manifiesto y reporte), `--overwrite`, `--features-csv`, `--output`.

### 6. Entrenamiento y evaluación (Random Forest)

```bash
python scripts/Train_random_forest.py
```

Entrena un **Random Forest** para distinguir la familia (Papilionidae vs Pieridae); la
familia se deduce de la columna `dataset`. Si existe `resultados/dataset_split/`, usa el
**modo dividido**:

1. Verifica que no haya imágenes ni especies repetidas entre particiones.
2. Ajusta hiperparámetros con una malla pequeña y validación cruzada agrupada por
   especie sobre train + val, eligiendo la configuración más simple dentro de un error
   estándar de la mejor.
3. Evalúa en val un modelo entrenado solo con train (control).
4. Ablación con validación cruzada: solo textura, solo color, solo forma (si hay), todas.
5. Evalúa **una sola vez en test** el modelo final (entrenado con train + val), con
   intervalos de confianza al 95 % por bootstrap por especie y comparación contra un
   clasificador que siempre predice la familia mayoritaria.
6. Importancia de features por permutación (medida en val) y agregada por grupo.

Con `--csv resultados/segmentacion/features_dataset_ready.csv` usa el **modo un solo CSV**
(exploratorio, con hold-out interno). Otras opciones: `--no-tune`, `--skip-ablation`,
`--n-estimators`, `--cv-folds`, `--seed`.

Salida en `resultados/rf/`: `model_family_rf.joblib`, `metrics_report.txt`,
`cv_results.csv`, `cv_grid_results.csv`, `test_predictions.csv`,
`feature_importance.csv` / `.png` y `confusion_matrix.png`.

Notas: no se escalan features (innecesario en Random Forest), `fg_pixels` se excluye
para no dar un atajo ligado al tamaño de imagen, los NaN se imputan con la mediana de
entrenamiento, y se usa `class_weight="balanced"` reportando balanced accuracy y F1 macro.
La especie de cada imagen se lee del nombre del archivo, porque los CSV de features no
traen la columna `label`; sin eso la validación "agrupada" no agruparía nada.

Resultado en test: balanced accuracy 0.811 (IC 95 % [0.674, 0.900]), AUC 0.887.

### 7. Estandarización de features

```bash
python scripts/Standardize_features.py
```

Lee `features_{train,val,test}.csv`, imputa los NaN con la mediana de **train** y aplica
z-score con la media y desviación de **train** (val y test usan esos mismos valores, para no
filtrar información del test). Calcula además, por imagen, la media y desviación de los
z-scores de cada grupo (histograma HSV, momentos de color, GLCM, Haralick, Gabor) y de los
agregados `Color` y `Textura` (columnas `m_*`). Guarda `features_<split>_std.csv` y
`scaler.joblib` en `resultados/dataset_split/standardized/`.

Random Forest no necesita este paso (sus cortes no dependen de la escala); sirve para
modelos sensibles a la escala. Las columnas `m_*` se pueden usar aparte, pero combinan las
mismas 147 features y están muy correlacionadas con ellas.

### 8. Comparación de modelos

```bash
python scripts/Train_models.py
```

Compara regresión logística, SVM (kernel RBF), kNN y Random Forest (referencia) sobre las
147 features estandarizadas, con una malla pequeña por modelo y validación cruzada agrupada
por especie sobre train + val. Evalúa en test **una sola vez**, solo el modelo con mayor
balanced accuracy en CV. Salida en `resultados/modelos/`: `comparacion_modelos_cv.csv`,
`metrics_report.txt`, `test_predictions.csv`, `confusion_matrix.png`, `model_best.joblib`.

Las diferencias entre regresión logística y SVM son menores que la desviación entre folds:
con 150 especies en train + val y 24 en test, no hay base para distinguirlos.

### 9. Modelo final

```bash
python scripts/Train_final.py
```

Entrena el modelo elegido (por defecto la **regresión logística**, el más simple dentro de
un error estándar del mejor; `--model "SVM (RBF)"` para cambiarlo) como un solo pipeline:
imputación por mediana → estandarización → clasificador, ajustado con train + val. Recibe
features **sin estandarizar**, así que funciona sin `Standardize_features.py`. Escribe en
`resultados/final/`: `final_model.joblib` (se comprueba al guardarlo que recargado da las
mismas predicciones), `metrics_report.txt`, `test_predictions.csv` y `confusion_matrix.png`.

Resultado actual en test (24 especies, 122 imágenes): balanced accuracy 0.868
(IC 95 % [0.753, 0.936]), AUC 0.951. El test ya se vio en varias corridas de
`Train_models.py`; no debe usarse para seguir ajustando.

### 10. Variante con descriptores de forma

Los errores del modelo se concentran en especies cuyo color y textura se parecen a los de la
otra familia (mimetismo), así que se probó agregar la **forma de la silueta**.

```bash
python scripts/formas.py                           # resultados/segmentacion/shape_features_all.csv
python scripts/prepare_dataset.py --shape-csv resultados/segmentacion/shape_features_all.csv --output resultados/features_forma
python scripts/Split_dataset.py --features-csv resultados/features_forma/features_dataset_ready.csv --output resultados/dataset_split_forma --no-copy
python scripts/Standardize_features.py --split-dir resultados/dataset_split_forma
python scripts/Train_models.py --std-dir resultados/dataset_split_forma/standardized --output resultados/modelos_forma
python scripts/Train_final.py --split-dir resultados/dataset_split_forma --output resultados/final_forma
```

`formas.py` calcula 16 descriptores sobre las máscaras, invariantes a escala y traslación:
solidez, extensión, relación de aspecto, circularidad, convexidad, excentricidad, 7 momentos
de Hu y la cantidad y profundidad de las muescas del contorno (colas y lóbulos). No incluye
el área absoluta, que depende del tamaño de la foto. `python scripts/formas.py --selftest`
comprueba los descriptores con figuras sintéticas. La textura es la misma de antes.

La partición es idéntica a la del vector sin forma, por lo que los resultados se comparan
directamente. En la CV agrupada (train + val) con regresión logística:

| Features | Balanced accuracy |
|---|---|
| Solo forma (16) | 0.747 ± 0.049 |
| Solo color (66) | 0.799 ± 0.048 |
| Solo textura (81) | 0.893 ± 0.032 |
| Color + textura (147) | 0.885 ± 0.036 |
| Todas, con forma (163) | 0.890 ± 0.034 |

La forma sola separa las familias mejor que el azar, pero agregada a color y textura no
mejora la CV de forma apreciable (0.885 → 0.890, dentro de la desviación). En test la
regresión logística pasa de 0.868 a 0.890 de balanced accuracy y de 15 a 14 errores,
pero cambian 11 predicciones (6 corregidas y 5 nuevas, 4 de ellas de *Dismorphia
theucharila*), así que la mejora no se distingue del ruido de un test de 24 especies.

### 11. App en Hugging Face (con rechazo de imágenes)

La carpeta `app/` es un Space de Gradio listo para subir: el usuario sube una foto, la app la segmenta, la
clasifica con el modelo final (`resultados/final`) y muestra la familia, las probabilidades y la segmentación.
Gradio y no Static, porque el modelo necesita Python en cada imagen (GrabCut, mahotas, scikit-image, scikit-learn).

```bash
python scripts/calibrar_rechazo.py    # calibra los umbrales de rechazo -> resultados/rechazo/
python scripts/exportar_app.py        # copia módulos y modelos a app/ y comprueba que funciona aislada
cd app && python app.py               # probar en local (http://127.0.0.1:7860)
```

**Rechazo de lo que no parece una mariposa**, en tres capas (`scripts/inferencia.py`), calibradas solo con
train + val:
1. **Segmentación:** el área de la máscara debe estar en el rango visto en el entrenamiento.
2. **Distancia al conjunto de entrenamiento:** Mahalanobis (covarianza Ledoit-Wolf) en el espacio estandarizado
   del modelo. El umbral es el percentil 95 de una CV **agrupada por especie**, así no se rechazan especies nuevas
   legítimas.
3. **Confianza:** si ninguna familia supera el 65 %, el resultado es "incierto".

Resultados (`resultados/rechazo/resumen.txt`): sobre las 122 fotos legítimas del test, 2 % rechazadas, 4 % inciertas
y 94 % aceptadas, con exactitud 0,896 en las aceptadas (0,877 sin filtro). Con 14 imágenes que no son mariposas (las de
ejemplo de scikit-image) rechazó 11 y dejó pasar 3: un gato, un café y un astronauta. Es una comprobación de cordura, no
una evaluación. **Las mariposas de otra familia no se rechazan de forma fiable**: se parecen a estas en color, textura y
silueta, y distinguirlas exigiría una tercera clase con datos que no hay.

`GrabCut` es aleatorio: la misma foto puede dar máscaras distintas entre ejecuciones (IoU hasta 0,86). La app fija la
semilla (`cv2.setRNGSeed(0)`) para que sea reproducible. `Segmentacion.py` no lo hace, así que volver a segmentar el
dataset cambiaría un poco las máscaras y, con ellas, las features.

**Probar con fotos propias:** `python scripts/probar_carpeta.py --carpeta <carpeta>` pasa las fotos por el mismo
código de la app y las compara con la familia real, que se lee del nombre (`Papilionidae3.jpg`, `pieridae1.webp`).

**Fotos de internet.** Con 8 fotos de internet, la app acertó 2, marcó 1 como incierta, rechazó 2 y se equivocó en 3.
Cuatro de los seis fallos vienen de la segmentación (una hoja, una flor o una sombra dentro de la máscara), el defecto
de GrabCut con fondos más ricos que los del catálogo. Una **normalización de la escala** (llevar siempre el lado mayor
a 640 px) se probó con las 122 fotos de test reducidas a 450 y 330 px (`scripts/validar_escala.py`) y **se descartó**: no
mejora la balanced accuracy (-0,014 y -0,057, con IC que incluyen el cero) y sesga las predicciones hacia Pieridae,
aunque sí mejora el AUC y reduce los falsos rechazos. La app sigue con la política original: reducir solo lo que pase
de 800 px.

**Segmentador `rembg`** (`scripts/validar_segmentador.py`, requiere `pip install rembg onnxruntime` y descarga un modelo de
176 MB). Solo en la inferencia, sin reentrenar, quita las hojas, flores y sombras de las fotos de internet (en las 8 fotos,
las aceptadas y correctas pasan de 2 a 3), pero en el catálogo baja algo la precisión (balanced accuracy -0,018, IC
[-0,064, +0,015]) y sube el rechazo (de 1,6 % a 4,9 %) y la incertidumbre (de 5,7 % a 11,5 %), porque el modelo y los umbrales
se calibraron con máscaras de GrabCut. **No está adoptado en la app.** Hacerlo bien implicaría regenerar las máscaras con
rembg y reentrenar y recalibrar todo. Dos de las fotos de internet se rechazan con ambos segmentadores aunque la
segmentación sea limpia: son mariposas pequeñas en el encuadre (12 a 15 % del área, frente a la mediana del 32 % en el
catálogo) y la capa de distancia las penaliza; sin rechazo, el modelo las clasifica bien.

**Subir el Space** (requiere una cuenta en Hugging Face; no se ha hecho desde este repositorio):

```bash
hf auth login
hf repos create <usuario>/<nombre> --type space --sdk gradio
hf upload <usuario>/<nombre> app . --type space --exclude "**/__pycache__/**"
```

Se sube solo `app/` (código y modelos, unos 400 KB), nunca el dataset: las fotos son de los catálogos de BioButterfly.
`app/requirements.txt` fija `scikit-learn==1.9.1` porque el modelo es un pickle sensible a la versión, y
`opencv-python-headless==5.0.0.93` para que GrabCut se comporte igual que en el entrenamiento. El Space usa Python 3.12
(NumPy 2.5 lo exige). No se ha probado la construcción en Hugging Face: lo probado es la app en local y su carpeta aislada.
Tras reentrenar o recalibrar, hay que volver a ejecutar `exportar_app.py` y volver a subir.

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
  una mariposa". Puede dejar dedos, pétalos, sombras u hojas dentro de la máscara.
  Revisar la carpeta `preview/` de cada dataset tras segmentar.
- **Features de textura/color/forma dependen de la máscara**: una segmentación fallida se
  propaga silenciosamente a los CSV de características — de ahí la auditoría de calidad
  en `prepare_dataset.py` (columna `low_quality`, `fg_pixels` bajo).
- **Test pequeño**: 24 especies en test, así que el resultado varía bastante según cuáles
  caigan ahí. Los errores se concentran en especies de mimetismo (Pieridae oscuras con
  manchas rojas o amarillas) y en Papilionidae pálidas, donde el color por sí solo no
  separa las familias.

## Estado actual / próximos pasos

- [x] Segmentación (GrabCut)
- [x] Extracción de características de textura (GLCM, Haralick, Gabor)
- [x] Extracción de características de color (histogramas y momentos HSV/BGR)
- [x] Extracción de descriptores de forma
- [x] Unión, auditoría de calidad y reporte de distribución de clases
- [x] División train / val / test agrupada por especie
- [x] Entrenamiento y evaluación del clasificador de familia (Random Forest)
- [x] Estandarización y comparación de modelos
- [x] Modelo final (regresión logística)
- [ ] Otros descriptores para los mimetismos (patrón de venas, LBP)
- [x] Estudio de viabilidad de la clasificación por género (`scripts/viabilidad_genero.py`): **descartada**.
      Solo 17 de 38 géneros tienen 3 o más especies, y en el piloto (CV agrupada, sin usar el test) la
      balanced accuracy es 0.30 a 0.43 (3 a 5 veces el azar, pero por debajo de la mitad); por familia
      es 0.87 a 0.89. El alcance del proyecto queda en la clasificación por familia.
