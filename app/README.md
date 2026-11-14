---
title: Clasificador de mariposas
emoji: 🦋
colorFrom: yellow
colorTo: green
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12"
app_file: app.py
pinned: false
---

# Clasificador de mariposas colombianas (Papilionidae vs Pieridae)

Proyecto de Reconocimiento de Patrones (Universidad Militar Nueva Granada). Dada una foto, la app separa la
mariposa del fondo con GrabCut, extrae características de **textura** (GLCM, Haralick, Gabor) y **color**
(histogramas y momentos HSV/BGR) y las clasifica con una regresión logística entrenada con 690 fotos de catálogo.

## Dos modelos, con selector

La app alterna entre dos modelos con el selector "Modelo":

- **Clásico** (predeterminado): características de textura y color + regresión logística, como se describe arriba.
- **Afinado**: **ConvNeXt-Tiny** (una red preentrenada con ImageNet) con **todos sus pesos ajustados** a las fotos del catálogo
  (fine-tuning), sobre el recorte de la mariposa y con su propia capa de clasificación. En el test de catálogo (24 especies
  que la red no vio) da balanced accuracy 0,99 y AUC 1,00. Ojo: ImageNet incluye mariposas, así que la cifra puede ser
  optimista para fotos distintas a las del catálogo, y con pocas fotos de entrenamiento la red puede ser menos robusta con
  ellas.

La primera vez que arranca el Space descarga el modelo de segmentación `rembg` (unos 170 MB), por lo que el arranque tarda;
el selector solo muestra los modelos disponibles.

## Rendimiento

En el conjunto de prueba (24 especies que el modelo no vio durante el entrenamiento): balanced accuracy 0,868
(IC 95 %: 0,753 a 0,936), AUC 0,951. El IC es ancho porque el conjunto de prueba es pequeño.

## Rechazo de imágenes

Tres capas, de la más simple a la más fina:

1. **Segmentación:** se rechaza si la máscara no tiene el tamaño visto en el entrenamiento.
2. **Distancia al conjunto de entrenamiento** (Mahalanobis en el espacio estandarizado). El umbral es el percentil 95
   de una validación cruzada agrupada por especie: rechaza el 2 % de las fotos legítimas del conjunto de prueba.
3. **Confianza:** si ninguna familia supera el 65 %, el resultado se marca como incierto.

## Límites

- Solo hay dos clases. Una mariposa de otra familia se parece a estas en color, textura y silueta y **no se rechaza
  de forma fiable**.
- El rechazo de imágenes que no son mariposas es imperfecto: en una prueba con 14 imágenes de ejemplo, rechazó 11 y
  dejó pasar 3 (un gato, un café y un astronauta).
- Entrenado con fotos de catálogo (fondos neutros, mariposas posadas). Con fotos de campo el rendimiento no se ha medido.
- La segmentación puede incluir dedos, hojas o sombras. Además GrabCut es aleatorio (la misma foto puede dar máscaras
  algo distintas entre ejecuciones); la app fija la semilla para que la misma foto dé siempre el mismo resultado.

## Datos

Las fotos de entrenamiento provienen de los catálogos de BioButterfly Database (butterflycatalogs.com) y no se
incluyen en este repositorio.
