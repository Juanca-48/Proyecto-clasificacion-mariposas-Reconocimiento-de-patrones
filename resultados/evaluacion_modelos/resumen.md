# Evaluación de los modelos del proyecto

Todos los modelos se evalúan sobre **las mismas 122 imágenes de test** (43 Papilionidae y 79 Pieridae) de **24 especies que ningún modelo vio al entrenar**. Clase positiva: Pieridae; umbral de decisión 0,5. Los intervalos de confianza (IC 95 %) salen de un bootstrap de **especies** (2000 remuestras): las fotos de una misma especie no son independientes, así que remuestrear imágenes sueltas daría intervalos demasiado estrechos.

## 1. Resultado principal

| modelo | exactitud | balanced accuracy [IC 95 %] | F1 macro [IC 95 %] | MCC [IC 95 %] | AUC [IC 95 %] | errores |
|---|---|---|---|---|---|---|
| Clásico LogReg (rembg) | 0.861 | 0.850 [0.702, 0.928] | 0.848 [0.695, 0.922] | 0.696 [0.392, 0.847] | 0.930 [0.845, 0.981] | 17 |
| Clásico LogReg (GrabCut, desplegado) | 0.877 | 0.868 [0.749, 0.933] | 0.866 [0.735, 0.925] | 0.732 [0.478, 0.851] | 0.951 [0.887, 0.985] | 15 |
| Random Forest (rembg) | 0.811 | 0.817 [0.677, 0.922] | 0.802 [0.622, 0.913] | 0.613 [0.304, 0.828] | 0.892 [0.745, 0.981] | 23 |
| ConvNeXt embeddings | 0.951 | 0.962 [0.907, 1.000] | 0.948 [0.838, 1.000] | 0.901 [0.721, 1.000] | 0.997 [0.987, 1.000] | 6 |
| ConvNeXt afinado | 0.984 | 0.987 [0.958, 1.000] | 0.982 [0.927, 1.000] | 0.965 [0.863, 1.000] | 0.999 [0.997, 1.000] | 2 |

### Conclusiones (se calculan con los números de arriba)

- **ConvNeXt afinado** frente a **ConvNeXt embeddings**: +0.025 de balanced accuracy [+0.000, +0.053], McNemar p = 0.219 → no concluyente (el IC incluye el cero).
- **ConvNeXt afinado** frente a **Clásico LogReg (GrabCut, desplegado)**: +0.119 de balanced accuracy [+0.057, +0.238], McNemar p = 0.001 → firme a favor de ConvNeXt afinado.
- **ConvNeXt embeddings** frente a **Clásico LogReg (GrabCut, desplegado)**: +0.094 de balanced accuracy [+0.030, +0.219], McNemar p = 0.035 → firme a favor de ConvNeXt embeddings.
- **Clásico LogReg (rembg)** frente a **Clásico LogReg (GrabCut, desplegado)**: -0.018 de balanced accuracy [-0.056, +0.012], McNemar p = 0.727 → no concluyente (el IC incluye el cero).
- **Clásico LogReg (rembg)** frente a **Random Forest (rembg)**: +0.033 de balanced accuracy [-0.095, +0.169], McNemar p = 0.238 → no concluyente (el IC incluye el cero).
- **Falsos positivos de las redes** (Papilionidae clasificadas como Pieridae): ConvNeXt embeddings 0, ConvNeXt afinado 0. Los errores de las redes son Pieridae que se confunden con Papilionidae.
- Los modelos clásicos pierden sobre todo en los **mimetismos**: Pieridae oscuras con manchas (*Archonias*, *Perrhybris*, *Dismorphia*) y Papilionidae pálidas (*Protesilaus*).

- **Mejor modelo en test:** ConvNeXt afinado (0.987 de balanced accuracy, IC [0.958, 1.000]).
- **Los intervalos son anchos** (con 24 especies de test): varios modelos se solapan, y las comparaciones de la sección 4 dicen cuáles diferencias se sostienen.
- Gráfico: `metricas_con_ic.png`. Tabla completa (con todas las métricas): `metricas_test.csv`.

## 2. Matrices de confusión

Filas: clase real; columnas: clase predicha. Gráfico: `matrices_confusion.png`.

| modelo | TN (Papil. bien) | FP (Papil. → Pier.) | FN (Pier. → Papil.) | TP (Pier. bien) |
|---|---|---|---|---|
| Clásico LogReg (rembg) | 35 | 8 | 9 | 70 |
| Clásico LogReg (GrabCut, desplegado) | 36 | 7 | 8 | 71 |
| Random Forest (rembg) | 36 | 7 | 16 | 63 |
| ConvNeXt embeddings | 43 | 0 | 6 | 73 |
| ConvNeXt afinado | 43 | 0 | 2 | 77 |

Lectura: **FP** son Papilionidae clasificadas como Pieridae; **FN**, Pieridae clasificadas como Papilionidae.

## 3. Precisión, recall y F1 por clase

| modelo | clase | precision | recall | F1 | soporte |
|---|---|---|---|---|---|
| Clásico LogReg (rembg) | Papilionidae | 0.795 | 0.814 | 0.805 | 43 |
| Clásico LogReg (rembg) | Pieridae | 0.897 | 0.886 | 0.892 | 79 |
| Clásico LogReg (GrabCut, desplegado) | Papilionidae | 0.818 | 0.837 | 0.828 | 43 |
| Clásico LogReg (GrabCut, desplegado) | Pieridae | 0.910 | 0.899 | 0.904 | 79 |
| Random Forest (rembg) | Papilionidae | 0.692 | 0.837 | 0.758 | 43 |
| Random Forest (rembg) | Pieridae | 0.900 | 0.797 | 0.846 | 79 |
| ConvNeXt embeddings | Papilionidae | 0.878 | 1.000 | 0.935 | 43 |
| ConvNeXt embeddings | Pieridae | 1.000 | 0.924 | 0.961 | 79 |
| ConvNeXt afinado | Papilionidae | 0.956 | 1.000 | 0.977 | 43 |
| ConvNeXt afinado | Pieridae | 1.000 | 0.975 | 0.987 | 79 |

F1 macro (promedio de las dos clases) y F1 ponderado (por soporte) están en `metricas_test.csv`. La balanced accuracy es el recall medio de las dos clases, y por eso no la infla la clase mayoritaria (Pieridae es el 65% del test).

## 4. Comparaciones entre modelos (pareadas, mismas imágenes)

Diferencia **A − B**, con IC de bootstrap por especies. `solo_A_acierta` / `solo_B_acierta` cuentan las imágenes en que un modelo acierta y el otro falla; `McNemar_p` es la prueba exacta de McNemar sobre esos desacuerdos. Una diferencia solo es firme si su IC no incluye el cero.

| A | B | dif. balanced acc. [IC 95 %] | dif. F1 macro [IC 95 %] | solo_A_acierta | solo_B_acierta | McNemar_p |
|---|---|---|---|---|---|---|
| ConvNeXt afinado | ConvNeXt embeddings | +0.025 [+0.000, +0.053] | +0.035 [+0.000, +0.090] | 5 | 1 | 0.219 |
| ConvNeXt afinado | Clásico LogReg (rembg) | +0.137 [+0.057, +0.287] | +0.134 [+0.056, +0.282] | 16 | 1 | 0.000 |
| ConvNeXt afinado | Clásico LogReg (GrabCut, desplegado) | +0.119 [+0.057, +0.238] | +0.116 [+0.063, +0.232] | 14 | 1 | 0.001 |
| ConvNeXt embeddings | Clásico LogReg (rembg) | +0.112 [+0.024, +0.266] | +0.099 [+0.007, +0.242] | 15 | 4 | 0.019 |
| ConvNeXt embeddings | Clásico LogReg (GrabCut, desplegado) | +0.094 [+0.030, +0.219] | +0.082 [+0.017, +0.189] | 12 | 3 | 0.035 |
| Clásico LogReg (rembg) | Clásico LogReg (GrabCut, desplegado) | -0.018 [-0.056, +0.012] | -0.018 [-0.059, +0.015] | 3 | 5 | 0.727 |
| Clásico LogReg (rembg) | Random Forest (rembg) | +0.033 [-0.095, +0.169] | +0.046 [-0.070, +0.178] | 12 | 6 | 0.238 |

## 5. Probabilidades y calibración

Brier y log-loss (más bajos, mejor), ECE (error de calibración, más bajo mejor), kappa de Cohen y precisión promedio. Gráficos: `curvas_roc_pr.png` y `calibracion.png`.

El mejor Brier es el de **ConvNeXt afinado** y el menor ECE, el de **ConvNeXt embeddings**. Las tres métricas no siempre coinciden: un modelo puede acertar más y tener probabilidades más extremas (peor calibradas). Con 122 imágenes, el ECE es ruidoso.

| modelo | Brier | log_loss | ECE | kappa | precision_promedio |
|---|---|---|---|---|---|
| Clásico LogReg (rembg) | 0.114 | 0.688 | 0.107 | 0.696 | 0.966 |
| Clásico LogReg (GrabCut, desplegado) | 0.091 | 0.296 | 0.057 | 0.732 | 0.976 |
| Random Forest (rembg) | 0.132 | 0.398 | 0.065 | 0.606 | 0.950 |
| ConvNeXt embeddings | 0.034 | 0.127 | 0.033 | 0.896 | 0.998 |
| ConvNeXt afinado | 0.019 | 0.103 | 0.077 | 0.964 | 1.000 |

## 6. Errores

- **Imágenes que fallan los 5 modelos:** 1 → 261_Archonias_brassolis_approximata.
- **Errores del modelo afinado (2):** 261_Archonias_brassolis_approximata (real Pieridae, P(Pieridae)=0.25), 263_Archonias_brassolis_approximata (real Pieridae, P(Pieridae)=0.45).
- Detalle por imagen: `errores_por_imagen.csv`; por especie: `errores_por_especie.csv`.

Errores por especie (solo las que fallan en algún modelo):

| especie | familia | imagenes | Clásico LogReg (rembg) | Clásico LogReg (GrabCut, desplegado) | Random Forest (rembg) | ConvNeXt embeddings | ConvNeXt afinado |
|---|---|---|---|---|---|---|---|
| archonias brassolis | Pieridae | 7 | 1 | 2 | 7 | 4 | 2 |
| dismorphia theucharila | Pieridae | 14 | 2 | 1 | 3 | 1 | 0 |
| perrhybris lorena | Pieridae | 6 | 3 | 2 | 3 | 1 | 0 |
| dismorphia medora | Pieridae | 2 | 1 | 1 | 1 | 0 | 0 |
| eurytides columbus | Papilionidae | 4 | 1 | 1 | 0 | 0 | 0 |
| heraclides isidorus | Papilionidae | 3 | 1 | 1 | 1 | 0 | 0 |
| heraclides thoas | Papilionidae | 13 | 1 | 1 | 1 | 0 | 0 |
| leptophobia eleone | Pieridae | 8 | 1 | 1 | 1 | 0 | 0 |
| leptophobia penthica | Pieridae | 5 | 1 | 1 | 1 | 0 | 0 |
| mimoides xeniades | Papilionidae | 2 | 1 | 1 | 0 | 0 | 0 |
| parides orellana | Papilionidae | 4 | 0 | 0 | 3 | 0 | 0 |
| protesilaus protesilaus | Papilionidae | 5 | 4 | 3 | 2 | 0 | 0 |

## 7. Rechazo de imágenes (las tres capas) sobre el test

Fracción de imágenes legítimas que el sistema acepta, rechaza o marca como inciertas, y exactitud de lo aceptado. Un buen rechazo filtra errores sin descartar demasiadas fotos buenas.

| modelo | rechazadas_area | rechazadas_distancia | inciertas | aceptadas | pct_aceptadas | exactitud_sin_filtro | exactitud_aceptadas | errores_que_se_filtran | errores_aceptados |
|---|---|---|---|---|---|---|---|---|---|
| ConvNeXt embeddings | 0 | 6 | 4 | 112 | 0.918 | 0.951 | 0.964 | 2 | 4 |
| ConvNeXt afinado | 0 | 10 | 4 | 108 | 0.885 | 0.984 | 0.991 | 1 | 1 |
| Clásico LogReg (rembg) | 0 | 1 | 4 | 117 | 0.959 | 0.861 | 0.889 | 4 | 13 |

No se incluye el clásico desplegado: su rechazo se calibró con máscaras de GrabCut, que ya no están en disco. El de rembg (arriba) usa los umbrales de `resultados/rechazo/`.

## 8. Validación cruzada frente al test

La CV (agrupada por especie, sobre train + val) y el test dan una idea de cuánto cambia el rendimiento con datos nuevos. Gráfico: `cv_comparacion.png`; tabla completa: `cv_comparacion.csv`.

| modelo | tipo | bal_acc_cv | desv | auc_cv | bal_acc_test | gap_cv_menos_test |
|---|---|---|---|---|---|---|
| Regresión logística | clásico (textura y color) | 0.889 | 0.015 | 0.955 | 0.850 | 0.039 |
| Random Forest (referencia) | clásico (textura y color) | 0.858 | 0.052 | 0.929 | 0.817 | 0.041 |
| convnext / recorte | red preentrenada | 0.979 | 0.002 | 0.997 | 0.962 | 0.017 |

El modelo afinado no tiene CV (reentrenar una red 5 veces era demasiado costoso); se evaluó con una partición fija y varias semillas (sección 9).

## 9. Fine-tuning: semillas y curvas

| semilla | val_bal_acc | val_auc | val_perdida | val_errores | test_bal_acc | test_auc | test_perdida | test_errores |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.994 | 1.000 | 0.072 | 1 | 0.981 | 0.999 | 0.110 | 3 |
| 1 | 0.994 | 1.000 | 0.080 | 1 | 0.981 | 0.999 | 0.114 | 3 |
| 2 | 0.994 | 1.000 | 0.070 | 1 | 0.987 | 0.999 | 0.103 | 2 |

Con un número fijo de épocas y sin parada temprana, las tres semillas dan resultados casi iguales: el afinado **varía poco entre semillas**. Se despliega la de mejor validación. La **validación está saturada** (1 error de 122 en las tres semillas), así que no sirve para distinguir modelos entre sí; el test es lo único que discrimina, y es pequeño. Curvas: `afinado_curvas.png`.

## 10. Fotos de internet (8 fotos fuera del catálogo)

Modelos tal como están en la app: el clásico desplegado (GrabCut) y el afinado. "acierta" = aceptada y con la familia correcta.

| foto | real | clásico (desplegado): estado | clásico (desplegado): familia | clásico (desplegado): P(Pieridae) | clásico (desplegado): acierta | afinado: estado | afinado: familia | afinado: P(Pieridae) | afinado: acierta |
|---|---|---|---|---|---|---|---|---|---|
| Papilionidae1.jfif | Papilionidae | aceptada | Papilionidae | 0.050 | True | aceptada | Papilionidae | 0.016 | True |
| papilionidae2.webp | Papilionidae | aceptada | Pieridae | 0.875 | False | aceptada | Papilionidae | 0.015 | True |
| Papilionidae3.JPG | Papilionidae | incierta | Pieridae | 0.567 | False | aceptada | Papilionidae | 0.010 | True |
| papilionidae4.jfif | Papilionidae | aceptada | Papilionidae | 0.075 | True | aceptada | Papilionidae | 0.016 | True |
| Pieridae1.jpeg | Pieridae | rechazada | - |  | False | aceptada | Pieridae | 0.905 | True |
| Pieridae2.webp | Pieridae | rechazada | - |  | False | aceptada | Pieridae | 0.913 | True |
| Pieridae3.jpg | Pieridae | aceptada | Papilionidae | 0.033 | False | aceptada | Pieridae | 0.885 | True |
| Pieridae4.jpg | Pieridae | aceptada | Papilionidae | 0.163 | False | aceptada | Pieridae | 0.873 | True |

Con 8 fotos no se puede estimar una exactitud con precisión; sirve para ver si hay un desajuste evidente con fotos no tomadas para el catálogo.

## 11. Cautelas para interpretar los números

- **El test es pequeño:** 24 especies y 122 imágenes. Una diferencia de un par de errores puede ser ruido; por eso se dan los IC y las pruebas pareadas.
- **El test ya se ha consultado varias veces** a lo largo del proyecto: al comparar modelos y al decidir qué ofrecer en la app (el clásico, el de embeddings y el afinado se vieron en test antes de decidir). Las cifras son, por tanto, algo optimistas.
- **ConvNeXt se preentrenó con ImageNet**, que incluye clases de mariposas: la comparación con el modelo clásico es justa en el protocolo, pero no en los datos de preentrenamiento.
- **Las dos familias vienen de catálogos distintos**; por eso los modelos de red usan el recorte segmentado y no la foto completa (riesgo de aprender el fondo o la iluminación del catálogo).
- **El afinado se entrenó solo con train** (568 imágenes), a diferencia de los demás, que se reentrenaron con train + val (690).
- Las fotos de internet son solo 8 y todas se aceptan o fallan por razones distintas: no sustituyen una evaluación con fotos de campo.

## 12. Índice de archivos

| Archivo | Contenido |
|---|---|
| `metricas_test.csv` | todas las métricas por modelo, con IC |
| `matrices_confusion.csv` / `.png` | matrices de confusión |
| `por_clase.csv` | precisión, recall y F1 por clase |
| `comparaciones_pareadas.csv` | diferencias entre modelos con IC y McNemar |
| `errores_por_imagen.csv`, `errores_por_especie.csv` | análisis de errores |
| `cobertura_rechazo.csv` | aceptadas, rechazadas e inciertas |
| `cv_comparacion.csv` / `.png` | validación cruzada |
| `afinado_semillas.csv`, `afinado_curvas.png` | fine-tuning |
| `fotos_internet.csv` | fotos de Pruebas/ |
| `curvas_roc_pr.png`, `calibracion.png`, `metricas_con_ic.png` | gráficos |

Generado con `python scripts/evaluacion/evaluar_modelos.py`.
