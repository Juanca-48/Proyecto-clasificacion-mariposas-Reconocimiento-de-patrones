"""
App de Gradio (Space de Hugging Face): clasifica la familia de una mariposa (Papilionidae vs Pieridae) y rechaza lo que
no parece una mariposa de esas familias. Ofrece hasta tres modelos que se pueden alternar:

  - Clásico (predeterminado): características de textura y color + regresión logística. Lógica en inferencia.py.
  - Embeddings: una red preentrenada (sin modificar) + regresión logística. Lógica en inferencia_embeddings.py.
    Solo aparece si existe modelo/modelo_embeddings.joblib.
  - Embeddings afinados: la misma red con todos sus pesos ajustados a las fotos del catálogo (fine-tuning).
    Solo aparece si existe modelo/modelo_afinado.joblib.
"""
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np

from inferencia import Clasificador

AQUI = Path(__file__).resolve().parent
CLASICO = "Clásico: textura y color (predeterminado)"
EMBEDDINGS = "Embeddings: red preentrenada"
AFINADO = "Embeddings afinados: red ajustada a nuestras fotos"
RUTAS = {EMBEDDINGS: AQUI / "modelo" / "modelo_embeddings.joblib", AFINADO: AQUI / "modelo" / "modelo_afinado.joblib"}

clasico = Clasificador(AQUI / "modelo" / "final_model.joblib", AQUI / "modelo" / "rechazo.joblib")


@lru_cache(maxsize=None)
def motor(nombre):
    if nombre == CLASICO:
        return clasico
    from inferencia_embeddings import ClasificadorEmbeddings   # importa torch solo si se usa
    return ClasificadorEmbeddings(RUTAS[nombre])


MODELOS = [CLASICO]
for nombre, ruta in RUTAS.items():
    if ruta.exists():                       # calentamiento: descarga y carga la red ahora, no en la primera foto del usuario
        try:
            motor(nombre).predecir(np.full((200, 300, 3), 128, np.uint8))
            MODELOS.append(nombre)
        except Exception as e:              # si falla, la app sigue con los demás modelos
            print(f"[AVISO] No se pudo preparar '{nombre}': {e}")

DESCRIPCION = """
# Clasificador de mariposas colombianas
Sube una foto de una mariposa y el modelo indica si es de la familia **Papilionidae** o **Pieridae**. La imagen se
segmenta (se separa la mariposa del fondo) y se clasifica con el modelo que elijas.

**Qué esperar:**
- Solo distingue estas dos familias. Una mariposa de **otra familia se clasificará igualmente como una de ellas**:
  ningún modelo puede detectarla.
- Se entrenó con fotos de catálogo. En fotos de campo o de internet el rendimiento real es menor y no se ha medido bien.
- Mira siempre el **recorte**: si la segmentación falló (incluye dedos, hojas o sombras), el resultado no es fiable.
- Rechaza imágenes que no parecen una mariposa, pero no todas.
"""

ESTILOS = {"aceptada": "✅", "incierta": "⚠️", "rechazada": "⛔"}
NOTAS_RED = {
    "convnext": "Ojo: la red se entrenó con millones de imágenes de ImageNet (que incluye varias clases de mariposas), "
                "así que la cifra puede ser optimista para fotos distintas a las del catálogo.",
    "bioclip": "Ojo: la red se entrenó con millones de fotos de seres vivos y su taxonomía, así que la cifra puede ser "
               "optimista para fotos distintas a las del catálogo.",
    "dinov2": "Ojo: la red se entrenó con millones de imágenes generales, así que la cifra puede ser optimista para fotos "
              "distintas a las del catálogo.",
}


def info_modelo(nombre):
    if nombre == CLASICO:
        return ("**Clásico:** características de textura (GLCM, Haralick, Gabor) y color (histogramas y momentos HSV) + "
                "regresión logística. En fotos de catálogo: balanced accuracy 0,87 (IC 95 % 0,75 a 0,94, 24 especies no vistas).")
    m = motor(nombre)
    met = m.b.get("metricas_test") or {}
    medida = (f" En fotos de catálogo (test, {met['especies']} especies no vistas): balanced accuracy {met['balanced_acc']:.2f}, "
              f"AUC {met['auc']:.2f}.") if met else ""
    if nombre == AFINADO:
        return (f"**Embeddings afinados:** {m.descripcion}. Se ajustan todos los pesos de la red con las fotos del catálogo.{medida} "
                "Ojo: con pocas fotos de entrenamiento, ajustar toda la red puede hacerla menos robusta con fotos distintas a las del catálogo.")
    return f"**Embeddings:** {m.descripcion} + regresión logística.{medida} {NOTAS_RED.get(m.b['red'], '')}"


def clasificar(imagen, nombre):
    if imagen is None:
        raise gr.Error("Sube una imagen.")
    r = motor(nombre).predecir(imagen)
    if r.estado == "aceptada":
        titulo = f"{ESTILOS['aceptada']} **{r.familia}** ({r.probabilidades[r.familia]:.0%})"
    elif r.estado == "incierta":
        titulo = f"{ESTILOS['incierta']} **Resultado incierto.** {r.motivo}\n\nLa familia más probable sería *{r.familia}*."
    else:
        titulo = f"{ESTILOS['rechazada']} **Imagen rechazada.** {r.motivo}"
    partes = [f"Modelo: {nombre}"]
    if r.area_mascara is not None:
        partes.append(f"área de la máscara: {r.area_mascara:.0%} de la imagen")
    if r.distancia is not None:
        partes.append(f"distancia al conjunto de entrenamiento: {r.distancia:.0f} (umbral {r.umbral_distancia:.0f})")
    return r.probabilidades or None, titulo, r.recorte_rgb, " · ".join(partes)


with gr.Blocks(title="Clasificador de mariposas") as demo:
    gr.Markdown(DESCRIPCION)
    modelo = gr.Radio(MODELOS, value=CLASICO, label="Modelo", visible=len(MODELOS) > 1)
    info = gr.Markdown(info_modelo(CLASICO), visible=len(MODELOS) > 1)
    with gr.Row():
        with gr.Column():
            entrada = gr.Image(type="numpy", label="Foto de la mariposa", image_mode="RGB")
            boton = gr.Button("Clasificar", variant="primary")
        with gr.Column():
            veredicto = gr.Markdown()
            probabilidades = gr.Label(label="Probabilidad por familia", num_top_classes=2)
            recorte = gr.Image(label="Segmentación (lo que ve el modelo)", interactive=False)
            detalle = gr.Markdown()
    modelo.change(info_modelo, modelo, info)
    boton.click(clasificar, [entrada, modelo], [probabilidades, veredicto, recorte, detalle], api_name="clasificar")

if __name__ == "__main__":
    demo.launch()
