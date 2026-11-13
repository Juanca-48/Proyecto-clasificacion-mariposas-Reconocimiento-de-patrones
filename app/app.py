"""
App de Gradio (Space de Hugging Face): clasifica la familia de una mariposa (Papilionidae vs Pieridae)
y rechaza lo que no parece una mariposa de esas familias. La lógica está en inferencia.py.
"""
from pathlib import Path

import gradio as gr

from inferencia import Clasificador

AQUI = Path(__file__).resolve().parent
clasificador = Clasificador(AQUI / "modelo" / "final_model.joblib", AQUI / "modelo" / "rechazo.joblib")

DESCRIPCION = """
# Clasificador de mariposas colombianas
Sube una foto de una mariposa y el modelo indica si es de la familia **Papilionidae** o **Pieridae**.
La imagen se segmenta (se separa la mariposa del fondo) y se clasifica con características de **textura y color**
(regresión logística).

**Qué esperar:**
- Solo distingue estas dos familias. Una mariposa de **otra familia se clasificará igualmente como una de ellas**:
  el modelo no puede detectarla.
- Se entrenó con fotos de catálogo. En fotos de campo el rendimiento real no se ha medido.
- En fotos de catálogo acierta alrededor del 88 % (balanced accuracy 0,87, IC 95 % 0,75 a 0,94, en 24 especies no vistas).
- Rechaza imágenes que no parecen una mariposa, pero no todas: en una prueba pequeña con 14 imágenes que no eran mariposas,
  rechazó 11 y dejó pasar 3 (un gato, un café y un astronauta).
- Mira siempre el **recorte**: si la segmentación falló (incluye dedos, hojas o sombras), el resultado no es fiable.
"""

ESTILOS = {"aceptada": "✅", "incierta": "⚠️", "rechazada": "⛔"}


def clasificar(imagen):
    if imagen is None:
        raise gr.Error("Sube una imagen.")
    r = clasificador.predecir(imagen)
    if r.estado == "aceptada":
        titulo = f"{ESTILOS['aceptada']} **{r.familia}** ({r.probabilidades[r.familia]:.0%})"
    elif r.estado == "incierta":
        titulo = f"{ESTILOS['incierta']} **Resultado incierto.** {r.motivo}\n\nLa familia más probable sería *{r.familia}*."
    else:
        titulo = f"{ESTILOS['rechazada']} **Imagen rechazada.** {r.motivo}"
    detalle = f"Área de la máscara: {r.area_mascara:.0%} de la imagen"
    if r.distancia is not None:
        detalle += f" · distancia al conjunto de entrenamiento: {r.distancia:.0f} (umbral {r.umbral_distancia:.0f})"
    return r.probabilidades or None, titulo, r.recorte_rgb, detalle


with gr.Blocks(title="Clasificador de mariposas") as demo:
    gr.Markdown(DESCRIPCION)
    with gr.Row():
        with gr.Column():
            entrada = gr.Image(type="numpy", label="Foto de la mariposa", image_mode="RGB")
            boton = gr.Button("Clasificar", variant="primary")
        with gr.Column():
            veredicto = gr.Markdown()
            probabilidades = gr.Label(label="Probabilidad por familia", num_top_classes=2)
            recorte = gr.Image(label="Segmentación (lo que ve el modelo)", interactive=False)
            detalle = gr.Markdown()
    boton.click(clasificar, entrada, [probabilidades, veredicto, recorte, detalle], api_name="clasificar")

if __name__ == "__main__":
    demo.launch()
