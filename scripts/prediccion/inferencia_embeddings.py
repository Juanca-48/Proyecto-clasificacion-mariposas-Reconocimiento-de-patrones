"""
inferencia_embeddings.py
------------------------
Segundo motor de clasificación de la app: en vez de características clásicas de textura y color, usa los embeddings
de una red preentrenada (BioCLIP, DINOv2 o ConvNeXt) y una regresión logística. Devuelve el mismo `Resultado` que
inferencia.Clasificador, así que la app puede alternar entre los dos sin cambiar nada más.

Flujo (la red, la entrada y los umbrales vienen guardados en el archivo que crea entrenar_embeddings.py):
  1. preparar     igual que el motor clásico.
  2. entrada      "recorte": se segmenta y se embebe la mariposa sobre fondo blanco (capa 1 de rechazo: área de la máscara);
                  "original": se embebe la foto completa.
  3. embedding    red preentrenada -> vector L2-normalizado.
  4. Capa 2       distancia de Mahalanobis al conjunto de entrenamiento, en el espacio estandarizado del embedding.
  5. clasificar   regresión logística. Capa 3: si la mayor probabilidad no llega al umbral, el resultado es "incierto".

torch y la librería de la red solo se importan al primera clasificación, para que arrancar la app sea rápido.
"""
from __future__ import annotations

# Localiza rutas.py (en scripts/, o junto a este archivo en la app) y agrega al path la carpeta de cada proceso
import sys
from pathlib import Path
sys.path[:0] = [str(p) for p in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]) if (p / "rutas.py").exists()][:1]
import rutas  # noqa: E402,F401


from pathlib import Path

import cv2
import joblib
import numpy as np
from PIL import Image
from sklearn.preprocessing import normalize

import inferencia as I


class ClasificadorEmbeddings:
    def __init__(self, ruta: Path):
        self.b = joblib.load(ruta)
        self.clases = list(self.b["clases"])
        self._embed = None

    @property
    def descripcion(self) -> str:
        red = f"{self.b['red']} afinada" if self.b.get("tipo") == "afinado" else self.b["red"]
        return f"{red} sobre {'recortes segmentados' if self.b['entrada'] == 'recorte' else 'la foto completa'}"

    def _embedder(self):
        if self._embed is None:
            if self.b.get("tipo") == "afinado":
                self._embed = self._construir_afinada()
            else:
                from embeddings import construir   # importa torch y la red solo al usar este motor
                self._embed = construir(self.b["red"])
        return self._embed

    def _construir_afinada(self):
        """Red afinada (afinar.py): devuelve (embedding, logits) de una imagen PIL."""
        import timm
        import torch
        b = self.b
        m = timm.create_model(b["arch"], pretrained=False, num_classes=2)
        m.load_state_dict({k: v.float() for k, v in b["pesos"].items()})
        m.eval()
        tf = timm.data.create_transform(**timm.data.resolve_data_config({}, model=m))

        def f(img):
            with torch.no_grad():   # el modo sin gradiente es por hilo y Gradio atiende cada petición en otro hilo
                feats = m.forward_head(m.forward_features(tf(img).unsqueeze(0)), pre_logits=True)
                return feats.numpy(), m.head.fc(feats).numpy()
        return f

    def analizar(self, img_rgb: np.ndarray, lado: int | None = None, segmentador: str | None = None) -> dict:
        """Calcula el embedding, la distancia y las probabilidades SIN decidir si se rechaza."""
        b = self.b
        img = I.preparar(img_rgb, lado)
        if b["entrada"] == "recorte":
            mask, cutout = I.segmentar(img, segmentador or b.get("segmentador", "rembg"))
            area, rgb = float((mask > 0).mean()), cv2.cvtColor(cutout, cv2.COLOR_BGR2RGB)
            vacia = (mask > 0).sum() < 50
        else:
            mask, area, rgb, vacia = None, None, cv2.cvtColor(img, cv2.COLOR_BGR2RGB), False
        a = dict(area=area, mascara=mask, recorte_rgb=rgb, distancia=None, probabilidades=None)
        if vacia:
            return a
        if b.get("tipo") == "afinado":   # la cabeza de la propia red da las probabilidades; la distancia se mide en su espacio
            feats, logits = self._embedder()(Image.fromarray(rgb))
            a["distancia"] = float(b["covarianza"].mahalanobis(feats)[0])
            p = np.exp(logits[0] - logits[0].max())
            a["probabilidades"] = {c: float(x) for c, x in zip(self.clases, p / p.sum())}
            return a
        e = normalize(self._embedder()([Image.fromarray(rgb)]))
        m = b["modelo"]
        z = m.named_steps["s"].transform(m.named_steps["i"].transform(e))
        a["distancia"] = float(b["covarianza"].mahalanobis(z)[0])
        a["probabilidades"] = {c: float(p) for c, p in zip(self.clases, m.predict_proba(e)[0])}
        return a

    def decidir(self, a: dict) -> I.Resultado:
        """Aplica las tres capas de rechazo a un análisis."""
        b = self.b
        base = dict(area_mascara=a["area"], mascara=a["mascara"], recorte_rgb=a["recorte_rgb"],
                    umbral_distancia=b["umbral_distancia"], distancia=a["distancia"])
        if a["area"] is not None and not (b["area_min"] <= a["area"] <= b["area_max"]):
            return I.Resultado("rechazada", f"La segmentación no aisló una mariposa del tamaño esperado "
                               f"(la máscara cubre el {a['area']:.0%} de la imagen; en las fotos de entrenamiento va de "
                               f"{b['area_min']:.0%} a {b['area_max']:.0%}).", **base)
        if a["probabilidades"] is None:
            return I.Resultado("rechazada", "No se detectó ningún objeto en la imagen.", **base)
        if a["distancia"] > b["umbral_distancia"]:
            return I.Resultado("rechazada", "La imagen no se parece a las fotos de mariposas Papilionidae o Pieridae "
                               "con las que se entrenó el modelo.", **base)
        probs = a["probabilidades"]
        familia = max(probs, key=probs.get)
        if probs[familia] < b["umbral_confianza"]:
            return I.Resultado("incierta", "El modelo no está seguro: ninguna familia supera el "
                               f"{b['umbral_confianza']:.0%} de confianza.", familia, probs, **base)
        return I.Resultado("aceptada", "", familia, probs, **base)

    def predecir(self, img_rgb: np.ndarray, lado: int | None = None, segmentador: str | None = None) -> I.Resultado:
        return self.decidir(self.analizar(img_rgb, lado, segmentador))
