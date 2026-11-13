"""
inferencia.py
-------------
Clasifica UNA imagen (familia Papilionidae vs Pieridae) con el modelo final, y rechaza lo que no
parece una mariposa de estas familias. La usan la app de Hugging Face y calibrar_rechazo.py.

Flujo, igual que en el entrenamiento (Segmentacion.py -> texturas.py + color_features.py -> modelo):
  1. preparar    RGB -> BGR, y reducir si el lado mayor supera MAX_SIDE (la textura depende de la escala
                 en píxeles; las fotos del catálogo miden hasta 800 px).
  2. segmentar   GrabCut. Capa 1 de rechazo: el área de la máscara debe estar en el rango visto en el
                 entrenamiento (si no, la segmentación no aisló un objeto del tamaño esperado).
  3. features    textura + color (147), en el orden que espera el modelo.
  4. Capa 2      distancia de Mahalanobis al conjunto de entrenamiento (espacio estandarizado): si es
                 mayor que el umbral, no se parece a las fotos con las que se entrenó.
  5. clasificar  regresión logística. Capa 3: si la mayor probabilidad no llega al umbral de confianza,
                 el resultado es "incierto".

Límite conocido: una mariposa de OTRA familia se parece a las del entrenamiento en color, textura y
silueta, así que no se rechaza de forma fiable.
"""
from __future__ import annotations

# Localiza rutas.py (en scripts/, o junto a este archivo en la app) y agrega al path la carpeta de cada proceso
import sys
from pathlib import Path
sys.path[:0] = [str(p) for p in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]) if (p / "rutas.py").exists()][:1]
import rutas  # noqa: E402,F401


from dataclasses import dataclass, field
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd

import color_features
import Segmentacion
import Segmentacion_rembg   # importa rembg solo al usarlo
import texturas

MAX_SIDE = 800   # lado mayor máximo de las fotos del catálogo (mediana 640)


def preparar(img_rgb: np.ndarray, lado: int | None = None) -> np.ndarray:
    """RGB / RGBA / gris (uint8) -> BGR de 3 canales.

    Escala: con `lado`, el lado mayor pasa a medir exactamente `lado` píxeles (se reduce o se amplía);
    sin `lado`, solo se reduce si supera MAX_SIDE."""
    img = np.asarray(img_rgb)
    if img.dtype == bool:
        img = img.astype(np.uint8) * 255
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    objetivo = lado if lado else (MAX_SIDE if max(h, w) > MAX_SIDE else None)
    if objetivo and max(h, w) != objetivo:
        s = objetivo / max(h, w)
        bgr = cv2.resize(bgr, (round(w * s), round(h * s)),
                         interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    return bgr


def segmentar(img_bgr: np.ndarray, segmentador: str = "grabcut"):
    """Máscara 0/255 y recorte de la mariposa sobre fondo blanco (como en Segmentacion.py).
    segmentador: "grabcut" (el del entrenamiento) o "rembg"."""
    # GrabCut usa el generador aleatorio global de OpenCV; sin fijar la semilla, la misma foto puede dar
    # máscaras distintas (IoU hasta 0,86 entre dos llamadas). Con semilla fija la salida es reproducible.
    cv2.setRNGSeed(0)
    mask = Segmentacion_rembg.segment_rembg(img_bgr) if segmentador == "rembg" else Segmentacion.segment_grabcut(img_bgr)
    cutout = np.full_like(img_bgr, 255)
    cutout[mask == 255] = img_bgr[mask == 255]
    return mask, cutout


def extraer_features(img_bgr: np.ndarray, mask: np.ndarray) -> dict | None:
    """Textura + color con las mismas funciones del entrenamiento. None si la máscara es diminuta."""
    tex = texturas.features_from_arrays(img_bgr, mask)
    col = color_features.features_from_arrays(img_bgr, mask)
    if tex is None or col is None:
        return None
    return {**tex, **col}


@dataclass
class Resultado:
    estado: str                       # "aceptada" | "incierta" | "rechazada"
    motivo: str                       # explicación para mostrar al usuario
    familia: str | None = None        # predicción (también si es incierta)
    probabilidades: dict = field(default_factory=dict)
    area_mascara: float | None = None
    distancia: float | None = None    # Mahalanobis al cuadrado
    umbral_distancia: float | None = None
    mascara: np.ndarray | None = None
    recorte_rgb: np.ndarray | None = None


class Clasificador:
    def __init__(self, modelo: Path, rechazo: Path):
        b = joblib.load(modelo)
        self.pipeline, self.columnas, self.clases = b["model"], list(b["features"]), list(b["classes"])
        self.rechazo = joblib.load(rechazo)

    def estandarizar(self, X: pd.DataFrame) -> np.ndarray:
        """Espacio en el que se mide la distancia: el mismo imputador y escalador del modelo."""
        return self.pipeline.named_steps["scaler"].transform(self.pipeline.named_steps["imputer"].transform(X))

    def analizar(self, img_rgb: np.ndarray, lado: int | None = None, segmentador: str | None = None) -> dict:
        """Segmenta, extrae features y mide distancia y probabilidades, SIN decidir si se rechaza.
        Útil para ver qué diría el modelo con y sin rechazo (la distancia y las probabilidades son None
        si la máscara es demasiado pequeña para calcular features)."""
        # por defecto, el segmentador con el que se calibró el modelo (queda guardado en rechazo.joblib)
        segmentador = segmentador or self.rechazo.get("segmentador", "grabcut")
        img = preparar(img_rgb, lado)
        mask, cutout = segmentar(img, segmentador)
        a = dict(area=float((mask > 0).mean()), mascara=mask, recorte_rgb=cv2.cvtColor(cutout, cv2.COLOR_BGR2RGB),
                 distancia=None, probabilidades=None)
        feats = extraer_features(img, mask)
        if feats is not None:
            X = pd.DataFrame([feats])[self.columnas]
            a["distancia"] = float(self.rechazo["covarianza"].mahalanobis(self.estandarizar(X))[0])
            a["probabilidades"] = {c: float(p) for c, p in zip(self.clases, self.pipeline.predict_proba(X)[0])}
        return a

    def decidir(self, a: dict) -> Resultado:
        """Aplica las tres capas de rechazo a un análisis."""
        r = self.rechazo
        base = dict(area_mascara=a["area"], mascara=a["mascara"], recorte_rgb=a["recorte_rgb"],
                    umbral_distancia=r["umbral_distancia"], distancia=a["distancia"])
        if not (r["area_min"] <= a["area"] <= r["area_max"]):
            return Resultado("rechazada", f"La segmentación no aisló una mariposa del tamaño esperado "
                             f"(la máscara cubre el {a['area']:.0%} de la imagen; en las fotos de entrenamiento va de "
                             f"{r['area_min']:.0%} a {r['area_max']:.0%}).", **base)
        if a["probabilidades"] is None:
            return Resultado("rechazada", "No se detectó ningún objeto en la imagen.", **base)
        if a["distancia"] > r["umbral_distancia"]:
            return Resultado("rechazada", "La imagen no se parece a las fotos de mariposas Papilionidae o Pieridae "
                             "con las que se entrenó el modelo.", **base)
        probs = a["probabilidades"]
        familia = max(probs, key=probs.get)
        if probs[familia] < r["umbral_confianza"]:
            return Resultado("incierta", "El modelo no está seguro: ninguna familia supera el "
                             f"{r['umbral_confianza']:.0%} de confianza.", familia, probs, **base)
        return Resultado("aceptada", "", familia, probs, **base)

    def predecir(self, img_rgb: np.ndarray, lado: int | None = None, segmentador: str | None = None) -> Resultado:
        return self.decidir(self.analizar(img_rgb, lado, segmentador))
