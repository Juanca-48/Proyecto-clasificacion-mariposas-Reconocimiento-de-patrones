#!/usr/bin/env python3
"""
formas.py
---------
Descriptores de FORMA de la silueta de la mariposa, calculados sobre las máscaras binarias
de resultados/segmentacion/<dataset>/masks/ (segmentación v1). Todos son invariantes a escala y
traslación (los de Hu y la excentricidad también a rotación), así que no dependen del
tamaño de la foto ni de dónde esté la mariposa. No se incluye el área absoluta.

Por imagen (16 columnas, prefijo shape_):
  solidity        área / área del casco convexo (huecos entre alas, colas)
  extent          área / área del rectángulo mínimo rotado
  aspect_ratio    lado largo / lado corto del rectángulo mínimo rotado
  circularity     4*pi*área / perímetro^2
  convexity       perímetro del casco / perímetro (qué tan rugoso es el borde)
  eccentricity    excentricidad de la elipse equivalente (momentos centrales)
  hu1..hu7        momentos de Hu en escala log con signo
  n_defects       muescas del contorno más profundas que 3 % de sqrt(área) (colas, lóbulos)
  max_defect      profundidad de la muesca mayor / sqrt(área)
  mean_defect     profundidad media de las muescas / sqrt(área)

Las máscaras v1 pueden incluir dedos, pétalos o sombras; esos objetos se reflejan en la forma.

Salida: resultados/segmentacion/shape_features_all.csv con (filename, dataset, shape_*).

Uso:
    python formas.py
    python formas.py --root "ruta\\segmentacion" --datasets Dataset_papilionidae Dataset_pieridae
    python formas.py --selftest
"""
from __future__ import annotations

# Localiza rutas.py (en scripts/, o junto a este archivo en la app) y agrega al path la carpeta de cada proceso
import sys
from pathlib import Path
sys.path[:0] = [str(p) for p in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]) if (p / "rutas.py").exists()][:1]
import rutas  # noqa: E402,F401


import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from rutas import SEGMENTATION_DIR as SEGMENTATION_ROOT
DEFAULT_DATASETS = ["Dataset_papilionidae", "Dataset_pieridae"]
VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

MIN_AREA = 50            # máscaras más pequeñas se descartan (segmentación fallida)
DEFECT_MIN_DEPTH = 0.03  # profundidad mínima de una muesca, relativa a sqrt(área)


def shape_features(mask: np.ndarray) -> dict | None:
    """Descriptores de forma de una máscara 0/255. None si no hay silueta utilizable."""
    mask = (mask > 127).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < MIN_AREA:
        return None
    perim = cv2.arcLength(cnt, True)
    hull = cv2.convexHull(cnt)
    hull_area, hull_perim = cv2.contourArea(hull), cv2.arcLength(hull, True)
    (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
    if min(rw, rh) < 1:
        return None

    # momentos de la máscara completa (más estables que los del polígono)
    m = cv2.moments(mask, binaryImage=True)
    common = np.sqrt(4 * m["mu11"] ** 2 + (m["mu20"] - m["mu02"]) ** 2)
    l1, l2 = 0.5 * (m["mu20"] + m["mu02"] + common), 0.5 * (m["mu20"] + m["mu02"] - common)
    hu = cv2.HuMoments(m).ravel()

    depths = []
    if len(cnt) > 3 and len(hull) > 3:
        idx = cv2.convexHull(cnt, returnPoints=False)
        try:
            defects = cv2.convexityDefects(cnt, idx)
        except cv2.error:   # contornos autointersectados: se trata como sin muescas
            defects = None
        if defects is not None:
            # OpenCV 4 devuelve (N,1,4) y OpenCV 5 (N,4); el cuarto valor es la profundidad x 256
            depths = [d / 256.0 / np.sqrt(area) for d in defects.reshape(-1, 4)[:, 3]]
    notches = [d for d in depths if d >= DEFECT_MIN_DEPTH]

    feats = {
        "shape_solidity": area / hull_area if hull_area > 0 else 1.0,
        "shape_extent": area / (rw * rh),
        "shape_aspect_ratio": max(rw, rh) / min(rw, rh),
        "shape_circularity": 4 * np.pi * area / perim ** 2 if perim > 0 else 0.0,
        "shape_convexity": hull_perim / perim if perim > 0 else 1.0,
        "shape_eccentricity": float(np.sqrt(max(0.0, 1 - l2 / l1))) if l1 > 0 else 0.0,
        "shape_n_defects": float(len(notches)),
        "shape_max_defect": float(max(depths)) if depths else 0.0,
        "shape_mean_defect": float(np.mean(depths)) if depths else 0.0,
    }
    for i, h in enumerate(hu, 1):
        feats[f"shape_hu{i}"] = float(-np.sign(h) * np.log10(abs(h) + 1e-30))
    return {k: float(v) for k, v in feats.items()}


def selftest():
    """Comprobaciones sobre figuras sintéticas con propiedades conocidas."""
    circ = np.zeros((400, 400), np.uint8)
    cv2.circle(circ, (200, 200), 100, 255, -1)
    f = shape_features(circ)
    # el perímetro de un contorno en píxeles sale ~10 % largo, por eso la circularidad no llega a 1
    assert f["shape_solidity"] > 0.98 and f["shape_circularity"] > 0.85 and f["shape_eccentricity"] < 0.1, f

    def ellipse(scale, angle):
        m = np.zeros((600, 600), np.uint8)
        cv2.ellipse(m, (300, 300), (int(120 * scale), int(40 * scale)), angle, 0, 360, 255, -1)
        return shape_features(m)

    a, b = ellipse(1.0, 0), ellipse(2.0, 40)        # misma forma, otra escala y rotación
    assert abs(a["shape_aspect_ratio"] - 3) < 0.15 and abs(b["shape_aspect_ratio"] - 3) < 0.15
    assert abs(a["shape_eccentricity"] - b["shape_eccentricity"]) < 0.02
    assert all(abs(a[f"shape_hu{i}"] - b[f"shape_hu{i}"]) < 0.3 for i in (1, 2)), "Hu no es invariante"

    c = np.zeros((400, 400), np.uint8)               # una "C" tiene una muesca profunda
    cv2.rectangle(c, (100, 100), (300, 300), 255, -1)
    cv2.rectangle(c, (200, 160), (300, 240), 0, -1)
    g = shape_features(c)
    assert g["shape_n_defects"] >= 1 and g["shape_solidity"] < 0.9, g
    assert shape_features(np.zeros((50, 50), np.uint8)) is None
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(SEGMENTATION_ROOT))
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    ap.add_argument("--output", default=None, help="CSV de salida (por defecto <root>/shape_features_all.csv)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return

    root = Path(args.root)
    rows, skipped = [], []
    for name in args.datasets:
        masks_dir = root / name / "masks"
        if not masks_dir.exists():
            print(f"[ERROR] No se encontró {masks_dir}")
            sys.exit(1)
        files = sorted(p for p in masks_dir.iterdir() if p.suffix.lower() in VALID_EXTS)
        print(f"{name}: {len(files)} máscaras")
        for p in files:
            mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            feats = shape_features(mask) if mask is not None else None
            if feats is None:
                skipped.append(f"{name}/{p.name}")
                continue
            rows.append({"filename": p.name, "dataset": name, **feats})

    if not rows:
        print("[ERROR] No se generó ninguna fila.")
        sys.exit(1)
    out = Path(args.output) if args.output else root / "shape_features_all.csv"
    df = pd.DataFrame(rows)
    assert not df.drop(columns=["filename", "dataset"]).isna().any().any(), "hay NaN en las features de forma"
    df.to_csv(out, index=False)
    print(f"\n{len(df)} imágenes con features de forma, {len(skipped)} omitidas. CSV: {out.resolve()}")
    if skipped:
        print(f"[AVISO] Omitidas: {skipped[:10]}")


if __name__ == "__main__":
    main()
