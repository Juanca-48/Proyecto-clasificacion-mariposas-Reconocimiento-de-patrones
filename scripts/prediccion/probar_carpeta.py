#!/usr/bin/env python3
"""
probar_carpeta.py
-----------------
Pasa las imágenes de una carpeta por el clasificador (el mismo código que la app de Hugging Face) y compara con la
familia real, que se lee del nombre del archivo: debe empezar por "Papilionidae" o "Pieridae" (sin distinguir
mayúsculas), por ejemplo Papilionidae3.JPG o pieridae1.webp.

Por imagen informa del estado (aceptada, incierta o rechazada), la familia predicha, el área de la máscara y la
distancia al conjunto de entrenamiento, y guarda una imagen con la foto y la segmentación que ve el modelo.

Salida en resultados/pruebas_app/: resumen.csv, informe.txt y una imagen por foto en segmentaciones/.

Uso:
    python scripts/prediccion/probar_carpeta.py
    python scripts/prediccion/probar_carpeta.py --carpeta "ruta\\mis_fotos"
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
from PIL import Image, ImageOps

from inferencia import Clasificador
from rutas import FINAL_DIR, REJECT_DIR, REPO_DIR, RESULTS_DIR

EXTS = {".jpg", ".jpeg", ".jfif", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def familia_real(nombre: str):
    n = nombre.lower()
    return "Papilionidae" if n.startswith("papilionidae") else "Pieridae" if n.startswith("pieridae") else None


def cargar(ruta: Path) -> np.ndarray:
    """Como lo hace Gradio: respeta la orientación EXIF y pasa a RGB."""
    return np.array(ImageOps.exif_transpose(Image.open(ruta)).convert("RGB"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--carpeta", default=str(REPO_DIR / "Pruebas"))
    ap.add_argument("--output", default=str(RESULTS_DIR / "pruebas_app"))
    args = ap.parse_args()

    fotos = sorted(p for p in Path(args.carpeta).iterdir() if p.suffix.lower() in EXTS)
    if not fotos:
        print(f"[ERROR] No hay imágenes en {args.carpeta}")
        sys.exit(1)
    out = Path(args.output); (out / "segmentaciones").mkdir(parents=True, exist_ok=True)
    clf = Clasificador(FINAL_DIR / "final_model.joblib", REJECT_DIR / "rechazo.joblib")
    lines: list = []

    def log(m=""):
        print(m); lines.append(str(m))

    filas = []
    for f in fotos:
        img = cargar(f)
        r = clf.predecir(img)
        real = familia_real(f.name)
        filas.append({"archivo": f.name, "real": real, "estado": r.estado, "predicha": r.familia,
                      "p_Papilionidae": r.probabilidades.get("Papilionidae"), "p_Pieridae": r.probabilidades.get("Pieridae"),
                      "area_mascara": round(r.area_mascara, 3),
                      "distancia": None if r.distancia is None else round(r.distancia),
                      "umbral_distancia": round(r.umbral_distancia), "alto": img.shape[0], "ancho": img.shape[1]})
        h = 320
        izq = cv2.resize(img, (round(img.shape[1] * h / img.shape[0]), h))
        der = cv2.resize(r.recorte_rgb, (round(r.recorte_rgb.shape[1] * h / r.recorte_rgb.shape[0]), h))
        cv2.imwrite(str(out / "segmentaciones" / f"{f.stem}.png"), cv2.cvtColor(np.hstack([izq, der]), cv2.COLOR_RGB2BGR))
    d = pd.DataFrame(filas)
    d.to_csv(out / "resumen.csv", index=False)

    log(f"{len(d)} imágenes de {args.carpeta}\n")
    log(d.drop(columns=["umbral_distancia"]).round(2).to_string(index=False))
    ok = (d.estado == "aceptada") & (d.predicha == d.real)
    log(f"\nCoinciden con la familia real (aceptadas y correctas): {ok.sum()} de {len(d)}")
    log(f"  rechazadas: {(d.estado == 'rechazada').sum()} | inciertas: {(d.estado == 'incierta').sum()} | "
        f"aceptadas pero con la familia equivocada: {((d.estado == 'aceptada') & (d.predicha != d.real)).sum()}")
    (out / "informe.txt").write_text("\n".join(lines), encoding="utf-8")
    log(f"\nResultados en: {out.resolve()}")


if __name__ == "__main__":
    main()
