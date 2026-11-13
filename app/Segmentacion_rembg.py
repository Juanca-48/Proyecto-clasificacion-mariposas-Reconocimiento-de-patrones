#!/usr/bin/env python3
"""
Segmentacion_rembg.py
---------------------
Segmenta las mariposas con rembg (red U2-Net) en lugar de GrabCut. Es el método que usa preprocesar.py.
Escribe lo mismo que Segmentacion.py (el método anterior, GrabCut, que se conserva), en las mismas carpetas:
resultados/segmentacion/<Dataset_...>/masks, segmented y preview.

La máscara de rembg pasa por la misma limpieza que la de GrabCut (morfología, mayor componente, huecos rellenos).
Quita bien hojas, flores y sombras que GrabCut deja dentro de la máscara. rembg es determinista: la misma foto da
siempre la misma máscara (GrabCut no).

Requiere: pip install rembg onnxruntime. La primera vez descarga el modelo (176 MB) en ~/.rembg.
OJO: como escribe en las mismas carpetas, reemplaza las máscaras de GrabCut. Las features, la división, los modelos
y la calibración del rechazo salen de ellas y hay que regenerarlos (preprocesar.py, entrenar.py, calibrar_rechazo.py).

Uso:
    python scripts/segmentacion/Segmentacion_rembg.py
    python scripts/segmentacion/Segmentacion_rembg.py --images "ruta\\fotos" --names Dataset_prueba --output "ruta\\salida"
    python scripts/segmentacion/Segmentacion_rembg.py --selftest
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

import Segmentacion as S   # limpieza de máscaras, bucle por carpeta y rutas por defecto

_SESION = None


def segment_rembg(img_bgr: np.ndarray) -> np.ndarray:
    """Máscara 0/255 de la mariposa con rembg + la limpieza de Segmentacion.py."""
    global _SESION
    from rembg import new_session, remove   # import perezoso: inferencia.py importa este módulo sin necesitar rembg
    if _SESION is None:
        _SESION = new_session("u2net")
    alfa = np.asarray(remove(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), session=_SESION, only_mask=True))
    return S.clean_mask(np.where(alfa > 127, 255, 0).astype(np.uint8))


def selftest():
    """Con una foto del dataset: máscara binaria, del tamaño de la imagen, de una sola pieza y con área razonable."""
    foto = next(p for p in sorted((S.DEFAULT_DATASETS[1][0]).iterdir()) if p.suffix.lower() in S.VALID_EXTS)
    img = cv2.imread(str(foto))
    mask = segment_rembg(img)
    assert mask.shape == img.shape[:2], mask.shape
    assert set(np.unique(mask)) <= {0, 255}, np.unique(mask)
    assert cv2.connectedComponents(mask)[0] == 2, "la máscara debe tener una sola pieza"
    assert 0.03 < (mask > 0).mean() < 0.9, (mask > 0).mean()
    print(f"selftest OK ({foto.name}: la máscara cubre el {(mask > 0).mean():.0%} de la imagen)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", nargs="+", default=None,
                    help="Una o varias carpetas con las imágenes originales (si se omite, usa los datasets por defecto)")
    ap.add_argument("--names", nargs="+", default=None, help="Nombres de salida correspondientes a --images")
    ap.add_argument("--output", default=str(S.DEFAULT_OUTPUT_DIR), help="Carpeta de salida raíz")
    ap.add_argument("--selftest", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return

    if args.images:
        names = args.names or [S.dataset_name_from_path(Path(p)) for p in args.images]
        datasets = list(zip([Path(p) for p in args.images], names))
    else:
        datasets = S.DEFAULT_DATASETS
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    for images_dir, name in datasets:
        print(f"\n=== Dataset: {name} ({images_dir}) ===")
        if not S.segment_folder(images_dir, output_root / name, method="rembg", segment_fn=segment_rembg):
            sys.exit(1)
    print(f"\nTodo listo. Resultados en: {output_root.resolve()}")


if __name__ == "__main__":
    main()
