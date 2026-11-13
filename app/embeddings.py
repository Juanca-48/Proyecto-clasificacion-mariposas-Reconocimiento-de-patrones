#!/usr/bin/env python3
"""
embeddings.py
-------------
Extrae embeddings de las 812 imágenes con redes preentrenadas (sin afinarlas), para compararlos con las
características clásicas de textura, color y forma. evaluar_embeddings.py los evalúa con el mismo protocolo.

Redes (se descargan la primera vez, de Hugging Face):
  bioclip    BioCLIP (ViT-B/16, entrenada con imágenes de seres vivos y su taxonomía)   open_clip, imageomics/bioclip
  dinov2     DINOv2 base (ViT-B/14, autosupervisada)                                    transformers, facebook/dinov2-base
  convnext   ConvNeXt-Tiny (CNN, ImageNet-22k -> 1k)                                    timm, convnext_tiny.fb_in22k_ft_in1k

Entradas:
  recorte    la mariposa segmentada sobre fondo blanco (resultados/segmentacion/<dataset>/segmented/)
  original   la foto completa (dataset/<Familia>/)

Salida en resultados/embeddings/: indice.csv (dataset, family, filename; el orden de las filas de los .npy) y un
<red>__<entrada>.npy por combinación, de forma (812, dimensión). Si el archivo ya existe se omite (--rehacer lo recalcula).

Uso:
    python scripts/redes_preentrenadas/embeddings.py
    python scripts/redes_preentrenadas/embeddings.py --modelos dinov2 --entradas recorte
    python scripts/redes_preentrenadas/embeddings.py --limite 6 --output <carpeta>        (prueba rápida)
"""
from __future__ import annotations

# Localiza rutas.py (en scripts/, o junto a este archivo en la app) y agrega al path la carpeta de cada proceso
import sys
from pathlib import Path
sys.path[:0] = [str(p) for p in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]) if (p / "rutas.py").exists()][:1]
import rutas  # noqa: E402,F401


import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import pandas as pd
from PIL import Image

from rutas import DATASET_DIR, RESULTS_DIR, SEGMENTATION_DIR

MODELOS = ["bioclip", "dinov2", "convnext"]
ENTRADAS = ["recorte", "original"]
DATASETS = {"Dataset_papilionidae": "Papilionidae", "Dataset_pieridae": "Pieridae"}


def construir(nombre: str):
    """Devuelve una función que convierte una lista de imágenes PIL en un array (n, dimensión)."""
    import torch

    def sin_grad(f):
        # el modo sin gradiente es por hilo y Gradio atiende cada petición en otro hilo: se protege cada llamada
        def g(ims):
            with torch.no_grad():
                return f(ims)
        return g

    if nombre == "bioclip":
        import open_clip
        modelo, _, prep = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip")
        modelo.eval()
        return sin_grad(lambda ims: modelo.encode_image(torch.stack([prep(i) for i in ims])).float().numpy())
    if nombre == "dinov2":
        from transformers import AutoImageProcessor, AutoModel
        proc, modelo = AutoImageProcessor.from_pretrained("facebook/dinov2-base"), AutoModel.from_pretrained("facebook/dinov2-base").eval()
        return sin_grad(lambda ims: modelo(**proc(images=ims, return_tensors="pt")).pooler_output.float().numpy())
    if nombre == "convnext":
        import timm
        modelo = timm.create_model("convnext_tiny.fb_in22k_ft_in1k", pretrained=True, num_classes=0).eval()
        tf = timm.data.create_transform(**timm.data.resolve_data_config({}, model=modelo))
        return sin_grad(lambda ims: modelo(torch.stack([tf(i) for i in ims])).float().numpy())
    raise ValueError(nombre)


def indice() -> pd.DataFrame:
    filas = [(ds, fam, p.name) for ds, fam in DATASETS.items()
             for p in sorted((SEGMENTATION_DIR / ds / "segmented").glob("*.png"))]
    return pd.DataFrame(filas, columns=["dataset", "family", "filename"])


def ruta_imagen(fila, entrada: str) -> Path:
    if entrada == "recorte":
        return SEGMENTATION_DIR / fila.dataset / "segmented" / fila.filename
    return DATASET_DIR / fila.family / fila.filename


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelos", nargs="+", choices=MODELOS, default=MODELOS)
    ap.add_argument("--entradas", nargs="+", choices=ENTRADAS, default=ENTRADAS)
    ap.add_argument("--output", default=str(RESULTS_DIR / "embeddings"))
    ap.add_argument("--lote", type=int, default=16)
    ap.add_argument("--limite", type=int, default=None, help="Solo las primeras N imágenes (prueba rápida)")
    ap.add_argument("--rehacer", action="store_true")
    args = ap.parse_args()

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    idx = indice()
    if idx.empty:
        print("[ERROR] No hay recortes en resultados/segmentacion. Ejecuta preprocesar.py (etapa segmentacion).")
        sys.exit(1)
    if args.limite:
        idx = idx.sample(args.limite, random_state=0).sort_values(["dataset", "filename"]).reset_index(drop=True)
    idx.to_csv(out / "indice.csv", index=False)
    print(f"{len(idx)} imágenes -> {out}")

    for nombre in args.modelos:
        embed = None
        for entrada in args.entradas:
            destino = out / f"{nombre}__{entrada}.npy"
            if destino.exists() and not args.rehacer:
                print(f"[omitido] {destino.name} ya existe")
                continue
            embed = embed or construir(nombre)       # el modelo se carga una vez por red
            t0, partes = time.time(), []
            filas = list(idx.itertuples())
            for i in range(0, len(filas), args.lote):
                ims = [Image.open(ruta_imagen(f, entrada)).convert("RGB") for f in filas[i:i + args.lote]]
                partes.append(embed(ims))
            e = np.concatenate(partes).astype(np.float32)
            assert e.shape[0] == len(idx) and np.isfinite(e).all() and (np.abs(e).sum(1) > 0).all(), f"embeddings inválidos: {destino.name}"
            np.save(destino, e)
            print(f"{destino.name}: {e.shape} en {time.time() - t0:.0f} s ({(time.time() - t0) / len(idx):.2f} s/imagen)", flush=True)


if __name__ == "__main__":
    main()
