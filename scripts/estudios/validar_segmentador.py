#!/usr/bin/env python3
"""
validar_segmentador.py
----------------------
Compara dos segmentadores en la inferencia: GrabCut (el del entrenamiento) y rembg (red U2-Net). El modelo y los
umbrales de rechazo NO cambian: solo la máscara con la que se calculan las features de la foto.

Sobre las 122 fotos del test de catálogo (familia conocida, 24 especies): balanced accuracy y AUC del modelo sin
rechazo, tasas de rechazo, diferencia entre segmentadores con IC 95 % (bootstrap por especie) y cuánto se parecen
las dos máscaras (IoU). Después, las fotos de una carpeta (por defecto Pruebas/), de forma cualitativa, y un montaje
con las dos segmentaciones lado a lado.

Para que valga la pena, rembg debe mejorar las fotos de internet SIN empeorar las de catálogo, porque el modelo se
entrenó con máscaras de GrabCut.

Salida en resultados/segmentador/: resumen.txt, resultados.csv y montaje_<carpeta>.png.

Uso:
    python scripts/estudios/validar_segmentador.py
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
import warnings
from functools import lru_cache
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from rutas import DATASET_DIR, FINAL_DIR, REJECT_DIR, REPO_DIR, RESULTS_DIR, SPLIT_DIR
from Train_random_forest import SEED, prepare_frame

OUT_DIR = RESULTS_DIR / "segmentador"
SEGMENTADORES = ["grabcut", "rembg"]


@lru_cache(maxsize=1)
def _clasificador():
    from inferencia import Clasificador
    return Clasificador(FINAL_DIR / "final_model.joblib", REJECT_DIR / "rechazo.joblib")


def _tarea(t):
    ruta, seg = t
    img = cv2.cvtColor(cv2.imread(ruta), cv2.COLOR_BGR2RGB)
    clf = _clasificador()
    a = clf.analizar(img, None, seg)
    r = clf.decidir(a)
    p = a["probabilidades"] or {}
    return {"seg": seg, "p_pier": p.get("Pieridae", np.nan), "distancia": a["distancia"], "area": a["area"],
            "estado": r.estado, "mascara": (a["mascara"] > 0)}


def metricas(d: pd.DataFrame) -> dict:
    y = (d.real == "Pieridae").values
    ok = d.p_pier.notna().values
    pred = d.p_pier.values > 0.5
    acept = (d.estado == "aceptada").values
    return {"balanced_acc": balanced_accuracy_score(y[ok], pred[ok]), "auc": roc_auc_score(y[ok], d.p_pier.values[ok]),
            "rechazadas": (d.estado == "rechazada").mean(), "inciertas": (d.estado == "incierta").mean(),
            "exactitud_aceptadas": float((pred[acept] == y[acept]).mean()) if acept.any() else np.nan,
            "mediana_distancia": d.distancia.median(), "area_media": d.area.mean()}


def ic_diferencia(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 1000, seed: int = SEED):
    rng = np.random.default_rng(seed)
    idx = {g: np.where(a.species_group.values == g)[0] for g in pd.unique(a.species_group.values)}
    claves, difs = list(idx), []
    for _ in range(n_boot):
        sel = np.concatenate([idx[claves[k]] for k in rng.integers(0, len(claves), len(claves))])
        y = (a.real.values[sel] == "Pieridae")
        if y.all() or not y.any():
            continue
        difs.append(balanced_accuracy_score(y, b.p_pier.values[sel] > .5) - balanced_accuracy_score(y, a.p_pier.values[sel] > .5))
    return np.percentile(difs, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--carpeta", default=str(REPO_DIR / "Pruebas"))
    ap.add_argument("--output", default=str(OUT_DIR))
    ap.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    lines: list = []

    def log(m=""):
        print(m, flush=True); lines.append(str(m))

    test = prepare_frame(pd.read_csv(SPLIT_DIR / "features_test.csv"), False, lambda *a: None, "test")
    rutas = [str(DATASET_DIR / r.family / r.filename) for r in test.itertuples()]
    trabajos = [(r, s) for s in SEGMENTADORES for r in rutas]
    log(f"{len(test)} fotos de test x {len(SEGMENTADORES)} segmentadores ({args.jobs} procesos)")
    res = []
    with get_context("spawn").Pool(args.jobs) as pool:
        for i, fila in enumerate(pool.imap(_tarea, trabajos, chunksize=4), 1):
            res.append(fila)
            if i % 60 == 0 or i == len(trabajos):
                log(f"  {i}/{len(trabajos)}")
    masks = {s: [r["mascara"] for r in res if r["seg"] == s] for s in SEGMENTADORES}
    d = pd.DataFrame([{k: v for k, v in r.items() if k != "mascara"} for r in res])
    d["real"] = np.tile(test.family.values, len(SEGMENTADORES))
    d["species_group"] = np.tile(test.species_group.values, len(SEGMENTADORES))
    d["archivo"] = np.tile(test.filename.values, len(SEGMENTADORES))
    d.to_csv(out / "resultados.csv", index=False)

    log("\n=== TEST DE CATÁLOGO por segmentador (el modelo y los umbrales son los mismos) ===")
    log(pd.DataFrame([{"segmentador": s, **metricas(g)} for s, g in d.groupby("seg", sort=False)]).round(3).to_string(index=False))
    a, b = (d[d.seg == s].reset_index(drop=True) for s in SEGMENTADORES)
    lo, hi = ic_diferencia(a, b)
    log(f"\nrembg - grabcut, balanced accuracy: {metricas(b)['balanced_acc'] - metricas(a)['balanced_acc']:+.3f}  "
        f"IC 95 % [{lo:+.3f}, {hi:+.3f}]  | {int(((a.p_pier > .5) != (b.p_pier > .5)).sum())} de {len(a)} predicciones cambian de familia")
    iou = np.array([(m1 & m2).sum() / max((m1 | m2).sum(), 1) for m1, m2 in zip(*masks.values())])
    log(f"IoU entre las dos máscaras en el catálogo: mediana {np.median(iou):.3f}, p10 {np.percentile(iou, 10):.3f}, mín {iou.min():.3f}")

    carpeta = Path(args.carpeta)
    if carpeta.exists():
        from PIL import Image, ImageOps
        from probar_carpeta import EXTS, familia_real
        clf = _clasificador()
        log(f"\n=== {carpeta.name}/: fotos de internet, comparación cualitativa ===")
        log(f"{'foto':<22}{'real':<14} | {'GrabCut':^30} | {'rembg':^30}")
        filas = []
        for f in sorted(p for p in carpeta.iterdir() if p.suffix.lower() in EXTS):
            img = np.array(ImageOps.exif_transpose(Image.open(f)).convert("RGB"))
            real, celdas, vistas = familia_real(f.name), [], []
            for seg in SEGMENTADORES:
                an = clf.analizar(img, None, seg); r = clf.decidir(an)
                pr = (an["probabilidades"] or {}).get("Pieridae", np.nan)
                marca = "" if np.isnan(pr) else ("OK" if (pr > .5) == (real == "Pieridae") else "MAL")
                celdas.append(f"P(Pier)={pr:4.2f} {marca:<3}{r.estado:>10}")
                vistas.append(an["recorte_rgb"])
            log(f"{f.name:<22}{real:<14} | {celdas[0]:<30} | {celdas[1]:<30}")
            h = 260
            fila = [cv2.resize(x, (round(x.shape[1] * h / x.shape[0]), h)) for x in [img] + vistas]
            fila = np.hstack(fila); cv2.putText(fila, f"{f.name} (original | GrabCut | rembg)", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 2)
            filas.append(cv2.resize(fila, (1200, round(fila.shape[0] * 1200 / fila.shape[1]))))
        if filas:
            Hm = max(x.shape[0] for x in filas)
            cv2.imwrite(str(out / f"montaje_{carpeta.name}.png"), cv2.cvtColor(np.vstack(
                [cv2.copyMakeBorder(x, 0, Hm - x.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for x in filas]), cv2.COLOR_RGB2BGR))
    (out / "resumen.txt").write_text("\n".join(lines), encoding="utf-8")
    log(f"\nResultados en: {out.resolve()}")


if __name__ == "__main__":
    main()
