#!/usr/bin/env python3
"""
validar_escala.py
-----------------
Valida la normalización de escala de las fotos subidas a la app. La textura (GLCM, Gabor) depende de la escala en
píxeles, y las fotos del catálogo miden unos 640 px de lado mayor. Compara dos políticas:

  actual (tope 800)   solo se reduce lo que supera 800 px; una foto pequeña se deja como está.
  normalizada 640     el lado mayor pasa siempre a 640 px (se reduce o se amplía).

Sobre las 122 fotos del test de catálogo (24 especies, familia conocida), con tres versiones de cada una: tamaño
original, reducida a 450 px y reducida a 330 px (simula fotos pequeñas de internet). Por foto se calculan las
probabilidades del modelo con y sin rechazo. Informa de balanced accuracy, AUC, rechazos y de la diferencia entre
políticas con su intervalo de confianza (bootstrap por especie). Al final pasa por las dos políticas las fotos de
una carpeta (por defecto Pruebas/), solo de forma cualitativa.

No se puede simular el caso contrario (fotos más grandes que el catálogo): las fotos de catálogo no tienen más detalle.

Salida en resultados/escala/: resumen.txt y resultados.csv.

Uso:
    python scripts/validar_escala.py
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from multiprocessing import get_context
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from rutas import DATASET_DIR, FINAL_DIR, REJECT_DIR, REPO_DIR, RESULTS_DIR, SPLIT_DIR
from Train_random_forest import SEED, prepare_frame

SCRIPTS = Path(__file__).resolve().parent
OUT_DIR = RESULTS_DIR / "escala"
CONDICIONES = {"original": None, "450 px": 450, "330 px": 330}      # lado mayor de la foto de entrada
POLITICAS = {"actual (tope 800)": None, "normalizada 640": 640}     # lado al que se lleva en la app


@lru_cache(maxsize=1)
def _clasificador():
    sys.path.insert(0, str(SCRIPTS))   # los procesos de joblib no heredan siempre el path
    from inferencia import Clasificador
    return Clasificador(FINAL_DIR / "final_model.joblib", REJECT_DIR / "rechazo.joblib")


def _tarea(ruta: str, cond: str, pol: str) -> dict:
    img = cv2.cvtColor(cv2.imread(ruta), cv2.COLOR_BGR2RGB)
    lado = CONDICIONES[cond]
    if lado and max(img.shape[:2]) > lado:                          # solo se simula reducir
        s = lado / max(img.shape[:2])
        img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    clf = _clasificador()
    a = clf.analizar(img, POLITICAS[pol])
    r = clf.decidir(a)
    p = a["probabilidades"] or {}
    return {"cond": cond, "pol": pol, "p_pier": p.get("Pieridae", np.nan), "distancia": a["distancia"],
            "area": a["area"], "estado": r.estado, "alto": img.shape[0], "ancho": img.shape[1]}


def _tarea_tupla(t):
    return _tarea(*t)


def metricas(d: pd.DataFrame) -> dict:
    y = (d.real == "Pieridae").values
    ok = d.p_pier.notna().values
    pred = d.p_pier.values > 0.5
    acept = (d.estado == "aceptada").values
    return {"balanced_acc": balanced_accuracy_score(y[ok], pred[ok]), "auc": roc_auc_score(y[ok], d.p_pier.values[ok]),
            "rechazadas": (d.estado == "rechazada").mean(), "inciertas": (d.estado == "incierta").mean(),
            "exactitud_aceptadas": float((pred[acept] == y[acept]).mean()) if acept.any() else np.nan}


def ic_diferencia(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 1000, seed: int = SEED):
    """IC 95 % de la diferencia de balanced accuracy (b - a) remuestreando especies completas."""
    rng = np.random.default_rng(seed)
    grupos = a.species_group.values
    idx = {g: np.where(grupos == g)[0] for g in pd.unique(grupos)}
    claves = list(idx)
    difs = []
    for _ in range(n_boot):
        sel = np.concatenate([idx[claves[k]] for k in rng.integers(0, len(claves), len(claves))])
        ya = (a.real.values[sel] == "Pieridae")
        if ya.all() or not ya.any():
            continue
        pa, pb = a.p_pier.values[sel] > 0.5, b.p_pier.values[sel] > 0.5
        difs.append(balanced_accuracy_score(ya, pb) - balanced_accuracy_score(ya, pa))
    return np.percentile(difs, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--carpeta", default=str(REPO_DIR / "Pruebas"), help="Fotos para la comparación cualitativa")
    ap.add_argument("--output", default=str(OUT_DIR))
    ap.add_argument("--jobs", type=int, default=min(6, os.cpu_count() or 1))
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    lines: list = []

    def log(m=""):
        print(m, flush=True); lines.append(str(m))

    test = prepare_frame(pd.read_csv(SPLIT_DIR / "features_test.csv"), False, lambda *a: None, "test")
    rutas = [str(DATASET_DIR / r.family / r.filename) for r in test.itertuples()]
    combos = [(c, p) for c in CONDICIONES for p in POLITICAS]
    log(f"{len(test)} fotos de test x {len(combos)} combinaciones = {len(test) * len(combos)} ejecuciones ({args.jobs} procesos)")
    trabajos = [(r, c, p) for c, p in combos for r in rutas]
    res = []
    with get_context("spawn").Pool(args.jobs) as pool:               # spawn: igual en Windows y Linux
        for i, fila in enumerate(pool.imap(_tarea_tupla, trabajos, chunksize=4), 1):
            res.append(fila)
            if i % 100 == 0 or i == len(trabajos):
                log(f"  {i}/{len(trabajos)}")
    d = pd.DataFrame(res)
    d["real"] = np.tile(test.family.values, len(combos))
    d["species_group"] = np.tile(test.species_group.values, len(combos))
    d["archivo"] = np.tile(test.filename.values, len(combos))
    d.to_csv(out / "resultados.csv", index=False)

    log("\n=== TEST DE CATÁLOGO por condición de entrada y política (balanced accuracy con el modelo sin rechazo) ===")
    filas = []
    for (c, p), g in d.groupby(["cond", "pol"], sort=False):
        filas.append({"entrada": c, "política": p, **metricas(g), "mediana_distancia": g.distancia.median()})
    tabla = pd.DataFrame(filas)
    log(tabla.round(3).to_string(index=False))

    log("\n=== DIFERENCIA normalizada 640 - actual (balanced accuracy), IC 95 % por bootstrap de especies ===")
    for c in CONDICIONES:
        a = d[(d.cond == c) & (d.pol == "actual (tope 800)")].reset_index(drop=True)
        b = d[(d.cond == c) & (d.pol == "normalizada 640")].reset_index(drop=True)
        lo, hi = ic_diferencia(a, b)
        flips = int(((a.p_pier > .5) != (b.p_pier > .5)).sum())
        log(f"  entrada {c:<9}: {metricas(b)['balanced_acc'] - metricas(a)['balanced_acc']:+.3f}  IC [{lo:+.3f}, {hi:+.3f}]  "
            f"| {flips} de {len(a)} predicciones cambian de familia")

    carpeta = Path(args.carpeta)
    if carpeta.exists():
        log(f"\n=== {carpeta.name}/: fotos de internet, comparación cualitativa (P(Pieridae) sin rechazo) ===")
        from PIL import Image, ImageOps
        sys.path.insert(0, str(SCRIPTS))
        from probar_carpeta import EXTS, familia_real
        clf = _clasificador()
        log(f"{'foto':<22}{'real':<14}{'tamaño':>11} | {'actual':>8}{'estado':>11} | {'normalizada':>12}{'estado':>11}")
        for f in sorted(p for p in carpeta.iterdir() if p.suffix.lower() in EXTS):
            img = np.array(ImageOps.exif_transpose(Image.open(f)).convert("RGB"))
            fila = []
            for lado in POLITICAS.values():
                a = clf.analizar(img, lado)
                fila.append((a["probabilidades"]["Pieridae"] if a["probabilidades"] else np.nan, clf.decidir(a).estado))
            real = familia_real(f.name)
            marca = lambda pr: "" if np.isnan(pr) else ("OK" if (pr > .5) == (real == "Pieridae") else "MAL")
            log(f"{f.name:<22}{real:<14}{img.shape[1]}x{img.shape[0]:>5} | {fila[0][0]:>5.2f} {marca(fila[0][0]):<3}{fila[0][1]:>11} "
                f"| {fila[1][0]:>8.2f} {marca(fila[1][0]):<3}{fila[1][1]:>11}")
    (out / "resumen.txt").write_text("\n".join(lines), encoding="utf-8")
    log(f"\nResultados en: {out.resolve()}")


if __name__ == "__main__":
    main()
