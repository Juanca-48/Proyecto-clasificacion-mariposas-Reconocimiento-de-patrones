#!/usr/bin/env python3
"""
viabilidad_genero.py
--------------------
Estudio de viabilidad de clasificar por GÉNERO (hoy el proyecto clasifica por familia).

1. Tabla de géneros: imágenes y especies por género. Con la validación agrupada por especie, un
   género necesita varias especies para tener alguna en train y otra en validación.
2. Cobertura: cuántos géneros e imágenes quedan según el mínimo de especies que se exija.
3. Piloto: regresión logística (la misma del modelo final) sobre las features estandarizadas, con CV
   agrupada por especie sobre train + val. EL TEST NO SE USA. Se prueba por familia (géneros dentro de
   Papilionidae, dentro de Pieridae) y plano (todos los géneros elegibles juntos), y se compara con
   el azar (balanced accuracy = 1 / nº de géneros).

El género y la especie se leen del nombre del archivo (001_Battus_crassus_crassus.png), porque
Pieridae.csv guarda rutas en la columna filename y no coincide con las fotos.

Salida en resultados/genero_viabilidad/: tabla_generos.csv, recall_por_genero.csv y resumen.txt.

Uso:
    python scripts/estudios/viabilidad_genero.py
    python scripts/estudios/viabilidad_genero.py --min-especies 2 --min-imagenes 5
"""
from __future__ import annotations

# Localiza rutas.py (en scripts/, o junto a este archivo en la app) y agrega al path la carpeta de cada proceso
import sys
from pathlib import Path
sys.path[:0] = [str(p) for p in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]) if (p / "rutas.py").exists()][:1]
import rutas  # noqa: E402,F401


import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from rutas import RESULTS_DIR
from Train_models import load_std
from Train_random_forest import SEED, SPLITS, label_from_filename

STD_DIR = RESULTS_DIR / "dataset_split_forma" / "standardized"   # vector con forma, estandarizado
OUT_DIR = RESULTS_DIR / "genero_viabilidad"


def genero_de(filename: str):
    label = label_from_filename(filename)
    return label.split("_")[0] if label else None


def piloto(df: pd.DataFrame, cols: list, generos: list, folds: int, seed: int):
    """CV agrupada por especie para clasificar género. Devuelve (métricas, recall por género) o None."""
    d = df[df.genus.isin(generos)].reset_index(drop=True)
    y, g = d.genus.values, d.species_group.values
    k = len(generos)
    n_splits = min(folds, int(d.groupby("genus").species_group.nunique().min()))
    if n_splits < 2:
        return None
    por_fold, y_true, y_pred = [], [], []
    for tr, te in StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(d[cols], y, g):
        m = LogisticRegression(class_weight="balanced", max_iter=5000).fit(d.loc[tr, cols], y[tr])
        p = m.predict(d.loc[te, cols])
        por_fold.append(balanced_accuracy_score(y[te], p))
        y_true += list(y[te]); y_pred += list(p)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    met = {"generos": k, "imagenes": len(d), "folds": n_splits,
           "balanced_acc": balanced_accuracy_score(y_true, y_pred), "bal_acc_std_folds": float(np.std(por_fold)),
           "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
           "accuracy": float((y_true == y_pred).mean()), "azar_balanced_acc": 1 / k}
    rec = pd.DataFrame({"genus": generos})
    rec["recall"] = [float((y_pred[y_true == q] == q).mean()) if (y_true == q).any() else np.nan for q in generos]
    return met, rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--std-dir", default=str(STD_DIR))
    ap.add_argument("--output", default=str(OUT_DIR))
    ap.add_argument("--min-especies", type=int, default=3, help="Especies mínimas por género para ser elegible")
    ap.add_argument("--min-imagenes", type=int, default=8, help="Imágenes mínimas por género para ser elegible")
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    lines: list = []

    def log(msg=""):
        print(msg); lines.append(str(msg))

    frames, cols = load_std(Path(args.std_dir), lambda *a: None)
    todo = pd.concat(frames.values(), ignore_index=True)
    todo["genus"] = todo.filename.map(genero_de)
    if todo.genus.isna().any():
        print(f"[ERROR] {int(todo.genus.isna().sum())} imágenes sin género deducible del nombre.")
        sys.exit(1)

    # --- 1) tabla de géneros (las 812 imágenes) ---
    tabla = (todo.groupby(["family", "genus"])
             .agg(imagenes=("filename", "size"), especies=("species_group", "nunique"),
                  max_imgs_una_especie=("species_group", lambda s: int(s.value_counts().max())))
             .reset_index().sort_values(["family", "imagenes"], ascending=[True, False]))
    tabla.to_csv(out / "tabla_generos.csv", index=False)
    log(f"=== GÉNEROS ({len(tabla)} en total, {len(todo)} imágenes) ===")
    for fam, t in tabla.groupby("family"):
        log(f"\n{fam}: {len(t)} géneros")
        log(t.drop(columns="family").to_string(index=False))

    # --- 2) cobertura según el mínimo de especies ---
    log("\n=== COBERTURA: géneros e imágenes que quedan según las especies mínimas exigidas ===")
    cob = []
    for ms in (2, 3, 4, 5):
        for fam, t in tabla.groupby("family"):
            el = t[(t.especies >= ms) & (t.imagenes >= args.min_imagenes)]
            cob.append({"min_especies": ms, "familia": fam, "generos": len(el), "de": len(t),
                        "pct_imagenes": round(100 * el.imagenes.sum() / t.imagenes.sum())})
    log(pd.DataFrame(cob).to_string(index=False))

    # --- 3) piloto con train + val (el test no se toca) ---
    elegibles = tabla[(tabla.especies >= args.min_especies) & (tabla.imagenes >= args.min_imagenes)]
    log(f"\n=== PILOTO (regresión logística, CV agrupada por especie, train + val; sin usar el test) ===")
    log(f"Géneros elegibles (>= {args.min_especies} especies y >= {args.min_imagenes} imágenes): "
        f"{len(elegibles)} de {len(tabla)}")
    fit = pd.concat([frames[s] for s in SPLITS if s != "test"], ignore_index=True)
    fit["genus"] = fit.filename.map(genero_de)
    resultados, recalls = [], []
    for nombre, filtro in [("Papilionidae", elegibles.family == "Papilionidae"),
                           ("Pieridae", elegibles.family == "Pieridae"), ("Plano (ambas familias)", elegibles.family != "")]:
        generos = list(elegibles[filtro].genus)
        # tras quitar el test, un género puede quedar con pocas especies: piloto() ajusta los folds
        r = piloto(fit, cols, generos, args.cv_folds, args.seed) if len(generos) >= 2 else None
        if r is None:
            log(f"\n{nombre}: no hay géneros suficientes para evaluar")
            continue
        met, rec = r
        resultados.append({"escenario": nombre, **met}); recalls.append(rec.assign(escenario=nombre))
        log(f"\n{nombre}: {met['generos']} géneros, {met['imagenes']} imágenes, {met['folds']} folds")
        log(f"  balanced_accuracy = {met['balanced_acc']:.3f} ± {met['bal_acc_std_folds']:.3f}   "
            f"(azar = {met['azar_balanced_acc']:.3f})   f1_macro = {met['f1_macro']:.3f}   accuracy = {met['accuracy']:.3f}")
        log("  recall por género: " + ", ".join(f"{a} {b:.2f}" for a, b in zip(rec.genus, rec.recall)))
    if recalls:
        pd.concat(recalls).to_csv(out / "recall_por_genero.csv", index=False)
    (out / "resumen.txt").write_text("\n".join(lines), encoding="utf-8")
    log(f"\nResultados en: {out.resolve()}")


if __name__ == "__main__":
    main()
