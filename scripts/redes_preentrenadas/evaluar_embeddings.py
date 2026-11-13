#!/usr/bin/env python3
"""
evaluar_embeddings.py
---------------------
Evalúa los embeddings de embeddings.py con el mismo protocolo que el modelo clásico: regresión logística sobre el
vector estandarizado, partición agrupada por especie (la de resultados/dataset_split) y el test una sola vez.

1. Por cada combinación red x entrada se ajusta el parámetro C con una CV agrupada por especie sobre train + val
   (5 folds x --repeticiones) y se informa de balanced accuracy y AUC. El test no interviene.
2. Como referencia, el modelo clásico (147 features de textura y color de preprocesar.py) con exactamente la misma CV.
3. Se elige la mejor combinación por esa CV y SOLO ella se evalúa en test, con IC 95 % por bootstrap de especies,
   junto con el modelo clásico en el mismo test.

Salida en resultados/embeddings/: evaluacion.txt y evaluacion_cv.csv.

Uso:
    python scripts/redes_preentrenadas/evaluar_embeddings.py
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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, normalize

from rutas import RESULTS_DIR, SEGMENTATION_DIR, SPLIT_DIR
from Train_random_forest import NON_FEATURE_COLS, SEED, cluster_bootstrap, label_from_filename

CS = [0.01, 0.1, 1, 10]


def modelo(C: float) -> Pipeline:
    return Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()),
                     ("c", LogisticRegression(C=C, class_weight="balanced", max_iter=5000))])


def cv_oof(X, y, grupos, C, repeticiones, folds=5):
    """Probabilidades fuera de muestra por repetición (repeticiones, n)."""
    out = np.zeros((repeticiones, len(y)))
    for r in range(repeticiones):
        for tr, te in StratifiedGroupKFold(folds, shuffle=True, random_state=r).split(X, y, grupos):
            out[r, te] = modelo(C).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return out


def resumen_cv(oof, y):
    ba = [balanced_accuracy_score(y, o > .5) for o in oof]
    return float(np.mean(ba)), float(np.std(ba)), float(np.mean([roc_auc_score(y, o) for o in oof]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embeddings", default=str(RESULTS_DIR / "embeddings"))
    ap.add_argument("--repeticiones", type=int, default=5)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    emb = Path(args.embeddings)
    lines: list = []

    def log(m=""):
        print(m, flush=True); lines.append(str(m))

    idx = pd.read_csv(emb / "indice.csv")
    sp = pd.read_csv(SPLIT_DIR / "splits.csv")[["family", "filename", "split"]]
    idx = idx.merge(sp, on=["family", "filename"], how="left")
    if idx.split.isna().any():
        print(f"[ERROR] {int(idx.split.isna().sum())} imágenes sin partición. Vuelve a ejecutar entrenar.py (etapa split).")
        sys.exit(1)
    y = (idx.family == "Pieridae").values.astype(int)
    grupos = np.array([" ".join(label_from_filename(f).lower().split("_")[:2]) for f in idx.filename])
    fit, test = (idx.split != "test").values, (idx.split == "test").values

    # --- conjuntos de features: embeddings y el modelo clásico ---
    conjuntos = {}
    for f in sorted(emb.glob("*__*.npy")):
        conjuntos[f.stem.replace("__", " / ")] = normalize(np.load(f))       # L2: lo habitual con embeddings
    clasico = pd.read_csv(SEGMENTATION_DIR / "features_dataset_ready.csv").set_index(["dataset", "filename"])
    clasico = clasico.loc[list(zip(idx.dataset, idx.filename))]
    conjuntos["CLÁSICO (textura + color)"] = clasico[[c for c in clasico.columns if c not in NON_FEATURE_COLS]].values
    log(f"{len(idx)} imágenes: train+val {fit.sum()}, test {test.sum()}; {len(set(grupos[fit]))} especies en train+val, "
        f"{len(set(grupos[test]))} en test. Regresión logística con C elegida por CV.")

    # --- 1) CV agrupada sobre train + val ---
    log(f"\n=== CV agrupada por especie sobre train + val ({args.repeticiones} repeticiones x 5 folds) ===")
    filas, mejores = [], {}
    for nombre, X in conjuntos.items():
        res = {C: cv_oof(X[fit], y[fit], grupos[fit], C, args.repeticiones) for C in CS}
        C = max(res, key=lambda c: resumen_cv(res[c], y[fit])[0])
        ba, sd, auc = resumen_cv(res[C], y[fit])
        mejores[nombre] = C
        filas.append({"conjunto": nombre, "dim": X.shape[1], "C": C, "balanced_acc": ba, "desv_repeticiones": sd, "auc": auc})
    cv = pd.DataFrame(filas).sort_values("balanced_acc", ascending=False)
    cv.to_csv(emb / "evaluacion_cv.csv", index=False)
    log(cv.round(3).to_string(index=False))

    # --- 2) test, una sola vez, para el mejor embedding y para el clásico ---
    mejor = cv[cv.conjunto != "CLÁSICO (textura + color)"].iloc[0].conjunto
    log(f"\n=== TEST (una sola evaluación): mejor embedding por la CV = '{mejor}' ===")
    for nombre in (mejor, "CLÁSICO (textura + color)"):
        X = conjuntos[nombre]
        m = modelo(mejores[nombre]).fit(X[fit], y[fit])
        p = m.predict_proba(X[test])[:, 1]; pred = (p > .5).astype(int)
        ci = cluster_bootstrap(y[test], pred, np.column_stack([1 - p, p]), grupos[test], 2, 1000, SEED)
        log(f"{nombre:<28} balanced_acc {balanced_accuracy_score(y[test], pred):.3f} "
            f"IC95 [{ci['balanced_accuracy'][0]:.3f}, {ci['balanced_accuracy'][1]:.3f}] | AUC {roc_auc_score(y[test], p):.3f} "
            f"IC95 [{ci['auc'][0]:.3f}, {ci['auc'][1]:.3f}] | errores {int((pred != y[test]).sum())} de {int(test.sum())}")
    (emb / "evaluacion.txt").write_text("\n".join(lines), encoding="utf-8")
    log(f"\nResultados en: {emb.resolve()}")


if __name__ == "__main__":
    main()
