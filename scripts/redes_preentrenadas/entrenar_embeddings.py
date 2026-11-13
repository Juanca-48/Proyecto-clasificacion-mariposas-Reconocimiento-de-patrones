#!/usr/bin/env python3
"""
entrenar_embeddings.py
----------------------
Entrena el segundo modelo de la app (embeddings de una red preentrenada + regresión logística) y calibra su rechazo
con el mismo método que calibrar_rechazo.py. Todo con train + val: el test solo se usa para informar.

  1. Elige la red y la entrada: las indicadas, o la mejor de evaluacion_cv.csv (la que dejó evaluar_embeddings.py).
  2. Regresión logística sobre el embedding L2-normalizado y estandarizado, con la C de esa evaluación.
  3. Rechazo: área de la máscara (solo con entrada "recorte"), distancia de Mahalanobis con umbral en el percentil
     --percentil de una CV agrupada por especie, y umbral de confianza.
  4. Guarda resultados/modelo_embeddings/modelo.joblib, con todo lo que necesita inferencia_embeddings.py.

Uso:
    python scripts/redes_preentrenadas/entrenar_embeddings.py
    python scripts/redes_preentrenadas/entrenar_embeddings.py --red dinov2 --entrada recorte
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

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import normalize

from evaluar_embeddings import CS, cv_oof, modelo, resumen_cv
from rutas import RESULTS_DIR, SEGMENTATION_DIR, SPLIT_DIR
from Train_random_forest import SEED, label_from_filename


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embeddings", default=str(RESULTS_DIR / "embeddings"))
    ap.add_argument("--output", default=str(RESULTS_DIR / "modelo_embeddings"))
    ap.add_argument("--red", choices=["bioclip", "dinov2", "convnext"])
    ap.add_argument("--entrada", choices=["recorte", "original"])
    ap.add_argument("--percentil", type=float, default=95)
    ap.add_argument("--umbral-confianza", type=float, default=0.65)
    ap.add_argument("--segmentador", choices=["grabcut", "rembg"], default="rembg",
                    help="Segmentador con el que se generaron los recortes (preprocesar.py usa rembg)")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    emb, out = Path(args.embeddings), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    cv_csv = emb / "evaluacion_cv.csv"
    if not args.red or not args.entrada:
        if not cv_csv.exists():
            print("[ERROR] Indica --red y --entrada, o ejecuta antes evaluar_embeddings.py para elegir la mejor.")
            sys.exit(1)
        mejor = pd.read_csv(cv_csv)
        mejor = mejor[~mejor.conjunto.str.startswith("CLÁSICO")].iloc[0].conjunto
        args.red, args.entrada = mejor.split(" / ")
    nombre = f"{args.red} / {args.entrada}"
    X = normalize(np.load(emb / f"{args.red}__{args.entrada}.npy"))

    idx = pd.read_csv(emb / "indice.csv").merge(pd.read_csv(SPLIT_DIR / "splits.csv")[["family", "filename", "split"]],
                                                on=["family", "filename"], how="left")
    y = (idx.family == "Pieridae").values.astype(int)
    grupos = np.array([" ".join(label_from_filename(f).lower().split("_")[:2]) for f in idx.filename])
    fit, test = (idx.split != "test").values, (idx.split == "test").values

    # C: la de la evaluación si existe; si no, se calcula con la CV agrupada
    if cv_csv.exists() and (pd.read_csv(cv_csv).conjunto == nombre).any():
        C = float(pd.read_csv(cv_csv).set_index("conjunto").loc[nombre, "C"])
    else:
        res = {c: cv_oof(X[fit], y[fit], grupos[fit], c, 3) for c in CS}
        C = max(res, key=lambda c: resumen_cv(res[c], y[fit])[0])
    m = modelo(C).fit(X[fit], y[fit])
    print(f"Modelo: {nombre}, C={C}, entrenado con {fit.sum()} imágenes (train + val).")

    # --- rechazo ---
    Z = m.named_steps["s"].transform(m.named_steps["i"].transform(X))
    d2 = np.zeros(fit.sum())
    Zf = Z[fit]
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(Zf, y[fit], grupos[fit]):
        d2[te] = LedoitWolf().fit(Zf[tr]).mahalanobis(Zf[te])
    umbral = float(np.percentile(d2, args.percentil))
    cov = LedoitWolf().fit(Zf)
    area_min = area_max = None
    if args.entrada == "recorte":
        areas = np.array([(cv2.imread(str(SEGMENTATION_DIR / r.dataset / "masks" / r.filename), cv2.IMREAD_GRAYSCALE) > 0).mean()
                          for r in idx[fit].itertuples()])
        area_min, area_max = float(0.8 * areas.min()), float(1.1 * areas.max())
        print(f"Área de la máscara en train + val: {areas.min():.3f} a {areas.max():.3f} -> rango [{area_min:.3f}, {area_max:.3f}]")
    print(f"Umbral de distancia (p{args.percentil:g} de la CV agrupada): {umbral:.0f}  (mediana {np.median(d2):.0f})")

    # --- informe en test (el test no intervino en nada de lo anterior) ---
    p = m.predict_proba(X[test])[:, 1]; pred = (p > .5).astype(int)
    d_t = cov.mahalanobis(Z[test]); conf = np.maximum(p, 1 - p)
    rech, inc = d_t > umbral, (d_t <= umbral) & (conf < args.umbral_confianza)
    acept = ~rech & ~inc
    metricas = {"balanced_acc": float(balanced_accuracy_score(y[test], pred)), "auc": float(roc_auc_score(y[test], p)),
                "imagenes": int(test.sum()), "especies": len(set(grupos[test]))}
    print(f"\nTEST ({test.sum()} imágenes, {len(set(grupos[test]))} especies): balanced_acc {balanced_accuracy_score(y[test], pred):.3f}, "
          f"AUC {roc_auc_score(y[test], p):.3f}")
    print(f"  rechazadas por distancia: {rech.sum()} ({rech.mean():.0%}) | inciertas: {inc.sum()} ({inc.mean():.0%}) | aceptadas: {acept.sum()}")
    print(f"  exactitud sin filtro: {(pred == y[test]).mean():.3f} | solo en las aceptadas: {(pred[acept] == y[test][acept]).mean():.3f}")
    joblib.dump({"red": args.red, "entrada": args.entrada, "segmentador": args.segmentador, "modelo": m,
                 "clases": ["Papilionidae", "Pieridae"], "covarianza": cov, "umbral_distancia": umbral,
                 "umbral_confianza": args.umbral_confianza, "area_min": area_min, "area_max": area_max,
                 "C": C, "percentil": args.percentil, "metricas_test": metricas}, out / "modelo.joblib")
    print(f"\nGuardado: {(out / 'modelo.joblib').resolve()}")


if __name__ == "__main__":
    main()
