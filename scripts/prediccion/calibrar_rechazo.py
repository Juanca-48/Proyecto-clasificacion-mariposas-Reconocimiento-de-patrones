#!/usr/bin/env python3
"""
calibrar_rechazo.py
-------------------
Calibra los tres umbrales con los que inferencia.py rechaza imágenes que no parecen una mariposa
de las familias del proyecto. TODO se calibra con train + val; el test solo se usa para medir.

  Capa 1  rango del área de la máscara: de 0,8 x el mínimo a 1,1 x el máximo visto en train + val.
  Capa 2  distancia de Mahalanobis (covarianza Ledoit-Wolf, espacio estandarizado del modelo). El umbral es
          el percentil --percentil de las distancias de una CV AGRUPADA POR ESPECIE: así se miden
          imágenes de especies que el ajuste no vio, y no se rechazan especies nuevas legítimas.
  Capa 3  umbral de confianza (--umbral-confianza); se muestra la cobertura/exactitud de la CV para elegirlo.

Después evalúa: (a) en test, qué fracción de imágenes legítimas se rechaza y cómo cambia la exactitud;
(b) de punta a punta con fotos originales de test (debe reproducir las predicciones guardadas);
(c) con imágenes que no son mariposas (las de ejemplo de scikit-image). Es una comprobación de cordura
con pocas imágenes, no una evaluación: las mariposas de otras familias NO se rechazan de forma fiable.

Salida en resultados/rechazo/: rechazo.joblib (lo usa la app) y resumen.txt.

Uso:
    python scripts/prediccion/calibrar_rechazo.py
    python scripts/prediccion/calibrar_rechazo.py --percentil 99 --umbral-confianza 0.7
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
from sklearn.base import clone
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import StratifiedGroupKFold

from rutas import DATASET_DIR, FINAL_DIR, RESULTS_DIR, SEGMENTATION_DIR, SPLIT_DIR
from Train_random_forest import SEED, prepare_frame

OUT_DIR = RESULTS_DIR / "rechazo"
NEGATIVOS = ["astronaut", "camera", "chelsea", "coffee", "coins", "horse", "moon", "page", "rocket", "text",
             "brick", "grass", "gravel", "colorwheel"]   # imágenes de ejemplo incluidas con scikit-image


def area_mascara(dataset: str, filename: str) -> float:
    return float((cv2.imread(str(SEGMENTATION_DIR / dataset / "masks" / filename), cv2.IMREAD_GRAYSCALE) > 0).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", default=str(FINAL_DIR / "final_model.joblib"))
    ap.add_argument("--split-dir", default=str(SPLIT_DIR))
    ap.add_argument("--output", default=str(OUT_DIR))
    ap.add_argument("--percentil", type=float, default=95, help="Percentil de la CV agrupada que fija el umbral de distancia")
    ap.add_argument("--umbral-confianza", type=float, default=0.65)
    ap.add_argument("--segmentador", choices=["grabcut", "rembg"], default="rembg",
                    help="Segmentador con el que se generaron las features de entrenamiento (preprocesar.py usa rembg). "
                         "Se guarda en rechazo.joblib y la inferencia lo usa por defecto")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--fotos-punta-a-punta", type=int, default=12, help="Fotos de test para la comprobación completa")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    lines: list = []

    def log(msg=""):
        print(msg); lines.append(str(msg))

    bundle = joblib.load(args.modelo)
    modelo, cols, clases = bundle["model"], list(bundle["features"]), list(bundle["classes"])
    split = Path(args.split_dir)
    fr = {s: prepare_frame(pd.read_csv(split / f"features_{s}.csv"), False, lambda *a: None, s) for s in ("train", "val", "test")}
    fit = pd.concat([fr["train"], fr["val"]], ignore_index=True)
    test = fr["test"]
    y = np.array([clases.index(f) for f in fit.family])
    estandarizar = lambda X: modelo.named_steps["scaler"].transform(modelo.named_steps["imputer"].transform(X))
    Z = estandarizar(fit[cols])
    log(f"Modelo: {bundle['name']} ({len(cols)} features). Calibración con train + val: {len(fit)} imágenes, "
        f"{fit.species_group.nunique()} especies.")

    # --- capa 1: área de la máscara ---
    areas = np.array([area_mascara(d, f) for d, f in zip(fit.dataset, fit.filename)])
    area_min, area_max = 0.8 * areas.min(), 1.1 * areas.max()
    log(f"\n[Capa 1] área de la máscara en train + val: mín {areas.min():.3f}, máx {areas.max():.3f} "
        f"-> rango aceptado [{area_min:.3f}, {area_max:.3f}]")

    # --- capas 2 y 3: CV agrupada por especie ---
    folds = list(StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
                 .split(Z, y, fit.species_group.values))
    d2_oof, proba_oof = np.zeros(len(fit)), np.zeros((len(fit), len(clases)))
    for tr, te in folds:
        d2_oof[te] = LedoitWolf().fit(Z[tr]).mahalanobis(Z[te])
        proba_oof[te] = clone(modelo).fit(fit.loc[tr, cols], y[tr]).predict_proba(fit.loc[te, cols])
    umbral_d2 = float(np.percentile(d2_oof, args.percentil))
    cov = LedoitWolf().fit(Z)
    d2_in = cov.mahalanobis(Z)
    log(f"\n[Capa 2] distancia de Mahalanobis^2 (mediana en el espacio estandarizado: {np.median(d2_oof):.0f})")
    log(f"  imágenes de especies NO vistas (CV agrupada): mediana {np.median(d2_oof):.0f}, "
        f"p{args.percentil:g} = {umbral_d2:.0f}  <- umbral")
    log(f"  imágenes dentro del propio ajuste: mediana {np.median(d2_in):.0f} (más cercanas, por eso no sirven para calibrar)")

    log("\n[Capa 3] cobertura y exactitud de la CV agrupada según el umbral de confianza (solo imágenes que pasan la capa 2)")
    ok = d2_oof <= umbral_d2
    conf, pred = proba_oof.max(1), proba_oof.argmax(1)
    tabla = []
    for t in (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9):
        m = ok & (conf >= t)
        tabla.append({"umbral": t, "cobertura": f"{m.sum() / ok.sum():.0%}", "exactitud_aceptadas": f"{(pred[m] == y[m]).mean():.3f}"})
    log(pd.DataFrame(tabla).to_string(index=False))
    log(f"  elegido: {args.umbral_confianza}")

    rechazo = {"area_min": float(area_min), "area_max": float(area_max), "covarianza": cov,
               "umbral_distancia": umbral_d2, "umbral_confianza": args.umbral_confianza,
               "percentil": args.percentil, "modelo": str(args.modelo), "segmentador": args.segmentador}
    joblib.dump(rechazo, out / "rechazo.joblib")

    # --- (a) test: fracción de imágenes legítimas rechazadas ---
    yt = np.array([clases.index(f) for f in test.family])
    d2_t = cov.mahalanobis(estandarizar(test[cols]))
    proba_t = modelo.predict_proba(test[cols])
    a_t = np.array([area_mascara(d, f) for d, f in zip(test.dataset, test.filename)])
    r1 = (a_t < area_min) | (a_t > area_max); r2 = ~r1 & (d2_t > umbral_d2)
    inc = ~r1 & ~r2 & (proba_t.max(1) < args.umbral_confianza)
    acept = ~r1 & ~r2 & ~inc
    pt = proba_t.argmax(1)
    log(f"\n(a) TEST: {len(test)} imágenes legítimas (24 especies; el test no intervino en la calibración)")
    log(f"  rechazadas por capa 1: {r1.sum()} | por capa 2: {r2.sum()} ({r2.mean():.0%}) | inciertas: {inc.sum()} ({inc.mean():.0%}) "
        f"| aceptadas: {acept.sum()} ({acept.mean():.0%})")
    log(f"  exactitud sin filtro: {(pt == yt).mean():.3f} | solo en las aceptadas: {(pt[acept] == yt[acept]).mean():.3f}")

    # --- (b) y (c): de punta a punta con el Clasificador ---
    from inferencia import Clasificador
    clf = Clasificador(args.modelo, out / "rechazo.joblib")
    log(f"\n(b) PUNTA A PUNTA: {args.fotos_punta_a_punta} fotos originales de test, con segmentación GrabCut incluida")
    guard = pd.read_csv(FINAL_DIR / "test_predictions.csv").set_index("filename")
    muestra = test.groupby("family", group_keys=False).sample(args.fotos_punta_a_punta // 2, random_state=args.seed)
    difs = []
    for _, r in muestra.iterrows():
        foto = DATASET_DIR / r.family / r.filename
        res = clf.predecir(cv2.cvtColor(cv2.imread(str(foto)), cv2.COLOR_BGR2RGB))
        p1 = res.probabilidades.get(clases[1], np.nan)
        difs.append(abs(p1 - guard.loc[r.filename, f"proba_{clases[1]}"]) if res.probabilidades else np.nan)
        log(f"  {r.filename[:38]:<38} real {r.family[:5]} -> {res.estado:<9} {res.familia or '-':<12} "
            f"d2={res.distancia if res.distancia is not None else float('nan'):6.0f}  área={res.area_mascara:.2f}")
    log(f"  diferencia máxima de probabilidad con las predicciones guardadas: {np.nanmax(difs):.2e} "
        "(no es cero porque GrabCut no es determinista: la máscara guardada y la recalculada tienen IoU de 0,86 a 0,99)")

    log(f"\n(c) NO MARIPOSAS: imágenes de ejemplo de scikit-image (comprobación de cordura, {len(NEGATIVOS)} imágenes)")
    import skimage.data as sd
    cuentas = {"aceptada": 0, "incierta": 0, "rechazada": 0}
    for nombre in NEGATIVOS:
        res = clf.predecir(getattr(sd, nombre)())
        cuentas[res.estado] += 1
        log(f"  {nombre:<11} -> {res.estado:<9} {(res.familia or '-'):<12} área={res.area_mascara:.2f} "
            f"d2={'-' if res.distancia is None else format(res.distancia, '.0f')}")
    log(f"  resumen: {cuentas}")

    (out / "resumen.txt").write_text("\n".join(lines), encoding="utf-8")
    log(f"\nResultados en: {out.resolve()}")


if __name__ == "__main__":
    main()
