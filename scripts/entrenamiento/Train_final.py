#!/usr/bin/env python3
"""
Train_final.py
--------------
Modelo final de familia (Papilionidae vs Pieridae). Entrena UN modelo elegido y lo guarda
como un solo pipeline que recibe features SIN estandarizar:

    imputación por mediana -> StandardScaler -> clasificador

Todo se ajusta con train + val (el escalador también), así el archivo guardado funciona
sobre features nuevas sin depender de Standardize_features.py.

Pasos:
  1. Hiperparámetros del modelo elegido con CV agrupada por especie sobre train + val.
  2. Reentrenamiento con train + val y evaluación en test, con IC 95 % por bootstrap por
     especie y comparación contra "siempre la familia mayoritaria".
  3. Se guarda final_model.joblib y se comprueba que recargado da las mismas predicciones.

El modelo por defecto es la regresión logística: en Train_models.py empata con el SVM
(diferencia mucho menor que la desviación entre folds) y es más simple. Esa elección se
tomó con la CV de train + val; el test ya se vio en las corridas de Train_models.py, así
que NO debe usarse para seguir ajustando.

Uso:
    python Train_final.py
    python Train_final.py --model "SVM (RBF)"
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

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from rutas import FINAL_DIR
from Train_models import MODELS
from Train_random_forest import (CV_FOLDS, DEFAULT_SPLIT_DIR, N_BOOT, SEED, SPLITS, check_splits,
                                 cluster_bootstrap, fmt_scores, make_splits, prepare_frame,
                                 save_confusion_matrix, score_dict, select_feature_cols)


def proba_of(model, X):
    """Probabilidades, o decision_function como 2 columnas (el AUC solo necesita el orden)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    d = model.decision_function(X)
    return np.column_stack([-d, d])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Regresión logística", choices=list(MODELS))
    ap.add_argument("--split-dir", default=str(DEFAULT_SPLIT_DIR))
    ap.add_argument("--output", default=str(FINAL_DIR))
    ap.add_argument("--cv-folds", type=int, default=CV_FOLDS)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines: list = []

    def log(msg=""):
        print(msg)
        lines.append(str(msg))

    split_dir = Path(args.split_dir)
    frames = {}
    for s in SPLITS:
        path = split_dir / f"features_{s}.csv"
        if not path.exists():
            print(f"[ERROR] No se encontró {path}. Corre primero Split_dataset.py.")
            sys.exit(1)
        frames[s] = prepare_frame(pd.read_csv(path), False, log, s)
    check_splits(frames, log)

    fit_df = pd.concat([frames["train"], frames["val"]], ignore_index=True)
    test_df = frames["test"]
    cols = select_feature_cols(list(frames.values()), fit_df, log)
    X, Xte = fit_df[cols], test_df[cols]
    enc = LabelEncoder().fit(fit_df["family"])
    y, yte = enc.transform(fit_df["family"]), enc.transform(test_df["family"])
    groups, classes = fit_df["species_group"].values, enc.classes_
    log(f"\nModelo: {args.model}. {len(cols)} features sin estandarizar. "
        f"Train+val: {len(y)} imágenes, {len(set(groups))} especies. Test: {len(yte)} imágenes.")

    est, grid = MODELS[args.model]
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()),
                     ("clf", clone(est))])
    n_splits = min(args.cv_folds, int(fit_df.groupby("family")["species_group"].nunique().min()))
    splits = make_splits("group", X, y, groups, n_splits, args.seed)
    gs = GridSearchCV(pipe, {f"clf__{k}": v for k, v in grid.items()} or {}, scoring="balanced_accuracy",
                      cv=splits, n_jobs=1, refit=True)
    gs.fit(X, y)
    params = {k.removeprefix("clf__"): v for k, v in gs.best_params_.items()}
    best = gs.cv_results_["mean_test_score"].argmax()
    log(f"\n=== CV agrupada por especie ({n_splits} folds, train + val) ===")
    log(f"Hiperparámetros elegidos: {params or 'fijos'}. "
        f"balanced_accuracy={gs.best_score_:.3f}±{gs.cv_results_['std_test_score'][best]:.3f}")

    model = gs.best_estimator_   # ya reentrenado con todo train + val
    pred, proba = model.predict(Xte), proba_of(model, Xte)
    log(f"\n=== TEST (modelo final entrenado con train + val) ===")
    log(f"Test: {len(yte)} imágenes, {test_df['species_group'].nunique()} especies")
    log("\n" + classification_report(yte, pred, target_names=classes, digits=3, zero_division=0))
    scores = score_dict(yte, pred, proba, 2)
    log("Métricas: " + fmt_scores(scores))
    ci = cluster_bootstrap(yte, pred, proba, test_df["species_group"].values, 2, args.n_boot, args.seed)
    log(f"IC 95 % (bootstrap por especie, {args.n_boot} remuestreos):")
    for k in ("balanced_accuracy", "f1_macro", "auc"):
        log(f"  {k:<18} {scores[k]:.3f}  [{ci[k][0]:.3f}, {ci[k][1]:.3f}]")
    dpred = DummyClassifier(strategy="most_frequent").fit(X, y).predict(Xte)
    log(f"Referencia, siempre la familia mayoritaria: balanced_accuracy={balanced_accuracy_score(yte, dpred):.3f}")
    log("\nMatriz de confusión (filas = real, columnas = predicho):")
    log(pd.DataFrame(confusion_matrix(yte, pred), index=[f"real {c}" for c in classes],
                     columns=[f"pred {c}" for c in classes]).to_string())

    save_confusion_matrix(yte, pred, classes, out_dir / "confusion_matrix.png", f"Test: {args.model}")
    pd.DataFrame({"filename": test_df["filename"], "species_group": test_df["species_group"],
                  "real": classes[yte], "predicha": classes[pred], "acierto": pred == yte,
                  f"proba_{classes[1]}": proba[:, 1]}).to_csv(out_dir / "test_predictions.csv", index=False)

    path = out_dir / "final_model.joblib"
    joblib.dump({"model": model, "features": cols, "classes": classes, "name": args.model,
                 "params": params, "trained_on": "train+val"}, path)
    bundle = joblib.load(path)   # autoverificación: el archivo guardado reproduce las predicciones
    assert (bundle["model"].predict(Xte[bundle["features"]]) == pred).all(), "el modelo recargado difiere"
    (out_dir / "metrics_report.txt").write_text("\n".join(lines), encoding="utf-8")
    log(f"\nModelo guardado y verificado: {path.resolve()}")


if __name__ == "__main__":
    main()
