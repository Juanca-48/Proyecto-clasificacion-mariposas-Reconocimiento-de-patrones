#!/usr/bin/env python3
"""
Train_models.py
---------------
Compara varios clasificadores de familia (Papilionidae vs Pieridae) sobre las features
ESTANDARIZADAS que genera Standardize_features.py (dataset_split/standardized/).

Mismo protocolo que Train_random_forest.py:
  1. Para cada modelo, malla pequeña de hiperparámetros con CV agrupada por especie
     sobre train + val (métrica: balanced_accuracy).
  2. Tabla comparativa de la CV (el mejor de cada modelo). El test NO interviene.
  3. Se elige el modelo con mayor balanced_accuracy en CV, se reentrena con train + val
     y se evalúa en test UNA SOLA VEZ, con IC 95 % por bootstrap por especie y contra
     el clasificador "siempre la familia mayoritaria".

Modelos: regresión logística, SVM (kernel RBF), kNN y Random Forest (parámetros fijos de
Train_random_forest.py, como referencia). Se usan solo las 147 features originales
estandarizadas, no las columnas m_* (son combinaciones de las mismas).

Uso:
    python Train_models.py
    python Train_models.py --std-dir "ruta\\standardized" --output "ruta\\resultados_modelos"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from Train_random_forest import (CV_FOLDS, DEFAULT_SPLIT_DIR, N_BOOT, NON_FEATURE_COLS, SEED, SPLITS,
                                 check_splits, cluster_bootstrap, fmt_cv, fmt_scores, make_splits,
                                 prepare_frame, run_cv, save_confusion_matrix, score_dict)

# nombre -> (estimador, malla). Las features ya vienen estandarizadas, no hay escalador aquí.
MODELS = {
    "Regresión logística": (LogisticRegression(class_weight="balanced", max_iter=5000),
                            {"C": [0.01, 0.1, 1, 10]}),
    "SVM (RBF)": (SVC(kernel="rbf", class_weight="balanced", random_state=SEED),
                  {"C": [0.1, 1, 10], "gamma": ["scale", 0.003, 0.01]}),
    "kNN": (KNeighborsClassifier(),   # kNN no admite class_weight; el desbalance se ve en balanced_accuracy
            {"n_neighbors": [3, 5, 9, 15], "weights": ["uniform", "distance"]}),
    "Random Forest (referencia)": (RandomForestClassifier(n_estimators=300, max_features=0.3, min_samples_leaf=5,
                                                          max_depth=8, class_weight="balanced", n_jobs=-1,
                                                          random_state=SEED),
                                   {}),
}


def load_std(std_dir: Path, log) -> tuple[dict, list]:
    frames = {}
    for s in SPLITS:
        path = std_dir / f"features_{s}_std.csv"
        if not path.exists():
            print(f"[ERROR] No se encontró {path}. Corre primero Standardize_features.py.")
            sys.exit(1)
        frames[s] = prepare_frame(pd.read_csv(path), False, log, s)
    first = frames["train"]
    cols = [c for c in first.columns
            if c not in NON_FEATURE_COLS and not c.startswith("m_") and pd.api.types.is_numeric_dtype(first[c])]
    return frames, cols


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--std-dir", default=str(DEFAULT_SPLIT_DIR / "standardized"))
    ap.add_argument("--output", default=str(DEFAULT_SPLIT_DIR.parent / "resultados_modelos"))
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

    frames, cols = load_std(Path(args.std_dir), log)
    check_splits(frames, log)
    fit_df = pd.concat([frames["train"], frames["val"]], ignore_index=True)
    test_df = frames["test"]
    X, Xte = fit_df[cols], test_df[cols]
    assert not X.isna().any().any() and not Xte.isna().any().any(), "hay NaN: ¿corriste Standardize_features.py?"
    enc = LabelEncoder().fit(fit_df["family"])
    y, yte = enc.transform(fit_df["family"]), enc.transform(test_df["family"])
    groups = fit_df["species_group"].values
    classes = enc.classes_
    log(f"\nFeatures estandarizadas: {len(cols)}. Train+val: {len(y)} imágenes, {len(set(groups))} especies. "
        f"Test: {len(yte)} imágenes.")

    # --- 1) ajuste por modelo y comparación en CV agrupada ---
    n_splits = min(args.cv_folds, int(fit_df.groupby("family")["species_group"].nunique().min()))
    splits = make_splits("group", X, y, groups, n_splits, args.seed)
    log(f"\n=== COMPARACIÓN DE MODELOS (CV agrupada por especie, {n_splits} folds, train + val) ===")
    fitted, rows = {}, []
    for name, (est, grid) in MODELS.items():
        gs = GridSearchCV(est, grid or {}, scoring="balanced_accuracy", cv=splits, n_jobs=1, refit=True)
        gs.fit(X, y)
        cv = run_cv(gs.best_estimator_, X, y, groups, "group", n_splits, args.seed, 2)
        fitted[name] = gs
        rows.append({"modelo": name, "params": gs.best_params_, **cv.mean().round(3).to_dict(),
                     "bal_acc_std": round(cv["balanced_accuracy"].std(), 3)})
        log(f"{name:<28} {fmt_cv(cv)}   mejores params: {gs.best_params_ or 'fijos'}")
    table = pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False)
    table.to_csv(out_dir / "comparacion_modelos_cv.csv", index=False)
    best_name = table.iloc[0]["modelo"]
    log(f"\nMejor en CV: {best_name}. Si la diferencia con los demás es menor que la desviación entre folds "
        f"(columna bal_acc_std), no es una diferencia fiable.")

    # --- 2) TEST, una sola vez, solo el modelo elegido ---
    model = fitted[best_name].best_estimator_  # GridSearchCV ya lo reentrenó con todo train + val
    pred = model.predict(Xte)
    # SVC sin probability: el AUC solo necesita el orden, así que sirve decision_function
    proba = model.predict_proba(Xte) if hasattr(model, "predict_proba") else np.column_stack(
        [-model.decision_function(Xte), model.decision_function(Xte)])
    log(f"\n=== TEST ({best_name}, entrenado con train + val; una sola evaluación) ===")
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

    save_confusion_matrix(yte, pred, classes, out_dir / "confusion_matrix.png", f"Test: {best_name}")
    pd.DataFrame({"filename": test_df["filename"], "species_group": test_df["species_group"],
                  "real": classes[yte], "predicha": classes[pred], "acierto": pred == yte}
                 ).to_csv(out_dir / "test_predictions.csv", index=False)
    joblib.dump({"model": model, "features": cols, "classes": classes, "name": best_name,
                 "params": fitted[best_name].best_params_, "trained_on": "train+val (estandarizado)"},
                out_dir / "model_best.joblib")
    (out_dir / "metrics_report.txt").write_text("\n".join(lines), encoding="utf-8")
    log(f"\nResultados en: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
