#!/usr/bin/env python3
"""
Standardize_features.py
-----------------------
Paso intermedio entre Split_dataset.py y Train_random_forest.py.

1. Lee features_train/val/test.csv (dataset_split/).
2. Imputa NaN con la mediana de TRAIN y estandariza (z-score) con media y desviación
   de TRAIN. val y test se transforman con esos mismos valores: ajustar el escalador
   con todo el dataset filtraría información del test.
3. Calcula, a partir del vector ya estandarizado, métricas por grupo de features
   (color: histograma HSV, momentos; textura: GLCM, Haralick, Gabor): media y
   desviación de los z-scores del grupo, más agregados "color" y "textura".
4. Guarda features_<split>_std.csv (identificadores + features estandarizadas +
   métricas) y scaler.joblib en --output.

Uso:
    python Standardize_features.py
    python Standardize_features.py --split-dir "ruta\\dataset_split" --output "ruta\\salida"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from Train_random_forest import (COLOR_GROUPS, DEFAULT_SPLIT_DIR, FEATURE_GROUPS, NON_FEATURE_COLS,
                                 TEXTURE_GROUPS, SPLITS, feature_group)


def group_metrics(z: pd.DataFrame) -> pd.DataFrame:
    """Media y desviación de los z-scores por grupo y por tipo (color / textura), por imagen."""
    by_group: dict = {}
    for col in z.columns:
        by_group.setdefault(feature_group(col), []).append(col)
    by_group.pop("Otras", None)
    sets = dict(by_group)
    sets["Color"] = [c for g in COLOR_GROUPS for c in by_group.get(g, [])]
    sets["Textura"] = [c for g in TEXTURE_GROUPS for c in by_group.get(g, [])]
    out = {}
    for name, cols in sets.items():
        if not cols:
            continue
        key = name.lower().replace(" ", "_")
        out[f"m_{key}_mean"] = z[cols].mean(axis=1)
        out[f"m_{key}_std"] = z[cols].std(axis=1)
    return pd.DataFrame(out, index=z.index)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-dir", default=str(DEFAULT_SPLIT_DIR), help="Carpeta con features_train/val/test.csv")
    ap.add_argument("--output", default=None, help="Carpeta de salida (por defecto <split-dir>/standardized)")
    args = ap.parse_args()

    split_dir = Path(args.split_dir)
    out_dir = Path(args.output) if args.output else split_dir / "standardized"
    frames = {}
    for s in SPLITS:
        path = split_dir / f"features_{s}.csv"
        if not path.exists():
            print(f"[ERROR] No se encontró {path}. Corre primero Split_dataset.py.")
            sys.exit(1)
        frames[s] = pd.read_csv(path)

    train = frames["train"]
    cols = [c for c in train.columns
            if c not in NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(train[c]) and train[c].notna().any()
            and all(c in frames[s].columns for s in SPLITS)]
    id_cols = [c for c in train.columns if c in NON_FEATURE_COLS]
    print(f"{len(cols)} features numéricas; columnas de identificación: {id_cols}")

    medians = train[cols].median()
    scaler = StandardScaler().fit(train[cols].fillna(medians))

    out_dir.mkdir(parents=True, exist_ok=True)
    for s, df in frames.items():
        z = pd.DataFrame(scaler.transform(df[cols].fillna(medians)), columns=cols, index=df.index)
        result = pd.concat([df[[c for c in id_cols if c in df.columns]], z, group_metrics(z)], axis=1)
        result.to_csv(out_dir / f"features_{s}_std.csv", index=False)
        print(f"  features_{s}_std.csv: {len(result)} filas, {result.shape[1]} columnas")

    # autoverificación: en train cada columna no constante queda con media 0 y desviación 1
    z_train = pd.read_csv(out_dir / "features_train_std.csv")[cols]
    nonconst = [c for c in cols if train[c].fillna(medians[c]).std(ddof=0) > 0]
    assert np.allclose(z_train[nonconst].mean(), 0, atol=1e-6), "la media en train no es 0"
    assert np.allclose(z_train[nonconst].std(ddof=0), 1, atol=1e-6), "la desviación en train no es 1"
    const = [c for c in cols if c not in nonconst]
    if const:
        print(f"[AVISO] {len(const)} columnas constantes en train (quedan en 0): {const[:5]}")

    joblib.dump({"scaler": scaler, "medians": medians, "feature_cols": cols}, out_dir / "scaler.joblib")
    print(f"\nListo. Salida en: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
