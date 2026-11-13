#!/usr/bin/env python3
"""
Train_random_forest.py
----------------------------
Entrena y evalúa un Random Forest para distinguir FAMILIAS de mariposas
(Papilionidae vs Pieridae) a partir de las características de textura y
color. La familia se deduce de la columna "dataset":
    Dataset_papilionidae(_v2) -> Papilionidae
    Dataset_pieridae(_v2)     -> Pieridae

Tiene dos modos, y elige solo cuál usar:

  MODO DIVIDIDO (el que debes usar): lee features_train.csv, features_val.csv
  y features_test.csv que genera Split_dataset.py (por defecto en
  resultados/dataset_split/). Protocolo:

    1. Verifica que no haya imágenes repetidas entre particiones y avisa si
       alguna especie aparece en más de una (fuga entre train y test).
    2. AJUSTE DE HIPERPARÁMETROS con una malla pequeña, usando validación
       cruzada agrupada por especie sobre train + val. Entre las
       configuraciones que quedan dentro de un error estándar de la mejor,
       elige la más simple (regla de 1 error estándar): con pocos datos,
       la más compleja casi siempre solo sobreajusta la validación.
    3. Evalúa en val un modelo entrenado solo con train (control).
    4. ABLACIÓN (solo textura, solo color, ambas) con validación cruzada
       sobre train + val. El test NO se usa para esto.
    5. TEST, UNA SOLA VEZ: el modelo final se entrena con train + val y se
       evalúa en test. Se reporta con intervalos de confianza al 95 %
       (bootstrap por especie, porque con pocas decenas de imágenes un solo
       número da falsa precisión) y se compara con un clasificador que
       siempre responde la familia mayoritaria.
    6. IMPORTANCIA de features por permutación, medida en val con el modelo
       entrenado solo con train (así el test no interviene en la
       interpretación), más la importancia agregada por grupo de features.

  MODO UN SOLO CSV (exploratorio): si pasas --csv, o si no existe la
  división, usa un único CSV (features_dataset_ready.csv), compara
  validación cruzada aleatoria vs agrupada por especie, hace la ablación
  y un hold-out interno. Útil para explorar, pero su "test" sale del mismo
  CSV y no es la división oficial del proyecto.

Notas:
  - Random Forest no necesita escalar las features.
  - fg_pixels se excluye de las features a propósito: depende del tamaño
    de la imagen, que podría diferir entre catálogos y darle al modelo un
    atajo que no tiene que ver con la mariposa.
  - Los NaN se imputan con la mediana, ajustada solo con los datos de
    entrenamiento de cada partición.
  - Se usa class_weight="balanced" y se reportan balanced_accuracy y F1 macro,
    que no se inflan por la familia mayoritaria.

Requisitos:
    pip install scikit-learn pandas numpy matplotlib joblib

Uso:
    python Train_random_forest.py                       (modo dividido si existe resultados/dataset_split/)
    python Train_random_forest.py --split-dir "ruta\\dataset_split" --output "ruta\\resultados_rf"
    python Train_random_forest.py --no-tune             (omite la búsqueda de hiperparámetros)
    python Train_random_forest.py --csv "ruta\\features_dataset_ready.csv"   (modo un solo CSV)

Para usar el modelo guardado:
    import joblib
    bundle = joblib.load("model_family_rf.joblib")
    pred = bundle["model"].predict(df_nuevo[bundle["features"]])
    familias = bundle["classes"][pred]
"""

from __future__ import annotations

# Localiza rutas.py (en scripts/, o junto a este archivo en la app) y agrega al path la carpeta de cada proceso
import sys
from pathlib import Path
sys.path[:0] = [str(p) for p in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]) if (p / "rutas.py").exists()][:1]
import rutas  # noqa: E402,F401


import argparse
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# ----------------------------------------------------------------------
# CONFIG (usado si no se pasan argumentos por línea de comandos)
# ----------------------------------------------------------------------
from rutas import RF_DIR, SEGMENTATION_DIR, SPLIT_DIR

DEFAULT_SPLIT_DIR = SPLIT_DIR
DEFAULT_CSV = SEGMENTATION_DIR / "features_dataset_ready.csv"
DEFAULT_OUTPUT_DIR = RF_DIR

N_ESTIMATORS = 500
CV_FOLDS = 5
TEST_SIZE = 0.2          # solo para el modo un solo CSV
SEED = 42
N_BOOT = 1000

SPLITS = ["train", "val", "test"]

# Malla pequeña a propósito: con pocos datos, una búsqueda grande solo sobreajusta la validación.
TUNING_GRID = {
    "rf__max_features": ["sqrt", "log2", 0.3],
    "rf__min_samples_leaf": [1, 2, 5],
    "rf__max_depth": [None, 8],
}

NON_FEATURE_COLS = {"filename", "dataset", "label", "genus", "fg_pixels",
                    "low_quality", "family", "species_group"}

FEATURE_GROUPS = {
    "glcm_": "GLCM",
    "haralick_": "Haralick",
    "gabor_": "Gabor",
    "hist_": "Histograma HSV",
    "moment_": "Momentos de color",
    "shape_": "Forma",
}
TEXTURE_GROUPS = {"GLCM", "Haralick", "Gabor"}
COLOR_GROUPS = {"Histograma HSV", "Momentos de color"}


# ----------------------------------------------------------------------
# Preparación de datos
# ----------------------------------------------------------------------

def derive_family(dataset_value) -> str:
    """'Dataset_papilionidae_v2' -> 'Papilionidae'."""
    s = str(dataset_value).strip().lower()
    s = re.sub(r"^dataset[_\- ]*", "", s)
    s = re.sub(r"[_\- ]*v\d+$", "", s)
    return s.strip("_- ").capitalize()


def derive_species_group(label, fallback) -> str:
    """Primeros dos tokens del label ('Battus crassus lepidus' -> 'battus crassus'),
    para que subespecies de una misma especie queden en el mismo grupo."""
    if not isinstance(label, str) or not label.strip():
        return f"__sin_label__{fallback}"
    tokens = re.split(r"[ _]+", label.strip())
    return " ".join(tokens[:2]).lower()


def label_from_filename(filename: str):
    """'013_Dismorphia_altis_altis.png' -> 'Dismorphia_altis_altis'; None si no hay 2 tokens."""
    tokens = [t for t in re.split(r"[ _]+", re.sub(r"^\d+_", "", Path(filename).stem)) if t]
    return "_".join(tokens) if len(tokens) >= 2 else None


def feature_group(col: str) -> str:
    for prefix, name in FEATURE_GROUPS.items():
        if col.startswith(prefix):
            return name
    return "Otras"


def prepare_frame(df: pd.DataFrame, keep_low_quality: bool, log, name: str = "") -> pd.DataFrame:
    """Deduce familia y grupo de especie, y descarta low_quality y duplicados."""
    tag = f"[{name}] " if name else ""
    for col in ("dataset", "filename"):
        if col not in df.columns:
            print(f"[ERROR] {tag}Falta la columna '{col}' en el CSV.")
            sys.exit(1)
    df = df.copy()
    df["family"] = df["dataset"].apply(derive_family)

    if "low_quality" in df.columns and not keep_low_quality:
        lq = df["low_quality"].astype(str).str.lower().isin(["true", "1"])
        if lq.any():
            log(f"{tag}Se descartan {int(lq.sum())} filas marcadas low_quality (usa --keep-low-quality para conservarlas).")
        df = df[~lq].copy()

    n_dups = int(df.duplicated(subset=["family", "filename"]).sum())
    if n_dups:
        log(f"[AVISO] {tag}{n_dups} filas repetían (familia, filename); seguramente mezclaste versiones v1 y v2. "
            f"Se conserva una por imagen (la última por orden alfabético de 'dataset').")
        df = df.sort_values("dataset").drop_duplicates(subset=["family", "filename"], keep="last")

    # Los CSV de features no traen "label": la especie se lee del nombre del archivo
    # (001_Battus_crassus_crassus.png), igual que en Split_dataset.py. Sin esto cada imagen
    # sería su propio grupo y la CV "agrupada" y el bootstrap por especie no agruparían nada.
    labels = df["label"] if "label" in df.columns else df["filename"].map(label_from_filename)
    df["species_group"] = [derive_species_group(lab, fname) for lab, fname in zip(labels, df["filename"])]
    return df.reset_index(drop=True)


def select_feature_cols(frames: list, fit_df: pd.DataFrame, log) -> list:
    """Columnas numéricas presentes en todos los frames y con algún dato en los
    datos de ajuste (si una columna está vacía al entrenar, el imputador la descartaría)."""
    first = frames[0]
    cols = [c for c in first.columns
            if c not in NON_FEATURE_COLS
            and all(c in f.columns for f in frames)
            and all(pd.api.types.is_numeric_dtype(f[c]) for f in frames)]
    empty = [c for c in cols if fit_df[c].isna().all()]
    if empty:
        log(f"Se descartan {len(empty)} columnas completamente vacías: {empty[:5]}{' ...' if len(empty) > 5 else ''}")
    return [c for c in cols if c not in empty]


def load_data(csv_path: Path, keep_low_quality: bool, log):
    """Modo un solo CSV."""
    if not csv_path.exists():
        print(f"[ERROR] No se encontró el CSV: {csv_path}")
        print("Genera primero features_dataset_ready.csv con prepare_dataset.py, o pasa --csv.")
        sys.exit(1)
    raw = pd.read_csv(csv_path).copy()  # .copy() desfragmenta el DataFrame (muchas columnas)
    log(f"CSV cargado: {len(raw)} filas, {raw.shape[1]} columnas ({csv_path.name})")
    df = prepare_frame(raw, keep_low_quality, log)
    if "label" not in df.columns:
        log("[AVISO] No hay columna 'label': no se puede agrupar por especie, se usará división aleatoria.")
    feature_cols = select_feature_cols([df], df, log)
    return df, feature_cols


# ----------------------------------------------------------------------
# Modelo y evaluación
# ----------------------------------------------------------------------

def make_model(n_estimators: int, seed: int, rf_params: dict | None = None) -> Pipeline:
    params = dict(rf_params or {})
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced",   # compensa si una familia tiene más fotos que la otra
            n_jobs=-1,
            random_state=seed,
            **params,
        )),
    ])


def make_splits(strategy: str, X, y, groups, n_splits: int, seed: int):
    if strategy == "group":
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(cv.split(X, y, groups))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(X, y))


def run_cv(model, X, y, groups, strategy: str, n_splits: int, seed: int, n_classes: int) -> pd.DataFrame:
    auc_name = "roc_auc" if n_classes == 2 else "roc_auc_ovr"
    scoring = {"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy",
               "f1_macro": "f1_macro", "auc": auc_name}
    splits = make_splits(strategy, X, y, groups, n_splits, seed)
    res = cross_validate(model, X, y, cv=splits, scoring=scoring, n_jobs=1)
    return pd.DataFrame({m: res[f"test_{m}"] for m in scoring})


def fmt_cv(df: pd.DataFrame) -> str:
    return "  ".join(f"{c}={df[c].mean():.3f}±{df[c].std():.3f}" for c in df.columns)


def holdout_split(X, y, groups, test_size: float, mode: str, seed: int):
    if mode == "group":
        k = max(2, int(round(1 / test_size)))
        cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        train_idx, test_idx = next(cv.split(X, y, groups))
        return train_idx, test_idx
    idx = np.arange(len(y))
    return train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)


def score_dict(y_true, y_pred, proba, n_classes: int) -> dict:
    out = {"accuracy": float((np.asarray(y_true) == np.asarray(y_pred)).mean()),
           "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
           "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0))}
    if len(set(y_true)) == n_classes:
        out["auc"] = float(roc_auc_score(y_true, proba[:, 1]) if n_classes == 2
                           else roc_auc_score(y_true, proba, multi_class="ovr"))
    else:
        out["auc"] = float("nan")
    return out


def fmt_scores(d: dict) -> str:
    return "  ".join(f"{k}={v:.3f}" for k, v in d.items())


# ----------------------------------------------------------------------
# Ajuste de hiperparámetros y bootstrap (modo dividido)
# ----------------------------------------------------------------------

def tune_rf(X, y, groups, n_splits: int, seed: int, n_estimators: int):
    """Búsqueda en malla con CV agrupada por especie + regla de 1 error estándar."""
    splits = make_splits("group", X, y, groups, n_splits, seed)
    gs = GridSearchCV(make_model(n_estimators, seed), TUNING_GRID, scoring="balanced_accuracy",
                      cv=splits, n_jobs=1, refit=False)
    gs.fit(X, y)
    res = pd.DataFrame(gs.cv_results_)
    res["se"] = res["std_test_score"] / np.sqrt(len(splits))

    best = res["mean_test_score"].idxmax()
    threshold = res.loc[best, "mean_test_score"] - res.loc[best, "se"]
    cand = res[res["mean_test_score"] >= threshold].copy()

    def depth_rank(d):
        return 10 ** 6 if d is None or (isinstance(d, float) and np.isnan(d)) else int(d)

    cand["depth_rank"] = cand["param_rf__max_depth"].apply(depth_rank)
    cand["leaf"] = cand["param_rf__min_samples_leaf"].astype(int)
    chosen = cand.sort_values(["depth_rank", "leaf", "mean_test_score"], ascending=[True, False, False]).iloc[0]

    mf = chosen["param_rf__max_features"]
    md = chosen["param_rf__max_depth"]
    params = {
        "max_features": float(mf) if isinstance(mf, (int, float, np.floating)) else str(mf),
        "min_samples_leaf": int(chosen["param_rf__min_samples_leaf"]),
        "max_depth": None if depth_rank(md) == 10 ** 6 else int(md),
    }
    return params, res, int(best), int(chosen.name), threshold


def cluster_bootstrap(y_true, y_pred, proba, groups, n_classes: int, n_boot: int, seed: int) -> dict:
    """IC 95 % remuestreando especies completas (no imágenes sueltas), porque las
    fotos de una misma especie no son independientes entre sí."""
    rng = np.random.default_rng(seed)
    y_true, y_pred, groups = np.asarray(y_true), np.asarray(y_pred), np.asarray(groups)
    uniq = pd.unique(groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in uniq}
    stats = {"balanced_accuracy": [], "f1_macro": [], "auc": []}
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([idx_by_group[uniq[k]] for k in pick])
        yt = y_true[idx]
        if len(np.unique(yt)) < n_classes:
            continue
        stats["balanced_accuracy"].append(balanced_accuracy_score(yt, y_pred[idx]))
        stats["f1_macro"].append(f1_score(yt, y_pred[idx], average="macro", zero_division=0))
        stats["auc"].append(roc_auc_score(yt, proba[idx, 1]) if n_classes == 2
                            else roc_auc_score(yt, proba[idx], multi_class="ovr"))
    out = {}
    for k, v in stats.items():
        out[k] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v)) if v else (float("nan"),) * 2 + (0,)
    return out


# ----------------------------------------------------------------------
# Gráficos
# ----------------------------------------------------------------------

def save_confusion_matrix(y_true, y_pred, class_names, path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, norm, sub in zip(axes, [None, "true"], ["Conteos", "Normalizada por fila (recall)"]):
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, labels=list(range(len(class_names))), display_labels=class_names, normalize=norm,
            cmap="Blues", ax=ax, colorbar=False,
            values_format=".2f" if norm else "d")
        ax.set_title(sub)
        ax.set_xlabel("Familia predicha")
        ax.set_ylabel("Familia real")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_importance_plot(imp_df: pd.DataFrame, group_imp: pd.Series, path: Path, where: str, top_n: int = 20):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = imp_df.sort_values("perm_importance_mean", ascending=False).head(top_n)[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [2, 1]})
    axes[0].barh(top["feature"], top["perm_importance_mean"], xerr=top["perm_importance_std"])
    axes[0].set_title(f"Top {top_n} features (importancia por permutación, {where})")
    axes[0].set_xlabel("Caída de balanced accuracy al permutar la feature")
    g = group_imp.sort_values()
    axes[1].barh(g.index, g.values)
    axes[1].set_title("Importancia agregada por grupo\n(impureza, suma = 1)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def importance_tables(model, Xeval, yeval, feature_cols, seed: int, log):
    perm = permutation_importance(model, Xeval, yeval, n_repeats=10, random_state=seed,
                                  scoring="balanced_accuracy", n_jobs=1)
    rf = model.named_steps["rf"]
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "group": [feature_group(c) for c in feature_cols],
        "impurity_importance": rf.feature_importances_,
        "perm_importance_mean": perm.importances_mean,
        "perm_importance_std": perm.importances_std,
    }).sort_values("perm_importance_mean", ascending=False)
    return imp_df


# ----------------------------------------------------------------------
# MODO DIVIDIDO
# ----------------------------------------------------------------------

def check_splits(frames: dict, log):
    names = list(frames)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = frames[names[i]], frames[names[j]]
            same_img = set(zip(a["family"], a["filename"])) & set(zip(b["family"], b["filename"]))
            if same_img:
                print(f"[ERROR] {len(same_img)} imágenes aparecen a la vez en '{names[i]}' y '{names[j]}'. "
                      f"La división está contaminada; vuelve a generarla con Split_dataset.py.")
                sys.exit(1)
            real = lambda f: {k for k in zip(f["family"], f["species_group"]) if not k[1].startswith("__sin_label__")}
            shared = real(a) & real(b)
            if shared:
                log(f"[AVISO] {len(shared)} especies aparecen en '{names[i]}' y en '{names[j]}' (fuga por especie). "
                    f"Las métricas serán optimistas; usa Split_dataset.py con --split-mode group.")
    log("Verificación: ninguna imagen se repite entre particiones.")


def run_split(args, split_dir: Path, out_dir: Path, log):
    frames = {}
    for s in SPLITS:
        raw = pd.read_csv(split_dir / f"features_{s}.csv").copy()
        frames[s] = prepare_frame(raw, args.keep_low_quality, log, s)

    log("\n=== DATOS ===")
    counts = pd.concat({s: f["family"].value_counts() for s, f in frames.items()}, axis=1).fillna(0).astype(int)
    counts = counts[SPLITS]
    counts["total"] = counts.sum(axis=1)
    log("Imágenes por familia y partición:\n" + counts.to_string())
    spp = pd.concat({s: f.groupby("family")["species_group"].nunique() for s, f in frames.items()}, axis=1).fillna(0).astype(int)
    log("Especies (grupos) por familia y partición:\n" + spp[SPLITS].to_string())
    check_splits(frames, log)

    fit_df = pd.concat([frames["train"], frames["val"]], ignore_index=True)
    test_df = frames["test"]
    feature_cols = select_feature_cols(list(frames.values()), fit_df, log)
    log(f"Features usadas: {len(feature_cols)}")

    encoder = LabelEncoder().fit(pd.concat([fit_df["family"], test_df["family"]]))
    class_names = encoder.classes_
    n_classes = len(class_names)
    if n_classes < 2:
        print("[ERROR] Se necesitan al menos 2 familias para clasificar.")
        sys.exit(1)
    for s, f in frames.items():
        missing = set(class_names) - set(f["family"])
        if missing:
            log(f"[AVISO] La partición '{s}' no tiene imágenes de {sorted(missing)}; sus métricas no serán confiables.")

    Xtr, ytr = frames["train"][feature_cols], encoder.transform(frames["train"]["family"])
    Xva, yva = frames["val"][feature_cols], encoder.transform(frames["val"]["family"])
    Xfit, yfit, gfit = fit_df[feature_cols], encoder.transform(fit_df["family"]), fit_df["species_group"].values
    Xte, yte = test_df[feature_cols], encoder.transform(test_df["family"])

    fit_counts = fit_df["family"].value_counts()
    fit_groups = fit_df.groupby("family")["species_group"].nunique()
    n_splits = min(args.cv_folds, int(fit_counts.min()), int(fit_groups.min()))
    if n_splits < 2:
        print("[ERROR] Muy pocos datos por clase/especie en train + val para validar de forma cruzada.")
        sys.exit(1)
    if n_splits < args.cv_folds:
        log(f"[AVISO] Se reducen los folds de {args.cv_folds} a {n_splits} por falta de datos en alguna clase.")
    if fit_counts.max() / fit_counts.min() > 1.5:
        log(f"[AVISO] Clases desbalanceadas en train + val ({fit_counts.max()} vs {fit_counts.min()}): "
            f"se usa class_weight='balanced' y se reportan métricas que no se inflan por la clase mayoritaria.")

    # --- 1) Hiperparámetros ---
    log(f"\n=== AJUSTE DE HIPERPARÁMETROS (CV agrupada por especie, {n_splits} folds, train + val) ===")
    if args.no_tune:
        params = {}
        (out_dir / "cv_grid_results.csv").unlink(missing_ok=True)   # evita dejar un resultado viejo
        log("--no-tune: se usan los valores por defecto de Random Forest.")
    else:
        n_combos = int(np.prod([len(v) for v in TUNING_GRID.values()]))
        log(f"Malla: {n_combos} configuraciones, métrica balanced_accuracy.")
        params, res, best_i, chosen_i, thr = tune_rf(Xfit, yfit, gfit, n_splits, args.seed, args.tune_estimators)
        cols = ["param_rf__max_features", "param_rf__min_samples_leaf", "param_rf__max_depth", "mean_test_score", "se"]
        show = res.sort_values("mean_test_score", ascending=False).head(6)[cols]
        show.columns = ["max_features", "min_samples_leaf", "max_depth", "balanced_acc", "error_estandar"]
        log("Mejores 6 configuraciones:\n" + show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        log(f"Mejor: balanced_acc={res.loc[best_i, 'mean_test_score']:.4f}. Umbral de 1 error estándar: {thr:.4f}.")
        log(f"Elegida (la más simple dentro del umbral): {params}")
        res.drop(columns=["params"], errors="ignore").to_csv(out_dir / "cv_grid_results.csv", index=False)

    # --- 2) Control en val ---
    log("\n=== VALIDACIÓN (modelo entrenado solo con train, evaluado en val) ===")
    model_tr = make_model(args.n_estimators, args.seed, params).fit(Xtr, ytr)
    val_scores = score_dict(yva, model_tr.predict(Xva), model_tr.predict_proba(Xva), n_classes)
    log(fmt_scores(val_scores) + f"   (n={len(yva)}; con tan pocas imágenes, cada error pesa mucho)")

    # --- 3) Brecha aleatoria vs agrupada + ablación (solo train + val) ---
    log("\n=== VALIDACIÓN CRUZADA SOBRE TRAIN + VAL ===")
    model_cv = make_model(args.n_estimators, args.seed, params)
    cv_group = run_cv(model_cv, Xfit, yfit, gfit, "group", n_splits, args.seed, n_classes)
    cv_random = run_cv(model_cv, Xfit, yfit, gfit, "random", n_splits, args.seed, n_classes)
    log(f"Agrupada por especie : {fmt_cv(cv_group)}")
    log(f"Aleatoria            : {fmt_cv(cv_random)}")
    gap = cv_random["balanced_accuracy"].mean() - cv_group["balanced_accuracy"].mean()
    log(f"-> Brecha de balanced_accuracy entre ambas: {gap:.3f}. "
        + ("Si es grande, parte del rendimiento aleatorio viene de reconocer especies ya vistas." if gap > 0.05
           else "Es pequeña: el modelo generaliza bien a especies no vistas."))
    pd.concat({"agrupada": cv_group, "aleatoria": cv_random}, names=["estrategia", "fold"]).to_csv(out_dir / "cv_results.csv")

    if not args.skip_ablation:
        log("\n=== ABLACIÓN (CV agrupada por especie, train + val) ===")
        subsets = {
            "Solo textura": [c for c in feature_cols if feature_group(c) in TEXTURE_GROUPS],
            "Solo color": [c for c in feature_cols if feature_group(c) in COLOR_GROUPS],
            "Solo forma": [c for c in feature_cols if feature_group(c) == "Forma"],   # vacío sin shape_*
            "Todas las features": feature_cols,
        }
        for name, cols in subsets.items():
            if not cols:
                log(f"{name:<16}: sin features disponibles")
                continue
            r = run_cv(model_cv, fit_df[cols], yfit, gfit, "group", n_splits, args.seed, n_classes)
            log(f"{name:<16} ({len(cols):>3} features): {fmt_cv(r)}")

    # --- 4) Importancia (se calcula en val con el modelo de train; se muestra al final) ---
    imp_df = importance_tables(model_tr, Xva, yva, feature_cols, args.seed, log)
    imp_df.to_csv(out_dir / "feature_importance.csv", index=False)

    # --- 5) Modelo final y TEST (una sola vez) ---
    log("\n=== TEST (una sola evaluación; modelo final entrenado con train + val) ===")
    final_model = make_model(args.n_estimators, args.seed, params).fit(Xfit, yfit)
    pred = final_model.predict(Xte)
    proba = final_model.predict_proba(Xte)
    log(f"Test: {len(yte)} imágenes, {test_df['species_group'].nunique()} grupos de especie")
    log("\n" + classification_report(yte, pred, labels=list(range(n_classes)), target_names=class_names, digits=3, zero_division=0))
    test_scores = score_dict(yte, pred, proba, n_classes)
    log("Métricas: " + fmt_scores(test_scores))

    if len(set(yte)) == n_classes:
        ci = cluster_bootstrap(yte, pred, proba, test_df["species_group"].values, n_classes, args.n_boot, args.seed)
        log(f"Intervalos de confianza al 95 % (bootstrap por especie, {args.n_boot} remuestreos):")
        for k in ("balanced_accuracy", "f1_macro", "auc"):
            lo, hi, n_ok = ci[k]
            log(f"  {k:<18} {test_scores[k]:.3f}  [{lo:.3f}, {hi:.3f}]")
        log("(Un intervalo ancho es información útil: indica cuánto puede cambiar el resultado "
            "con otro conjunto de especies de prueba.)")

    dummy = DummyClassifier(strategy="most_frequent").fit(Xfit, yfit)
    dpred = dummy.predict(Xte)
    log(f"Referencia, siempre la familia mayoritaria: balanced_accuracy={balanced_accuracy_score(yte, dpred):.3f}, "
        f"accuracy={(dpred == yte).mean():.3f}")

    cm = confusion_matrix(yte, pred, labels=list(range(n_classes)))
    log("\nMatriz de confusión (filas = real, columnas = predicho):")
    log(pd.DataFrame(cm, index=[f"real {c}" for c in class_names],
                     columns=[f"pred {c}" for c in class_names]).to_string())
    save_confusion_matrix(yte, pred, class_names, out_dir / "confusion_matrix.png",
                          "Matriz de confusión en test (modelo final)")

    pred_df = test_df[[c for c in ("filename", "dataset", "label", "species_group") if c in test_df.columns]].copy()
    pred_df["real"] = class_names[yte]
    pred_df["predicha"] = class_names[pred]
    for i, c in enumerate(class_names):
        pred_df[f"proba_{c}"] = proba[:, i]
    pred_df["acierto"] = pred == yte
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False)
    wrong = pred_df[~pred_df["acierto"]]
    log(f"\nImágenes de test mal clasificadas: {len(wrong)} de {len(pred_df)} (detalle completo en test_predictions.csv)")
    if len(wrong):
        by_sp = wrong.groupby("species_group").size().sort_values(ascending=False)
        log("Especies con más errores:\n" + by_sp.head(10).to_string())

    # --- 6) Importancia por grupo (modelo final) y gráfico ---
    group_imp = pd.Series(final_model.named_steps["rf"].feature_importances_,
                          index=[feature_group(c) for c in feature_cols]).groupby(level=0).sum()
    log("\n=== IMPORTANCIA DE FEATURES ===")
    log("Por grupo (impureza, modelo final entrenado con train + val, suma = 1):")
    log(group_imp.sort_values(ascending=False).round(3).to_string())
    log("\nTop 15 features por permutación, medida en val con el modelo entrenado solo con train "
        "(así el test no interviene en la interpretación):")
    log(imp_df.head(15)[["feature", "group", "perm_importance_mean", "perm_importance_std"]]
        .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    log("(Con features muy correlacionadas, como GLCM a distintas distancias, la importancia por "
        "permutación se reparte entre ellas y cada una puede verse baja; interpreta mejor por grupo.)")
    save_importance_plot(imp_df, group_imp, out_dir / "feature_importance.png", "val")

    joblib.dump({"model": final_model, "features": feature_cols, "classes": class_names,
                 "params": params, "trained_on": "train+val"}, out_dir / "model_family_rf.joblib")
    return ["model_family_rf.joblib", "cv_results.csv", "cv_grid_results.csv", "test_predictions.csv",
            "feature_importance.csv", "feature_importance.png", "confusion_matrix.png", "metrics_report.txt"]


# ----------------------------------------------------------------------
# MODO UN SOLO CSV (exploratorio)
# ----------------------------------------------------------------------

def run_single_csv(args, csv_path: Path, out_dir: Path, log):
    df, feature_cols = load_data(csv_path, args.keep_low_quality, log)

    counts = df["family"].value_counts()
    log("\n=== DATASET ===")
    log(f"Imágenes usadas: {len(df)}  |  Features: {len(feature_cols)}")
    log("Imágenes por familia:\n" + counts.to_string())
    if len(counts) < 2:
        print("[ERROR] Se necesitan al menos 2 familias para clasificar.")
        sys.exit(1)
    if counts.max() / counts.min() > 1.5:
        log(f"[AVISO] Clases desbalanceadas ({counts.max()} vs {counts.min()}): se usa class_weight='balanced' "
            f"y se reportan balanced_accuracy / F1 macro, que no se inflan por la clase mayoritaria.")

    encoder = LabelEncoder()
    y = encoder.fit_transform(df["family"])
    class_names = encoder.classes_
    X = df[feature_cols]
    groups = df["species_group"].values
    n_classes = len(class_names)

    n_groups_per_class = df.groupby("family")["species_group"].nunique()
    log("Especies (grupos) por familia:\n" + n_groups_per_class.to_string())

    n_splits = min(args.cv_folds, int(counts.min()), int(n_groups_per_class.min()))
    if n_splits < 2:
        print("[ERROR] Muy pocos datos por clase/especie para validar de forma cruzada.")
        sys.exit(1)
    if n_splits < args.cv_folds:
        log(f"[AVISO] Se reducen los folds de {args.cv_folds} a {n_splits} por falta de datos en alguna clase.")

    model = make_model(args.n_estimators, args.seed)

    # Sin etiqueta de especie no se puede agrupar: todo pasa a ser aleatorio y se avisa.
    can_group = "label" in df.columns
    strict = "group" if can_group else "random"
    split_mode = args.split_mode if can_group else "random"

    log(f"\n=== VALIDACIÓN CRUZADA ({n_splits} folds) ===")
    cv_random = run_cv(model, X, y, groups, "random", n_splits, args.seed, n_classes)
    log(f"Aleatoria            : {fmt_cv(cv_random)}")
    if can_group:
        cv_group = run_cv(model, X, y, groups, "group", n_splits, args.seed, n_classes)
        log(f"Agrupada por especie : {fmt_cv(cv_group)}")
        gap = cv_random["balanced_accuracy"].mean() - cv_group["balanced_accuracy"].mean()
        if gap > 0.05:
            log(f"-> La exactitud balanceada cae {gap:.3f} al agrupar por especie: parte del rendimiento "
                f"'aleatorio' viene de reconocer especies ya vistas, no familias. Usa la cifra agrupada como la realista.")
        else:
            log(f"-> Diferencia de solo {gap:.3f} entre ambas: el modelo generaliza bien a especies no vistas.")
        pd.concat({"aleatoria": cv_random, "agrupada": cv_group}, names=["estrategia", "fold"]).to_csv(out_dir / "cv_results.csv")
    else:
        log("Agrupada por especie : NO DISPONIBLE (el CSV no tiene columna 'label'). Los resultados son "
            "optimistas: imágenes de una misma especie pueden estar en train y test a la vez.")
        cv_random.to_csv(out_dir / "cv_results.csv")

    if not args.skip_ablation:
        log(f"\n=== ABLACIÓN (validación {'agrupada por especie' if can_group else 'aleatoria'}) ===")
        subsets = {
            "Solo textura": [c for c in feature_cols if feature_group(c) in TEXTURE_GROUPS],
            "Solo color": [c for c in feature_cols if feature_group(c) in COLOR_GROUPS],
            "Solo forma": [c for c in feature_cols if feature_group(c) == "Forma"],   # vacío sin shape_*
            "Todas las features": feature_cols,
        }
        for name, cols in subsets.items():
            if not cols:
                log(f"{name:<16}: sin features disponibles")
                continue
            r = run_cv(model, df[cols], y, groups, strict, n_splits, args.seed, n_classes)
            log(f"{name:<16} ({len(cols):>3} features): {fmt_cv(r)}")

    train_idx, test_idx = holdout_split(X, y, groups, args.test_size, split_mode, args.seed)
    Xtr, Xte, ytr, yte = X.iloc[train_idx], X.iloc[test_idx], y[train_idx], y[test_idx]
    log(f"\n=== HOLD-OUT INTERNO ({split_mode}) ===")
    log("(Sale del mismo CSV; para la evaluación oficial usa el modo dividido con Split_dataset.py.)")
    log(f"Train: {len(train_idx)} imágenes  |  Test: {len(test_idx)} imágenes")
    if len(set(ytr)) < n_classes or len(set(yte)) < n_classes:
        log("[AVISO] Alguna familia no aparece en train o en test; los resultados del hold-out no son confiables.")

    holdout_model = make_model(args.n_estimators, args.seed).fit(Xtr, ytr)
    pred = holdout_model.predict(Xte)
    proba = holdout_model.predict_proba(Xte)
    log("\n" + classification_report(yte, pred, labels=list(range(n_classes)), target_names=class_names, digits=3, zero_division=0))
    sc = score_dict(yte, pred, proba, n_classes)
    if not np.isnan(sc["auc"]):
        log(f"AUC (hold-out): {sc['auc']:.3f}")
    cm = confusion_matrix(yte, pred, labels=list(range(n_classes)))
    log("Matriz de confusión (filas = real, columnas = predicho):")
    log(pd.DataFrame(cm, index=[f"real {c}" for c in class_names],
                     columns=[f"pred {c}" for c in class_names]).to_string())
    modo = "agrupado por especie" if split_mode == "group" else "aleatorio"
    save_confusion_matrix(yte, pred, class_names, out_dir / "confusion_matrix.png", f"Matriz de confusión, hold-out {modo}")

    log("\n=== IMPORTANCIA DE FEATURES ===")
    imp_df = importance_tables(holdout_model, Xte, yte, feature_cols, args.seed, log)
    imp_df.to_csv(out_dir / "feature_importance.csv", index=False)
    group_imp = imp_df.groupby("group")["impurity_importance"].sum()
    log("Importancia agregada por grupo (impureza, suma = 1):")
    log(group_imp.sort_values(ascending=False).round(3).to_string())
    log("\nTop 15 features por permutación (caída de balanced_accuracy en el hold-out):")
    log(imp_df.head(15)[["feature", "group", "perm_importance_mean", "perm_importance_std"]]
        .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    log("(Con features muy correlacionadas, como GLCM a distintas distancias, la importancia por "
        "permutación se reparte entre ellas y cada una puede verse baja; interpreta mejor por grupo.)")
    save_importance_plot(imp_df, group_imp, out_dir / "feature_importance.png", "hold-out")

    final_model = make_model(args.n_estimators, args.seed).fit(X, y)
    joblib.dump({"model": final_model, "features": feature_cols, "classes": class_names,
                 "params": {}, "trained_on": "todo el CSV"}, out_dir / "model_family_rf.joblib")
    return ["model_family_rf.joblib", "cv_results.csv", "feature_importance.csv",
            "feature_importance.png", "confusion_matrix.png", "metrics_report.txt"]


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split-dir", default=str(DEFAULT_SPLIT_DIR),
                        help="Carpeta con features_train/val/test.csv (salida de Split_dataset.py)")
    parser.add_argument("--csv", default=None,
                        help="Fuerza el modo un solo CSV con este archivo (features_dataset_ready.csv)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR), help="Carpeta de resultados")
    parser.add_argument("--n-estimators", type=int, default=N_ESTIMATORS, help="Árboles del modelo final")
    parser.add_argument("--tune-estimators", type=int, default=300, help="Árboles durante la búsqueda de hiperparámetros")
    parser.add_argument("--no-tune", action="store_true", help="Omitir la búsqueda de hiperparámetros")
    parser.add_argument("--n-boot", type=int, default=N_BOOT, help="Remuestreos del bootstrap para los intervalos de confianza")
    parser.add_argument("--cv-folds", type=int, default=CV_FOLDS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--keep-low-quality", action="store_true", help="Conservar las filas marcadas low_quality")
    parser.add_argument("--skip-ablation", action="store_true", help="Omitir la comparación textura / color / ambas")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE, help="(modo un solo CSV) fracción del hold-out interno")
    parser.add_argument("--split-mode", choices=["group", "random"], default="group",
                        help="(modo un solo CSV) hold-out agrupado por especie o aleatorio")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(msg=""):
        print(msg)
        lines.append(str(msg))

    split_dir = Path(args.split_dir)
    have_split = all((split_dir / f"features_{s}.csv").exists() for s in SPLITS)

    if args.csv:
        log(f"Modo UN SOLO CSV (pedido con --csv): {args.csv}")
        files = run_single_csv(args, Path(args.csv), out_dir, log)
    elif have_split:
        log(f"Modo DIVIDIDO: se leen features_train/val/test.csv de {split_dir}")
        files = run_split(args, split_dir, out_dir, log)
    else:
        log(f"[AVISO] No se encontró la división en {split_dir} (faltan features_train/val/test.csv). "
            f"Genera la división con Split_dataset.py; mientras tanto se usa el modo UN SOLO CSV.")
        files = run_single_csv(args, DEFAULT_CSV, out_dir, log)

    log("\n=== ARCHIVOS GENERADOS ===")
    for name in files:
        if name == "metrics_report.txt" or (out_dir / name).exists():
            log(f"  {out_dir / name}")
    (out_dir / "metrics_report.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()