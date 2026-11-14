#!/usr/bin/env python3
"""
evaluar_modelos.py
------------------
Evaluación unificada de los modelos del proyecto sobre LAS MISMAS 122 imágenes de test (24 especies que ningún modelo vio al
entrenar), para poder compararlos y analizar el proceso. Clase positiva: Pieridae; umbral de decisión 0,5.

Modelos:
  Clásico LogReg (rembg)               regresión logística sobre textura y color, con máscaras de rembg (resultados/final)
  Clásico LogReg (GrabCut, desplegado) el mismo método con máscaras de GrabCut: el que está en app/modelo y en el Space; sus
                                       características se recuperan del historial de git (commit d0645bb)
  Random Forest (rembg)                línea base de árboles (resultados/rf)
  ConvNeXt embeddings                  ConvNeXt-Tiny con pesos de ImageNet sin cambiar + regresión logística
  ConvNeXt afinado                     ConvNeXt-Tiny con todos los pesos ajustados (fine-tuning)

Por modelo: matriz de confusión; exactitud, balanced accuracy, precisión/recall/F1 por clase, F1 macro y ponderado, MCC, kappa
de Cohen, AUC, precisión promedio, Brier, log-loss y ECE; IC 95 % por bootstrap de ESPECIES (las fotos de una especie no son
independientes); comparaciones pareadas con IC y prueba de McNemar exacta; errores por imagen y por especie; cobertura de las
tres capas de rechazo; contexto de la validación cruzada; curvas del fine-tuning; y las fotos de internet de Pruebas/.

Salida en resultados/evaluacion_modelos/: resumen.md (empieza por aquí), tablas en CSV y gráficos en PNG.

Uso:
    python scripts/evaluacion/evaluar_modelos.py
    python scripts/evaluacion/evaluar_modelos.py --bootstrap 5000
"""
# Localiza rutas.py (en scripts/, o junto a este archivo en la app) y agrega al path la carpeta de cada proceso
import sys
from pathlib import Path
sys.path[:0] = [str(p) for p in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]) if (p / "rutas.py").exists()][:1]
import rutas  # noqa: E402,F401

import argparse
import io
import re
import subprocess
import warnings
from math import ceil

import cv2
import joblib
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import binomtest
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss, classification_report,
                             cohen_kappa_score, confusion_matrix, f1_score, log_loss, matthews_corrcoef,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.preprocessing import normalize

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from embeddings import indice, ruta_imagen
from inferencia import Clasificador
from inferencia_embeddings import ClasificadorEmbeddings
from probar_carpeta import cargar, familia_real
from rutas import APP_DIR, REPO_DIR, RESULTS_DIR, SEGMENTATION_DIR, SPLIT_DIR
from Train_random_forest import label_from_filename

OUT = RESULTS_DIR / "evaluacion_modelos"
CLASES = ["Papilionidae", "Pieridae"]
COMMIT_GRABCUT = "d0645bb"       # último commit con las características calculadas con GrabCut (modelo desplegado)
M_RB, M_GC, M_RF, M_EM, M_AF = ("Clásico LogReg (rembg)", "Clásico LogReg (GrabCut, desplegado)", "Random Forest (rembg)",
                                "ConvNeXt embeddings", "ConvNeXt afinado")
COLOR = {M_RB: "#7f7f7f", M_GC: "#bcbd22", M_RF: "#8c564b", M_EM: "#1f77b4", M_AF: "#d62728"}


def md(df: pd.DataFrame, dec: int = 3) -> str:
    """Tabla en Markdown (sin depender de tabulate)."""
    f = lambda v: "" if pd.isna(v) else (f"{v:.{dec}f}" if isinstance(v, (float, np.floating)) else str(v))
    filas = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "|".join("---" for _ in df.columns) + "|"]
    filas += ["| " + " | ".join(f(r[c]) for c in df.columns) + " |" for _, r in df.astype(object).iterrows()]   # object: iterrows no convierte los enteros a decimal
    return "\n".join(filas)


# ----------------------------------------------------------------------------------------------------------------------
# Datos de test y probabilidades de cada modelo
# ----------------------------------------------------------------------------------------------------------------------

def universo() -> pd.DataFrame:
    idx = indice()
    idx["pos"] = range(len(idx))     # posición en los .npy de embeddings
    idx = idx.merge(pd.read_csv(SPLIT_DIR / "splits.csv")[["family", "filename", "split"]], on=["family", "filename"], how="left")
    te = idx[idx.split == "test"].reset_index(drop=True)
    te["grupo"] = [" ".join(label_from_filename(f).lower().split("_")[:2]) for f in te.filename]
    te["y"] = (te.family == "Pieridae").astype(int)
    te["area"] = [float((cv2.imread(str(SEGMENTATION_DIR / r.dataset / "masks" / r.filename), cv2.IMREAD_GRAYSCALE) > 0).mean())
                  for r in te.itertuples()]
    return te


def probabilidades(te: pd.DataFrame, log):
    """Devuelve {modelo: p(Pieridae)} y los datos que necesita el análisis de rechazo."""
    P, rechazo = {}, {}
    p = pd.read_csv(RESULTS_DIR / "final" / "test_predictions.csv").set_index("filename")
    P[M_RB] = p.loc[te.filename, "proba_Pieridae"].values
    p = pd.read_csv(RESULTS_DIR / "rf" / "test_predictions.csv").set_index("filename")
    P[M_RF] = p.loc[te.filename, "proba_Pieridae"].values

    # modelo clásico desplegado: sus características están en el historial de git
    out = subprocess.run(["git", "show", f"{COMMIT_GRABCUT}:resultados/dataset_split/features_test.csv"], capture_output=True, text=True, cwd=REPO_DIR)
    if out.returncode == 0:
        b = joblib.load(APP_DIR / "modelo" / "final_model.joblib")
        X = pd.read_csv(io.StringIO(out.stdout)).set_index("filename").loc[te.filename, b["features"]]
        P[M_GC] = b["model"].predict_proba(X)[:, 1]
        ba = balanced_accuracy_score(te.y, P[M_GC] > .5)
        log(f"[comprobación] clásico desplegado: balanced accuracy {ba:.3f} (se reportó 0,868 al entrenarlo)")
    else:
        log("[AVISO] No se pudo leer el historial de git: se omite el clásico desplegado.")

    # embeddings sin afinar
    be = joblib.load(RESULTS_DIR / "modelo_embeddings" / "modelo.joblib")
    X = normalize(np.load(RESULTS_DIR / "embeddings" / "convnext__recorte.npy"))[te.pos.values]
    P[M_EM] = be["modelo"].predict_proba(X)[:, 1]
    Z = be["modelo"].named_steps["s"].transform(be["modelo"].named_steps["i"].transform(X))
    rechazo[M_EM] = dict(b=be, d2=be["covarianza"].mahalanobis(Z))

    # afinado: se pasa la red sobre los recortes de test
    eng = ClasificadorEmbeddings(RESULTS_DIR / "modelo_afinado" / "modelo_afinado.joblib"); emb = eng._embedder()
    feats, pa = [], []
    for r in te.itertuples():
        f, lg = emb(Image.open(ruta_imagen(r, "recorte")).convert("RGB")); feats.append(f[0]); e = np.exp(lg[0] - lg[0].max()); pa.append((e / e.sum())[1])
    P[M_AF] = np.array(pa)
    rechazo[M_AF] = dict(b=eng.b, d2=eng.b["covarianza"].mahalanobis(np.array(feats)))

    # clásico (rembg): distancia con su propio escalador y covarianza
    clf = Clasificador(RESULTS_DIR / "final" / "final_model.joblib", RESULTS_DIR / "rechazo" / "rechazo.joblib")
    ft = pd.read_csv(SPLIT_DIR / "features_test.csv").set_index("filename").loc[te.filename, clf.columnas]
    rechazo[M_RB] = dict(b=clf.rechazo, d2=clf.rechazo["covarianza"].mahalanobis(clf.estandarizar(ft)))
    orden = [m for m in (M_RB, M_GC, M_RF, M_EM, M_AF) if m in P]
    return {m: P[m] for m in orden}, rechazo, eng, clf


# ----------------------------------------------------------------------------------------------------------------------
# Métricas
# ----------------------------------------------------------------------------------------------------------------------

def ece(y, p, bins=10):
    conf, ok = np.maximum(p, 1 - p), ((p > .5).astype(int) == y)
    b = np.digitize(conf, np.linspace(.5, 1, bins + 1)[1:-1])
    return float(sum((b == k).mean() * abs(ok[b == k].mean() - conf[b == k].mean()) for k in range(bins) if (b == k).any()))


def metricas(y, p):
    pred = (p > .5).astype(int); tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    rep = classification_report(y, pred, labels=[0, 1], target_names=CLASES, output_dict=True, zero_division=0)
    return {"exactitud": (tp + tn) / len(y), "balanced_accuracy": balanced_accuracy_score(y, pred),
            "precision_Pieridae": rep["Pieridae"]["precision"], "sensibilidad_Pieridae": rep["Pieridae"]["recall"],
            "especificidad_Pieridae": rep["Papilionidae"]["recall"], "F1_Papilionidae": rep["Papilionidae"]["f1-score"],
            "F1_Pieridae": rep["Pieridae"]["f1-score"], "F1_macro": f1_score(y, pred, average="macro"),
            "F1_ponderado": f1_score(y, pred, average="weighted"), "MCC": matthews_corrcoef(y, pred),
            "kappa": cohen_kappa_score(y, pred), "AUC": roc_auc_score(y, p), "precision_promedio": average_precision_score(y, p),
            "Brier": brier_score_loss(y, p), "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)), "ECE": ece(y, p),
            "errores": int((pred != y).sum()), "TN": tn, "FP": fp, "FN": fn, "TP": tp}


CLAVES_IC = ["exactitud", "balanced_accuracy", "F1_macro", "MCC", "AUC"]


def rapidas(y, p, s):
    """Las métricas con IC, calculadas sobre la remuestra s sin pasar por sklearn (salvo el AUC)."""
    ys, ps = y[s], p[s]; pr = (ps > .5).astype(int)
    tp, tn = ((pr == 1) & (ys == 1)).sum(), ((pr == 0) & (ys == 0)).sum(); fp, fn = ((pr == 1) & (ys == 0)).sum(), ((pr == 0) & (ys == 1)).sum()
    r1, r0 = tp / max(tp + fn, 1), tn / max(tn + fp, 1); den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return np.array([(tp + tn) / len(ys), (r1 + r0) / 2, (2 * tp / max(2 * tp + fp + fn, 1) + 2 * tn / max(2 * tn + fn + fp, 1)) / 2,
                     (tp * tn - fp * fn) / den if den else 0.0, roc_auc_score(ys, ps)])


def remuestras(y, grupos, n, seed=0):
    rng = np.random.default_rng(seed); ix = {g: np.where(grupos == g)[0] for g in pd.unique(grupos)}; ks = list(ix); out = []
    while len(out) < n:
        s = np.concatenate([ix[ks[k]] for k in rng.integers(0, len(ks), len(ks))])
        if 0 < y[s].sum() < len(s):
            out.append(s)
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Gráficos
# ----------------------------------------------------------------------------------------------------------------------

def graf_confusion(y, P, out):
    n = len(P); cols = min(n, 3); filas = ceil(n / cols); fig, axs = plt.subplots(filas, cols, figsize=(4.3 * cols, 4.2 * filas), squeeze=False)
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    for ax, (m, p) in zip(axs.ravel(), P.items()):
        cm = confusion_matrix(y, (p > .5).astype(int), labels=[0, 1]); nm = cm / cm.sum(1, keepdims=True)
        ax.imshow(nm, vmin=0, vmax=1, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n({nm[i, j]:.0%})", ha="center", va="center", color="white" if nm[i, j] > .5 else "black", fontsize=13)
        ax.set_xticks([0, 1], CLASES, rotation=20); ax.set_yticks([0, 1], CLASES); ax.set_xlabel("predicha"); ax.set_ylabel("real")
        ax.set_title(f"{m}\nbalanced acc. {balanced_accuracy_score(y, (p > .5)):.3f}", fontsize=10)
    fig.suptitle("Matrices de confusión en test (122 imágenes; porcentaje por fila)", y=1.0); fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)


def graf_curvas(y, P, out):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5))
    for m, p in P.items():
        fpr, tpr, _ = roc_curve(y, p); a.plot(fpr, tpr, color=COLOR[m], label=f"{m} (AUC {roc_auc_score(y, p):.3f})")
        pr, rc, _ = precision_recall_curve(y, p); b.plot(rc, pr, color=COLOR[m], label=f"{m} (AP {average_precision_score(y, p):.3f})")
    a.plot([0, 1], [0, 1], "k:", lw=1); a.set(xlabel="tasa de falsos positivos", ylabel="tasa de verdaderos positivos", title="Curva ROC (positiva: Pieridae)")
    b.set(xlabel="recall", ylabel="precisión", title="Curva precisión-recall (positiva: Pieridae)", ylim=(0.5, 1.02))
    a.legend(fontsize=8, loc="lower right"); b.legend(fontsize=8, loc="lower left"); fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def graf_calibracion(y, P, out):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for m, p in P.items():
        fr, md_ = calibration_curve(y, p, n_bins=5, strategy="quantile"); ax.plot(md_, fr, "o-", color=COLOR[m], label=f"{m} (ECE {ece(y, p):.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1); ax.set(xlabel="probabilidad predicha de Pieridae", ylabel="fracción real de Pieridae", title="Calibración (5 cuantiles; pocas imágenes por punto)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def graf_barras(tabla, out):
    metr = ["balanced_accuracy", "F1_macro", "MCC", "AUC"]; fig, axs = plt.subplots(1, 4, figsize=(16, 4.2), sharey=True)
    for ax, k in zip(axs, metr):
        y = np.arange(len(tabla)); v = tabla[k].values; lo, hi = tabla[k + "_ic_inf"].values, tabla[k + "_ic_sup"].values
        izq = min(0.6, lo.min() - .02)
        ax.barh(y, v, color=[COLOR[m] for m in tabla.index], xerr=[v - lo, hi - v], capsize=3); ax.set_title(k); ax.set_xlim(izq, 1.02)
        for i, x in enumerate(v):   # el valor va a la izquierda de la barra, lejos de las líneas del intervalo
            ax.text(izq + .008, i, f"{x:.3f}", ha="left", va="center", color="white", fontsize=9, fontweight="bold")
    axs[0].set_yticks(np.arange(len(tabla)), tabla.index, fontsize=8); axs[0].invert_yaxis()
    fig.suptitle("Métricas en test con IC 95 % (bootstrap por especies)", y=1.02); fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)


def graf_afinado(hist, out):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for s, g in hist.groupby("semilla"):
        axs[0].plot(g.epoca, g.perdida_train, label=f"semilla {s}"); axs[1].plot(g.epoca, g.val_perdida, label=f"semilla {s}"); axs[2].plot(g.epoca, g.val_bal_acc, label=f"semilla {s}")
    axs[0].set(title="Pérdida de entrenamiento", xlabel="época"); axs[1].set(title="Pérdida de validación", xlabel="época"); axs[2].set(title="Balanced accuracy de validación (saturada)", xlabel="época", ylim=(0.9, 1.005))
    axs[0].legend(); fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def graf_cv(cv, out):
    d = cv.sort_values("bal_acc_cv"); fig, ax = plt.subplots(figsize=(9, 0.42 * len(d) + 1.5))
    ax.barh(d.modelo, d.bal_acc_cv, xerr=d.desv.fillna(0), color=["#1f77b4" if t == "red preentrenada" else "#7f7f7f" for t in d.tipo], capsize=3)
    ax.set(xlim=(0.7, 1.0), xlabel="balanced accuracy (CV agrupada por especie, train + val)", title="Validación cruzada: modelos clásicos (gris) y embeddings (azul)")
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


# ----------------------------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bootstrap", type=int, default=2000, help="Remuestras de especies para los IC")
    ap.add_argument("--output", default=str(OUT))
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    avisos = []
    log = lambda m: (print(m, flush=True), avisos.append(m))

    te = universo(); y, grupos = te.y.values, te.grupo.values
    log(f"Test: {len(te)} imágenes ({int((y == 0).sum())} Papilionidae, {int((y == 1).sum())} Pieridae), {len(set(grupos))} especies.")
    P, rechazo, eng, clf = probabilidades(te, log)
    for m, p in P.items():
        assert len(p) == len(te) and np.isfinite(p).all(), m

    # --- 1) métricas con IC ---
    pts = {m: metricas(y, p) for m, p in P.items()}
    muestras = remuestras(y, grupos, args.bootstrap)
    boot = {m: np.array([rapidas(y, p, s) for s in muestras]) for m, p in P.items()}
    filas = []
    for m in P:
        fila = {"modelo": m, **pts[m]}
        for i, k in enumerate(CLAVES_IC):
            fila[k + "_ic_inf"], fila[k + "_ic_sup"] = np.percentile(boot[m][:, i], [2.5, 97.5])
        filas.append(fila)
    tabla = pd.DataFrame(filas).set_index("modelo")
    tabla.round(4).to_csv(out / "metricas_test.csv")
    pd.DataFrame([{"modelo": m, "clase_real": CLASES[i], "predicha_Papilionidae": confusion_matrix(y, (p > .5), labels=[0, 1])[i, 0],
                   "predicha_Pieridae": confusion_matrix(y, (p > .5), labels=[0, 1])[i, 1]} for m, p in P.items() for i in (0, 1)]).to_csv(out / "matrices_confusion.csv", index=False)
    por_clase = []
    for m, p in P.items():
        rep = classification_report(y, (p > .5).astype(int), labels=[0, 1], target_names=CLASES, output_dict=True, zero_division=0)
        por_clase += [{"modelo": m, "clase": c, "precision": rep[c]["precision"], "recall": rep[c]["recall"], "F1": rep[c]["f1-score"], "soporte": int(rep[c]["support"])} for c in CLASES]
    pd.DataFrame(por_clase).round(4).to_csv(out / "por_clase.csv", index=False)

    # --- 2) comparaciones pareadas ---
    pares = [(M_AF, M_EM), (M_AF, M_RB), (M_AF, M_GC), (M_EM, M_RB), (M_EM, M_GC), (M_RB, M_GC), (M_RB, M_RF)]
    pareadas = []
    for a, b in pares:
        if a not in P or b not in P:
            continue
        ra, rb = (P[a] > .5) == y, (P[b] > .5) == y; n01, n10 = int((~ra & rb).sum()), int((ra & ~rb).sum())     # a falla y b acierta / al revés
        fila = {"A": a, "B": b, "solo_A_acierta": n10, "solo_B_acierta": n01,
                "McNemar_p": binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0}
        for k in ("balanced_accuracy", "F1_macro", "MCC"):
            i = CLAVES_IC.index(k); d = boot[a][:, i] - boot[b][:, i]
            fila[f"dif_{k}"] = pts[a][k] - pts[b][k]; fila[f"dif_{k}_ic_inf"], fila[f"dif_{k}_ic_sup"] = np.percentile(d, [2.5, 97.5])
        pareadas.append(fila)
    pareadas = pd.DataFrame(pareadas); pareadas.round(4).to_csv(out / "comparaciones_pareadas.csv", index=False)

    # --- 3) errores por imagen y por especie ---
    det = te[["filename", "family", "grupo"]].copy()
    for m, p in P.items():
        det[f"{m}: P(Pieridae)"] = p; det[f"{m}: error"] = ((p > .5).astype(int) != y)
    det["modelos_que_fallan"] = det[[f"{m}: error" for m in P]].sum(axis=1)
    det.sort_values(["modelos_que_fallan", "grupo"], ascending=[False, True]).round(3).to_csv(out / "errores_por_imagen.csv", index=False)
    esp = te.groupby("grupo").agg(familia=("family", "first"), imagenes=("filename", "size")).join(
        pd.DataFrame({m: det.groupby("grupo")[f"{m}: error"].sum() for m in P}))
    esp = esp.sort_values([M_AF, M_EM], ascending=False); esp.to_csv(out / "errores_por_especie.csv")

    # --- 4) rechazo: cobertura y exactitud de lo aceptado ---
    cob = []
    for m, r in rechazo.items():
        b, d2, p = r["b"], r["d2"], P[m]; amin, amax = b["area_min"], b["area_max"]
        c1 = (te.area.values < amin) | (te.area.values > amax); c2 = ~c1 & (d2 > b["umbral_distancia"]); c3 = ~c1 & ~c2 & (np.maximum(p, 1 - p) < b["umbral_confianza"]); ok = ~c1 & ~c2 & ~c3
        pred = (p > .5).astype(int)
        cob.append({"modelo": m, "rechazadas_area": int(c1.sum()), "rechazadas_distancia": int(c2.sum()), "inciertas": int(c3.sum()), "aceptadas": int(ok.sum()),
                    "pct_aceptadas": ok.mean(), "exactitud_sin_filtro": (pred == y).mean(), "exactitud_aceptadas": (pred[ok] == y[ok]).mean(),
                    "errores_que_se_filtran": int(((pred != y) & ~ok).sum()), "errores_aceptados": int(((pred != y) & ok).sum())})
    cob = pd.DataFrame(cob); cob.round(4).to_csv(out / "cobertura_rechazo.csv", index=False)

    # --- 5) contexto de la validación cruzada ---
    cvm = pd.read_csv(RESULTS_DIR / "modelos" / "comparacion_modelos_cv.csv").rename(columns={"modelo": "modelo", "balanced_accuracy": "bal_acc_cv", "bal_acc_std": "desv", "auc": "auc_cv"})
    cvm = cvm.assign(tipo="clásico (textura y color)")[["modelo", "tipo", "bal_acc_cv", "desv", "auc_cv"]]
    cve = pd.read_csv(RESULTS_DIR / "embeddings" / "evaluacion_cv.csv").rename(columns={"conjunto": "modelo", "balanced_acc": "bal_acc_cv", "desv_repeticiones": "desv", "auc": "auc_cv"})
    cve = cve[~cve.modelo.str.startswith("CLÁSICO")].assign(tipo="red preentrenada")[["modelo", "tipo", "bal_acc_cv", "desv", "auc_cv"]]
    cv = pd.concat([cvm, cve], ignore_index=True); cv.round(4).to_csv(out / "cv_comparacion.csv", index=False)

    # --- 6) fine-tuning ---
    hist = pd.read_csv(RESULTS_DIR / "modelo_afinado" / "historial.csv")
    sem = []
    for linea in (RESULTS_DIR / "modelo_afinado" / "resumen.txt").read_text(encoding="utf-8").splitlines():
        v = linea.split()
        if len(v) == 9 and re.fullmatch(r"\d+", v[0]) and all(re.fullmatch(r"[\d.]+", x) for x in v):
            sem.append(dict(zip(["semilla", "val_bal_acc", "val_auc", "val_perdida", "val_errores", "test_bal_acc", "test_auc", "test_perdida", "test_errores"], map(float, v))))
    sem = pd.DataFrame(sem)
    for c in ("semilla", "val_errores", "test_errores"):
        sem[c] = sem[c].astype(int)
    sem.to_csv(out / "afinado_semillas.csv", index=False)

    # --- 7) fotos de internet ---
    web = pd.DataFrame()
    carpeta = REPO_DIR / "Pruebas"
    if carpeta.exists():
        cc = Clasificador(APP_DIR / "modelo" / "final_model.joblib", APP_DIR / "modelo" / "rechazo.joblib"); filas = []
        for f in sorted(p for p in carpeta.iterdir() if p.is_file()):
            img, real = cargar(f), familia_real(f.name)
            fila = {"foto": f.name, "real": real}
            for nom, motor in (("clásico (desplegado)", cc), ("afinado", eng)):
                r = motor.predecir(img); pr = (r.probabilidades or {}).get("Pieridae")
                fila[f"{nom}: estado"] = r.estado; fila[f"{nom}: familia"] = r.familia or "-"; fila[f"{nom}: P(Pieridae)"] = pr
                fila[f"{nom}: acierta"] = bool(r.estado == "aceptada" and r.familia == real)
            filas.append(fila)
        web = pd.DataFrame(filas); web.round(3).to_csv(out / "fotos_internet.csv", index=False)

    # --- gráficos ---
    graf_confusion(y, P, out / "matrices_confusion.png"); graf_curvas(y, P, out / "curvas_roc_pr.png"); graf_calibracion(y, P, out / "calibracion.png")
    graf_barras(tabla, out / "metricas_con_ic.png"); graf_afinado(hist, out / "afinado_curvas.png"); graf_cv(cv, out / "cv_comparacion.png")

    # --- resumen.md ---
    principal = tabla.reset_index()[["modelo", "exactitud", "balanced_accuracy", "F1_macro", "MCC", "AUC", "errores"]].copy()
    ic = lambda k: [f"{tabla.loc[m, k]:.3f} [{tabla.loc[m, k + '_ic_inf']:.3f}, {tabla.loc[m, k + '_ic_sup']:.3f}]" for m in tabla.index]
    principal_ic = pd.DataFrame({"modelo": tabla.index, "exactitud": tabla.exactitud.round(3).values, "balanced accuracy [IC 95 %]": ic("balanced_accuracy"),
                                 "F1 macro [IC 95 %]": ic("F1_macro"), "MCC [IC 95 %]": ic("MCC"), "AUC [IC 95 %]": ic("AUC"), "errores": tabla.errores.values})
    cm_md = pd.DataFrame([{"modelo": m, "TN (Papil. bien)": tabla.loc[m, "TN"], "FP (Papil. → Pier.)": tabla.loc[m, "FP"], "FN (Pier. → Papil.)": tabla.loc[m, "FN"], "TP (Pier. bien)": tabla.loc[m, "TP"]} for m in tabla.index])
    clase_md = pd.DataFrame(por_clase)
    prob_md = tabla.reset_index()[["modelo", "Brier", "log_loss", "ECE", "kappa", "precision_promedio"]]
    par_md = pareadas.assign(**{"dif. balanced acc. [IC 95 %]": [f"{r.dif_balanced_accuracy:+.3f} [{r.dif_balanced_accuracy_ic_inf:+.3f}, {r.dif_balanced_accuracy_ic_sup:+.3f}]" for r in pareadas.itertuples()],
                                "dif. F1 macro [IC 95 %]": [f"{r.dif_F1_macro:+.3f} [{r.dif_F1_macro_ic_inf:+.3f}, {r.dif_F1_macro_ic_sup:+.3f}]" for r in pareadas.itertuples()]})[
        ["A", "B", "dif. balanced acc. [IC 95 %]", "dif. F1 macro [IC 95 %]", "solo_A_acierta", "solo_B_acierta", "McNemar_p"]]
    def veredicto(a, b):
        f = pareadas[(pareadas.A == a) & (pareadas.B == b)]
        if f.empty:
            return None
        r = f.iloc[0]; lo, hi = r.dif_balanced_accuracy_ic_inf, r.dif_balanced_accuracy_ic_sup
        firme = "firme a favor de " + (a if lo > 0 else b) if (lo > 0 or hi < 0) else "no concluyente (el IC incluye el cero)"
        return f"**{a}** frente a **{b}**: {r.dif_balanced_accuracy:+.3f} de balanced accuracy [{lo:+.3f}, {hi:+.3f}], McNemar p = {r.McNemar_p:.3f} → {firme}."
    conclusiones = [v for v in (veredicto(M_AF, M_EM), veredicto(M_AF, M_GC), veredicto(M_EM, M_GC), veredicto(M_RB, M_GC), veredicto(M_RB, M_RF)) if v]
    fp_redes = {m: int(tabla.loc[m, "FP"]) for m in (M_EM, M_AF) if m in tabla.index}
    todos_fallan = det[det.modelos_que_fallan == len(P)][["filename", "grupo", "family"]]
    fallos_af = det[det[f"{M_AF}: error"]][["filename", "grupo", "family", f"{M_AF}: P(Pieridae)"]]
    ctx = cv.merge(pd.DataFrame({"modelo": ["Regresión logística", "Random Forest (referencia)", "convnext / recorte"], "bal_acc_test": [tabla.loc[M_RB, "balanced_accuracy"], tabla.loc[M_RF, "balanced_accuracy"], tabla.loc[M_EM, "balanced_accuracy"]]}), on="modelo")
    ctx["gap_cv_menos_test"] = ctx.bal_acc_cv - ctx.bal_acc_test
    af_ic = tabla.loc[M_AF]
    texto = f"""# Evaluación de los modelos del proyecto

Todos los modelos se evalúan sobre **las mismas {len(te)} imágenes de test** ({int((y == 0).sum())} Papilionidae y {int((y == 1).sum())} Pieridae) de **{len(set(grupos))} especies que ningún modelo vio al entrenar**. Clase positiva: Pieridae; umbral de decisión 0,5. Los intervalos de confianza (IC 95 %) salen de un bootstrap de **especies** ({args.bootstrap} remuestras): las fotos de una misma especie no son independientes, así que remuestrear imágenes sueltas daría intervalos demasiado estrechos.

## 1. Resultado principal

{md(principal_ic)}

### Conclusiones (se calculan con los números de arriba)

{chr(10).join("- " + c for c in conclusiones)}
- **Falsos positivos de las redes** (Papilionidae clasificadas como Pieridae): {", ".join(f"{m} {n}" for m, n in fp_redes.items())}. Los errores de las redes son Pieridae que se confunden con Papilionidae.
- Los modelos clásicos pierden sobre todo en los **mimetismos**: Pieridae oscuras con manchas (*Archonias*, *Perrhybris*, *Dismorphia*) y Papilionidae pálidas (*Protesilaus*).

- **Mejor modelo en test:** {tabla.balanced_accuracy.idxmax()} ({tabla.balanced_accuracy.max():.3f} de balanced accuracy, IC [{tabla.loc[tabla.balanced_accuracy.idxmax(), 'balanced_accuracy_ic_inf']:.3f}, {tabla.loc[tabla.balanced_accuracy.idxmax(), 'balanced_accuracy_ic_sup']:.3f}]).
- **Los intervalos son anchos** (con 24 especies de test): varios modelos se solapan, y las comparaciones de la sección 4 dicen cuáles diferencias se sostienen.
- Gráfico: `metricas_con_ic.png`. Tabla completa (con todas las métricas): `metricas_test.csv`.

## 2. Matrices de confusión

Filas: clase real; columnas: clase predicha. Gráfico: `matrices_confusion.png`.

{md(cm_md, 0)}

Lectura: **FP** son Papilionidae clasificadas como Pieridae; **FN**, Pieridae clasificadas como Papilionidae.

## 3. Precisión, recall y F1 por clase

{md(clase_md)}

F1 macro (promedio de las dos clases) y F1 ponderado (por soporte) están en `metricas_test.csv`. La balanced accuracy es el recall medio de las dos clases, y por eso no la infla la clase mayoritaria (Pieridae es el {y.mean():.0%} del test).

## 4. Comparaciones entre modelos (pareadas, mismas imágenes)

Diferencia **A − B**, con IC de bootstrap por especies. `solo_A_acierta` / `solo_B_acierta` cuentan las imágenes en que un modelo acierta y el otro falla; `McNemar_p` es la prueba exacta de McNemar sobre esos desacuerdos. Una diferencia solo es firme si su IC no incluye el cero.

{md(par_md)}

## 5. Probabilidades y calibración

Brier y log-loss (más bajos, mejor), ECE (error de calibración, más bajo mejor), kappa de Cohen y precisión promedio. Gráficos: `curvas_roc_pr.png` y `calibracion.png`.

El mejor Brier es el de **{tabla.Brier.idxmin()}** y el menor ECE, el de **{tabla.ECE.idxmin()}**. Las tres métricas no siempre coinciden: un modelo puede acertar más y tener probabilidades más extremas (peor calibradas). Con {len(te)} imágenes, el ECE es ruidoso.

{md(prob_md)}

## 6. Errores

- **Imágenes que fallan los {len(P)} modelos:** {len(todos_fallan)} → {', '.join(todos_fallan.filename.str.replace('.png', '', regex=False)) if len(todos_fallan) else 'ninguna'}.
- **Errores del modelo afinado ({len(fallos_af)}):** {', '.join(f"{r.filename.replace('.png', '')} (real {r.family}, P(Pieridae)={getattr(r, '_4'):.2f})" for r in fallos_af.itertuples()) if len(fallos_af) else 'ninguno'}.
- Detalle por imagen: `errores_por_imagen.csv`; por especie: `errores_por_especie.csv`.

Errores por especie (solo las que fallan en algún modelo):

{md(esp[(esp[list(P)].sum(axis=1) > 0)].reset_index().rename(columns={"grupo": "especie"}), 0)}

## 7. Rechazo de imágenes (las tres capas) sobre el test

Fracción de imágenes legítimas que el sistema acepta, rechaza o marca como inciertas, y exactitud de lo aceptado. Un buen rechazo filtra errores sin descartar demasiadas fotos buenas.

{md(cob)}

No se incluye el clásico desplegado: su rechazo se calibró con máscaras de GrabCut, que ya no están en disco. El de rembg (arriba) usa los umbrales de `resultados/rechazo/`.

## 8. Validación cruzada frente al test

La CV (agrupada por especie, sobre train + val) y el test dan una idea de cuánto cambia el rendimiento con datos nuevos. Gráfico: `cv_comparacion.png`; tabla completa: `cv_comparacion.csv`.

{md(ctx)}

El modelo afinado no tiene CV (reentrenar una red 5 veces era demasiado costoso); se evaluó con una partición fija y varias semillas (sección 9).

## 9. Fine-tuning: semillas y curvas

{md(sem)}

Con un número fijo de épocas y sin parada temprana, las tres semillas dan resultados casi iguales: el afinado **varía poco entre semillas**. Se despliega la de mejor validación. La **validación está saturada** (1 error de 122 en las tres semillas), así que no sirve para distinguir modelos entre sí; el test es lo único que discrimina, y es pequeño. Curvas: `afinado_curvas.png`.

## 10. Fotos de internet (8 fotos fuera del catálogo)

Modelos tal como están en la app: el clásico desplegado (GrabCut) y el afinado. "acierta" = aceptada y con la familia correcta.

{md(web) if len(web) else '(no se encontró la carpeta Pruebas/)'}

Con 8 fotos no se puede estimar una exactitud con precisión; sirve para ver si hay un desajuste evidente con fotos no tomadas para el catálogo.

## 11. Cautelas para interpretar los números

- **El test es pequeño:** {len(set(grupos))} especies y {len(te)} imágenes. Una diferencia de un par de errores puede ser ruido; por eso se dan los IC y las pruebas pareadas.
- **El test ya se ha consultado varias veces** a lo largo del proyecto: al comparar modelos y al decidir qué ofrecer en la app (el clásico, el de embeddings y el afinado se vieron en test antes de decidir). Las cifras son, por tanto, algo optimistas.
- **ConvNeXt se preentrenó con ImageNet**, que incluye clases de mariposas: la comparación con el modelo clásico es justa en el protocolo, pero no en los datos de preentrenamiento.
- **Las dos familias vienen de catálogos distintos**; por eso los modelos de red usan el recorte segmentado y no la foto completa (riesgo de aprender el fondo o la iluminación del catálogo).
- **El afinado se entrenó solo con train** (568 imágenes), a diferencia de los demás, que se reentrenaron con train + val (690).
- Las fotos de internet son solo 8 y todas se aceptan o fallan por razones distintas: no sustituyen una evaluación con fotos de campo.

## 12. Índice de archivos

| Archivo | Contenido |
|---|---|
| `metricas_test.csv` | todas las métricas por modelo, con IC |
| `matrices_confusion.csv` / `.png` | matrices de confusión |
| `por_clase.csv` | precisión, recall y F1 por clase |
| `comparaciones_pareadas.csv` | diferencias entre modelos con IC y McNemar |
| `errores_por_imagen.csv`, `errores_por_especie.csv` | análisis de errores |
| `cobertura_rechazo.csv` | aceptadas, rechazadas e inciertas |
| `cv_comparacion.csv` / `.png` | validación cruzada |
| `afinado_semillas.csv`, `afinado_curvas.png` | fine-tuning |
| `fotos_internet.csv` | fotos de Pruebas/ |
| `curvas_roc_pr.png`, `calibracion.png`, `metricas_con_ic.png` | gráficos |

Generado con `python scripts/evaluacion/evaluar_modelos.py`.
"""
    (out / "resumen.md").write_text(texto, encoding="utf-8")
    log(f"\nEvaluación guardada en: {out.resolve()}")


if __name__ == "__main__":
    main()
