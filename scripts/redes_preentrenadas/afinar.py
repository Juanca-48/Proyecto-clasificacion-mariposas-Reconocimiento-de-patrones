#!/usr/bin/env python3
"""
afinar.py
---------
Fine-tuning completo de ConvNeXt-Tiny (preentrenada con ImageNet) para distinguir Papilionidae de Pieridae sobre los
recortes segmentados. A diferencia de embeddings.py (que no cambia los pesos), aquí se ajustan TODOS los pesos con tus
datos, partiendo de los de ImageNet.

Protocolo (el mismo reparto por especie que el resto del proyecto, resultados/dataset_split/splits.csv):
  - Se entrena con la partición train. Número fijo de épocas con decaimiento coseno: no hay parada temprana, porque
    elegir la mejor época sobre las 25 especies de validación solo añadiría ruido.
  - Se repite con varias semillas para medir cuánto varía. Se despliega la semilla con mejor validación (nunca la de
    mejor test).
  - La partición val sirve para elegir la semilla y para calibrar el rechazo (son especies que la red no vio durante
    el entrenamiento; con una red afinada no se puede calibrar con CV como en entrenar_embeddings.py).
  - El test se informa una sola vez, para todas las semillas, sin intervenir en ninguna decisión.
  - A diferencia de los otros modelos, el modelo final NO se reentrena con train + val.

Detalles: tasa de aprendizaje por capa (la cabeza nueva aprende más rápido que las capas preentrenadas, y las primeras
etapas menos), AdamW con decaimiento de pesos, profundidad estocástica 0,1, suavizado de etiquetas 0,1, pesos por clase
y aumentos moderados (giros, recortes, brillo y contraste; NO se altera el matiz, porque el color informa).

Salida en resultados/modelo_afinado/: modelo_afinado.joblib (pesos en media precisión, covarianza y umbrales del
rechazo; lo usa inferencia_embeddings.py), historial.csv (por época y semilla) y resumen.txt.

Uso:
    python scripts/redes_preentrenadas/afinar.py                              (3 semillas x 10 épocas, toda la red)
    python scripts/redes_preentrenadas/afinar.py --semillas 0 --epocas 5
    python scripts/redes_preentrenadas/afinar.py --congelar-hasta 3           (solo la última etapa y la cabeza)
    python scripts/redes_preentrenadas/afinar.py --limite 48 --epocas 1 --output <carpeta>   (prueba rápida)
"""
# Localiza rutas.py (en scripts/, o junto a este archivo en la app) y agrega al path la carpeta de cada proceso
import sys
from pathlib import Path
sys.path[:0] = [str(p) for p in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]) if (p / "rutas.py").exists()][:1]
import rutas  # noqa: E402,F401

import argparse
import time
import warnings

import cv2
import joblib
import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from sklearn.covariance import LedoitWolf
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torchvision import transforms as T

from embeddings import indice, ruta_imagen
from rutas import RESULTS_DIR, SEGMENTATION_DIR, SPLIT_DIR
from Train_random_forest import label_from_filename

ARQUITECTURA = "convnext_tiny.fb_in22k_ft_in1k"
CLASES = ["Papilionidae", "Pieridae"]


def cargar_datos(limite):
    idx = indice().merge(pd.read_csv(SPLIT_DIR / "splits.csv")[["family", "filename", "split"]], on=["family", "filename"], how="left")
    if idx.split.isna().any():
        sys.exit("[ERROR] Hay imágenes sin partición. Ejecuta entrenar.py (etapa split).")
    idx["y"] = (idx.family == "Pieridae").astype(int)
    idx["ruta"] = [str(ruta_imagen(f, "recorte")) for f in idx.itertuples()]
    idx["grupo"] = [" ".join(label_from_filename(f).lower().split("_")[:2]) for f in idx.filename]
    if limite:   # prueba rápida: pocas imágenes por partición
        idx = pd.concat([g.sample(min(len(g), limite), random_state=0) for _, g in idx.groupby("split")]).reset_index(drop=True)
    return idx


def crear_modelo(args, congelar_hasta):
    m = timm.create_model(ARQUITECTURA, pretrained=True, num_classes=2, drop_path_rate=0.1)
    if congelar_hasta:
        for p in m.stem.parameters():
            p.requires_grad = False
        for i, etapa in enumerate(m.stages):
            if i < congelar_hasta:
                for p in etapa.parameters():
                    p.requires_grad = False
    return m


def grupos_de_parametros(m, lr_base, decaimiento, lr_cabeza, wd):
    """Una tasa de aprendizaje por etapa (la última, la mayor entre las preentrenadas) y otra, mayor, para la cabeza nueva."""
    def lr_de(nombre):
        if nombre.startswith("head.fc"):
            return lr_cabeza
        if nombre.startswith("head"):
            return lr_base
        if nombre.startswith("stages."):
            return lr_base * decaimiento ** (3 - int(nombre.split(".")[1]))
        return lr_base * decaimiento ** 3          # stem
    grupos = {}
    for n, p in m.named_parameters():
        if p.requires_grad:
            clave = (lr_de(n), p.ndim >= 2)         # sin decaimiento de pesos en sesgos, normalizaciones ni layer scale
            grupos.setdefault(clave, []).append(p)
    return [{"params": ps, "lr": lr, "lr_max": lr, "weight_decay": wd if con_wd else 0.0} for (lr, con_wd), ps in grupos.items()]


@torch.no_grad()
def inferir(m, rutas, tf, lote=32):
    """Embeddings (antes de la capa final) y logits, sin aumentos."""
    m.eval(); feats, logits = [], []
    for i in range(0, len(rutas), lote):
        x = torch.stack([tf(Image.open(r).convert("RGB")) for r in rutas[i:i + lote]])
        f = m.forward_head(m.forward_features(x), pre_logits=True)
        feats.append(f); logits.append(m.head.fc(f))
    return torch.cat(feats).numpy(), torch.cat(logits).numpy()


def metricas(logits, y):
    p = torch.softmax(torch.tensor(logits), 1).numpy()[:, 1]
    perdida = float(torch.nn.functional.cross_entropy(torch.tensor(logits), torch.tensor(y)))
    return {"bal_acc": float(balanced_accuracy_score(y, p > .5)), "auc": float(roc_auc_score(y, p)), "perdida": perdida,
            "errores": int(((p > .5) != y).sum())}


def entrenar(args, semilla, tr, va, tf_eval, tf_train, log):
    torch.manual_seed(semilla); np.random.seed(semilla)
    m = crear_modelo(args, args.congelar_hasta)
    opt = torch.optim.AdamW(grupos_de_parametros(m, args.lr, args.decaimiento_capas, args.lr_cabeza, args.wd))
    pesos = torch.tensor(len(tr) / (2 * np.bincount(tr.y.values)), dtype=torch.float32)
    perdida = torch.nn.CrossEntropyLoss(weight=pesos, label_smoothing=0.1)
    por_epoca = int(np.ceil(len(tr) / args.lote)); total = por_epoca * args.epocas; paso = 0; hist = []
    gen = torch.Generator().manual_seed(semilla)
    for ep in range(1, args.epocas + 1):
        m.train(); t0 = time.time(); acum = 0.0
        orden = torch.randperm(len(tr), generator=gen).numpy()
        for i in range(0, len(tr), args.lote):
            lote = tr.iloc[orden[i:i + args.lote]]
            f = paso / por_epoca / args.calentamiento if paso < por_epoca * args.calentamiento else \
                0.5 * (1 + np.cos(np.pi * (paso - por_epoca * args.calentamiento) / max(1, total - por_epoca * args.calentamiento)))
            for g in opt.param_groups:
                g["lr"] = g["lr_max"] * f
            x = torch.stack([tf_train(Image.open(r).convert("RGB")) for r in lote.ruta])
            opt.zero_grad(); l = perdida(m(x), torch.tensor(lote.y.values)); l.backward(); opt.step()
            acum += float(l) * len(lote); paso += 1
        _, lg = inferir(m, va.ruta.tolist(), tf_eval); mv = metricas(lg, va.y.values)
        hist.append({"semilla": semilla, "epoca": ep, "perdida_train": acum / len(tr), **{f"val_{k}": v for k, v in mv.items()}})
        log(f"  semilla {semilla} época {ep:>2}/{args.epocas}: pérdida train {acum / len(tr):.3f} | val: bal_acc {mv['bal_acc']:.3f}, "
            f"auc {mv['auc']:.3f}, pérdida {mv['perdida']:.3f} ({time.time() - t0:.0f} s)")
    return m, hist


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--semillas", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epocas", type=int, default=10)
    ap.add_argument("--lote", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4, help="Tasa de aprendizaje de la última etapa preentrenada")
    ap.add_argument("--lr-cabeza", type=float, default=1e-3, help="Tasa de aprendizaje de la capa final (nueva)")
    ap.add_argument("--decaimiento-capas", type=float, default=0.5, help="Factor por etapa hacia las primeras capas")
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--calentamiento", type=float, default=1.0, help="Épocas de calentamiento lineal")
    ap.add_argument("--congelar-hasta", type=int, default=0, choices=[0, 1, 2, 3], help="0 = toda la red; 3 = solo la última etapa")
    ap.add_argument("--percentil", type=float, default=95, help="Percentil de la validación que fija el umbral de distancia")
    ap.add_argument("--umbral-confianza", type=float, default=0.65)
    ap.add_argument("--segmentador", choices=["grabcut", "rembg"], default="rembg")
    ap.add_argument("--limite", type=int, default=None, help="Imágenes por partición (prueba rápida)")
    ap.add_argument("--output", default=str(RESULTS_DIR / "modelo_afinado"))
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    lineas: list = []

    def log(m=""):
        print(m, flush=True); lineas.append(str(m))

    idx = cargar_datos(args.limite)
    tr, va, te = (idx[idx.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    log(f"ConvNeXt-Tiny, {'toda la red' if not args.congelar_hasta else f'congelando hasta la etapa {args.congelar_hasta}'}: "
        f"train {len(tr)}, val {len(va)}, test {len(te)} imágenes; {args.epocas} épocas x {len(args.semillas)} semillas.")
    ref = timm.create_model(ARQUITECTURA, pretrained=True, num_classes=2)
    cfg = timm.data.resolve_data_config({}, model=ref)
    tf_eval = timm.data.create_transform(**cfg)
    tf_train = T.Compose([T.RandomRotation(15, interpolation=T.InterpolationMode.BILINEAR, fill=(255, 255, 255)),
                          T.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.8, 1.25), interpolation=T.InterpolationMode.BICUBIC),
                          T.RandomHorizontalFlip(), T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                          T.ToTensor(), T.Normalize(cfg["mean"], cfg["std"])])

    historial, filas, mejor = [], [], None
    for s in args.semillas:
        m, hist = entrenar(args, s, tr, va, tf_eval, tf_train, log)
        historial += hist
        _, lv = inferir(m, va.ruta.tolist(), tf_eval); _, lt = inferir(m, te.ruta.tolist(), tf_eval)
        mv, mt = metricas(lv, va.y.values), metricas(lt, te.y.values)
        filas.append({"semilla": s, **{f"val_{k}": v for k, v in mv.items()}, **{f"test_{k}": v for k, v in mt.items()}})
        if mejor is None or (mv["bal_acc"], -mv["perdida"]) > (mejor[0]["bal_acc"], -mejor[0]["perdida"]):
            mejor = (mv, mt, s, {k: v.detach().cpu().half() for k, v in m.state_dict().items()})
    pd.DataFrame(historial).to_csv(out / "historial.csv", index=False)

    res = pd.DataFrame(filas)
    log("\n=== RESULTADOS POR SEMILLA (modelo entrenado solo con train; el test no interviene en ninguna decisión) ===")
    log(res.round(3).to_string(index=False))
    log(f"\nMedia ± desviación entre semillas -> val: bal_acc {res.val_bal_acc.mean():.3f} ± {res.val_bal_acc.std(ddof=0):.3f} | "
        f"test: bal_acc {res.test_bal_acc.mean():.3f} ± {res.test_bal_acc.std(ddof=0):.3f}, auc {res.test_auc.mean():.3f}, "
        f"errores {res.test_errores.mean():.1f} de {len(te)}")
    mv, mt, semilla, estado = mejor
    log(f"Se despliega la semilla {semilla} (mejor validación): val bal_acc {mv['bal_acc']:.3f}; test bal_acc {mt['bal_acc']:.3f}, AUC {mt['auc']:.3f}.")

    # --- rechazo: covarianza con las imágenes de train, umbral de distancia con la validación (especies no vistas) ---
    m = timm.create_model(ARQUITECTURA, pretrained=False, num_classes=2)
    m.load_state_dict({k: v.float() for k, v in estado.items()})
    f_tr, _ = inferir(m, tr.ruta.tolist(), tf_eval); f_va, lg_va = inferir(m, va.ruta.tolist(), tf_eval); f_te, lg_te = inferir(m, te.ruta.tolist(), tf_eval)
    cov = LedoitWolf().fit(f_tr)
    umbral = float(np.percentile(cov.mahalanobis(f_va), args.percentil))
    areas = np.array([(cv2.imread(str(SEGMENTATION_DIR / r.dataset / "masks" / r.filename), cv2.IMREAD_GRAYSCALE) > 0).mean()
                      for r in pd.concat([tr, va]).itertuples()])
    a_min, a_max = float(0.8 * areas.min()), float(1.1 * areas.max())
    p_te = torch.softmax(torch.tensor(lg_te), 1).numpy(); d_te = cov.mahalanobis(f_te)
    rech = d_te > umbral; inc = ~rech & (p_te.max(1) < args.umbral_confianza); acept = ~rech & ~inc
    pred = p_te.argmax(1)
    log(f"\nRechazo (umbral de distancia = p{args.percentil:g} de la validación = {umbral:.0f}): en test, rechazadas por distancia "
        f"{rech.sum()} ({rech.mean():.0%}), inciertas {inc.sum()} ({inc.mean():.0%}), aceptadas {acept.sum()}; exactitud sin filtro "
        f"{(pred == te.y.values).mean():.3f}, solo en las aceptadas {(pred[acept] == te.y.values[acept]).mean():.3f}")

    joblib.dump({"tipo": "afinado", "red": "convnext", "arch": ARQUITECTURA, "entrada": "recorte", "segmentador": args.segmentador,
                 "pesos": estado, "clases": CLASES, "covarianza": cov, "umbral_distancia": umbral,
                 "umbral_confianza": args.umbral_confianza, "area_min": a_min, "area_max": a_max,
                 "metricas_test": {"balanced_acc": mt["bal_acc"], "auc": mt["auc"], "imagenes": len(te), "especies": te.grupo.nunique(),
                                   "semillas": len(args.semillas), "bal_acc_media_semillas": float(res.test_bal_acc.mean())}},
                out / "modelo_afinado.joblib")
    (out / "resumen.txt").write_text("\n".join(lineas), encoding="utf-8")
    log(f"\nGuardado: {(out / 'modelo_afinado.joblib').resolve()}")


if __name__ == "__main__":
    main()
