from __future__ import annotations

# Localiza rutas.py (en scripts/, o junto a este archivo en la app) y agrega al path la carpeta de cada proceso
import sys
from pathlib import Path
sys.path[:0] = [str(p) for p in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]) if (p / "rutas.py").exists()][:1]
import rutas  # noqa: E402,F401


import argparse
import random
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

# ----------------------------------------------------------------------
# CONFIG (usado si no se pasan argumentos por línea de comandos)
# ----------------------------------------------------------------------
from rutas import DATASET_DIR, SEGMENTATION_DIR as SEGMENTATION_ROOT, SPLIT_DIR

# Datasets de entrada (subcarpetas de SEGMENTATION_ROOT que contienen segmented/ y masks/).
# Usa UNA sola versión de segmentación por familia, nunca dos a la vez.
DEFAULT_DATASETS = ["Dataset_papilionidae", "Dataset_pieridae"]

DEFAULT_FEATURES_CSV = SEGMENTATION_ROOT / "features_dataset_ready.csv"
DEFAULT_ORIGINAL_ROOT = DATASET_DIR         # fotos originales en dataset/<Familia>/
DEFAULT_OUTPUT_DIR = SPLIT_DIR

SPLIT_NAMES = ["train", "val", "test"]
DEFAULT_RATIOS = (0.70, 0.15, 0.15)
SEED = 42
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ----------------------------------------------------------------------
# Utilidades de nombres y etiquetas
# ----------------------------------------------------------------------

def derive_family(dataset_name: str) -> str:
    """'Dataset_papilionidae_v2' -> 'Papilionidae' (misma regla que train_family_classifier.py)."""
    s = str(dataset_name).strip().lower()
    s = re.sub(r"^dataset[_\- ]*", "", s)
    s = re.sub(r"[_\- ]*v\d+$", "", s)
    return s.strip("_- ").capitalize()


def derive_species_group(label) -> str:
    """Primeros dos tokens del label ('Battus crassus lepidus' -> 'battus crassus'),
    para que las subespecies de una misma especie queden juntas. Devuelve ''
    si no hay etiqueta."""
    if not isinstance(label, str) or not label.strip():
        return ""
    return " ".join(re.split(r"[ _]+", label.strip())[:2]).lower()


def label_from_filename(filename: str):
    """Intenta sacar la especie del nombre de archivo. Devuelve None si no se puede."""
    stem = Path(filename).stem
    stem = re.sub(r"^\d+_", "", stem)                                   # 013_Dismorphia_... -> Dismorphia_...
    stem = re.sub(r"_image_?\d+$", "", stem, flags=re.IGNORECASE)       # ..._Image_12
    tokens = [t for t in re.split(r"[ _]+", stem) if t]
    if len(tokens) < 2 or not all(re.search(r"[A-Za-z]", t) for t in tokens[:2]):
        return None
    return "_".join(tokens)


def load_labels_csvs(paths: list) -> dict:
    mapping = {}
    for p in paths:
        p = Path(p)
        if not p.exists():
            print(f"[AVISO] No se encontró el CSV de labels: {p} (se ignora)")
            continue
        df = pd.read_csv(p)
        cols = {c.lower(): c for c in df.columns}
        fcol = cols.get("filename") or cols.get("file") or cols.get("image") or df.columns[0]
        lcol = cols.get("label") or cols.get("species") or cols.get("class") or df.columns[1]
        for fname, lab in zip(df[fcol], df[lcol]):
            if isinstance(lab, str) and lab.strip():
                mapping[str(fname)] = lab.strip()
    return mapping


# ----------------------------------------------------------------------
# Inventario de imágenes
# ----------------------------------------------------------------------

def build_inventory(root: Path, names: list) -> pd.DataFrame:
    rows = []
    for name in names:
        ds_dir = root / name
        seg_dir = ds_dir / "segmented"
        if not seg_dir.exists():
            print(f"[ERROR] No se encontró la carpeta: {seg_dir}")
            if root.exists():
                available = sorted(p.name for p in root.iterdir() if (p / "segmented").exists())
                print(f"Datasets con carpeta 'segmented' dentro de {root}: {available if available else 'ninguno'}")
                print("Usa --datasets con los nombres correctos, o --root si la carpeta raíz es otra.")
            sys.exit(1)
        mask_dir = ds_dir / "masks"
        family = derive_family(name)
        for p in sorted(seg_dir.iterdir()):
            if p.suffix.lower() in VALID_EXTS:
                mp = mask_dir / p.name
                rows.append({"dataset": name, "family": family, "filename": p.name,
                             "seg_path": str(p), "mask_path": str(mp) if mp.exists() else ""})
    df = pd.DataFrame(rows)
    if df.empty:
        print("[ERROR] No se encontró ninguna imagen en las carpetas 'segmented'.")
        sys.exit(1)
    if df.duplicated(subset=["family", "filename"]).any():
        print("[ERROR] Hay imágenes repetidas dentro de una misma familia: seguramente pasaste dos versiones "
              "(v1 y v2) del mismo dataset. Usa solo una versión por familia en --datasets.")
        sys.exit(1)
    return df


def index_originals(original_root: Path, families: list) -> dict:
    """{(familia, stem): ruta} de las fotos originales en original_root/<Familia>/."""
    index = {}
    for fam in families:
        fam_dir = original_root / fam
        if not fam_dir.exists():
            continue
        for p in fam_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in VALID_EXTS:
                index[(fam, p.stem)] = p
    return index


# ----------------------------------------------------------------------
# Etiquetas y calidad
# ----------------------------------------------------------------------

def attach_labels(df: pd.DataFrame, features_df, labels_csvs: list, log) -> pd.DataFrame:
    df = df.copy()
    labels_dict = load_labels_csvs(labels_csvs) if labels_csvs else {}

    lab_csv = df["filename"].map(labels_dict) if labels_dict else pd.Series([None] * len(df), index=df.index)

    lab_feat = pd.Series([None] * len(df), index=df.index)
    if features_df is not None and "label" in features_df.columns:
        fmap = {(f, n): l for f, n, l in zip(features_df["family"], features_df["filename"], features_df["label"])
                if isinstance(l, str) and l.strip()}
        lab_feat = pd.Series([fmap.get((f, n)) for f, n in zip(df["family"], df["filename"])], index=df.index)

    lab_name = df["filename"].map(label_from_filename)

    labels, sources = [], []
    for a, b, c in zip(lab_csv, lab_feat, lab_name):
        if isinstance(a, str):
            labels.append(a); sources.append("--labels")
        elif isinstance(b, str):
            labels.append(b); sources.append("features_csv")
        elif isinstance(c, str):
            labels.append(c); sources.append("nombre_de_archivo")
        else:
            labels.append(""); sources.append("sin_label")
    df["label"] = labels
    df["label_source"] = sources
    df["species_group"] = [derive_species_group(l) for l in labels]

    log("Origen de la etiqueta de especie:\n" + df["label_source"].value_counts().to_string())
    ejemplo = df[df["label_source"] == "nombre_de_archivo"].head(2)
    for _, r in ejemplo.iterrows():
        log(f"  ejemplo de lectura por nombre: '{r['filename']}' -> '{r['label']}'")
    return df


# ----------------------------------------------------------------------
# Asignación de particiones
# ----------------------------------------------------------------------

def assign_splits(df: pd.DataFrame, ratios, seed: int, mode: str) -> pd.DataFrame:
    """Reparte GRUPOS completos (especies) a cada partición, familia por familia.

    Cada grupo va a la partición con mayor déficit relativo respecto a su
    objetivo; los grupos grandes se colocan primero y los pequeños rellenan
    los huecos, así las proporciones finales quedan cerca de las pedidas.
    """
    rng = random.Random(seed)
    df = df.copy()
    # sin especie, o en modo random, cada imagen es su propio grupo
    if mode == "random":
        df["group_key"] = df["filename"]
    else:
        df["group_key"] = [g if g else f"__sin_label__{fn}" for g, fn in zip(df["species_group"], df["filename"])]

    split_of = {}
    for family, fam in df.groupby("family"):
        sizes = fam.groupby("group_key").size().to_dict()
        keys = list(sizes)
        rng.shuffle(keys)                      # el azar desempata entre grupos del mismo tamaño
        keys.sort(key=lambda k: -sizes[k])     # orden estable: primero los grupos grandes
        total = len(fam)
        targets = [r * total for r in ratios]
        current = [0, 0, 0]
        for k in keys:
            deficits = [(targets[i] - current[i]) / targets[i] if targets[i] > 0 else float("-inf")
                        for i in range(3)]
            i = max(range(3), key=lambda j: deficits[j])
            split_of[(family, k)] = SPLIT_NAMES[i]
            current[i] += sizes[k]

    df["split"] = [split_of[(f, k)] for f, k in zip(df["family"], df["group_key"])]
    return df


def build_report(df: pd.DataFrame, ratios, mode: str, log):
    log("\n=== RESULTADO DE LA DIVISIÓN ===")
    counts = pd.crosstab(df["family"], df["split"]).reindex(columns=SPLIT_NAMES, fill_value=0)
    counts["total"] = counts.sum(axis=1)
    log("Imágenes por familia y partición:\n" + counts.to_string())

    frac = counts[SPLIT_NAMES].div(counts["total"], axis=0).round(3)
    log("\nProporción lograda por familia (objetivo: "
        + " / ".join(f"{n} {r:.2f}" for n, r in zip(SPLIT_NAMES, ratios)) + "):\n" + frac.to_string())

    glob_counts = df["split"].value_counts().reindex(SPLIT_NAMES, fill_value=0)
    log("\nTotal por partición: " + ", ".join(f"{n}={glob_counts[n]} ({glob_counts[n] / len(df):.1%})" for n in SPLIT_NAMES))

    labeled = df[df["species_group"] != ""]
    if len(labeled):
        spp = labeled.groupby(["family", "split"])["species_group"].nunique().unstack(fill_value=0)
        spp = spp.reindex(columns=SPLIT_NAMES, fill_value=0)
        log("\nEspecies distintas por familia y partición:\n" + spp.to_string())

    # --- Verificaciones ---
    log("\n=== VERIFICACIONES ===")
    problems = 0
    for fam in sorted(df["family"].unique()):
        for s in SPLIT_NAMES:
            if counts.loc[fam, s] == 0:
                log(f"[AVISO] La familia {fam} no tiene ninguna imagen en '{s}' (muy pocas especies para repartir).")
                problems += 1
    n_unlabeled = int((df["species_group"] == "").sum())
    if n_unlabeled:
        log(f"[AVISO] {n_unlabeled} imágenes sin especie reconocible se repartieron de forma aleatoria.")
    if mode == "group" and len(labeled):
        spread = labeled.groupby(["family", "species_group"])["split"].nunique()
        n_leak = int((spread > 1).sum())
        if n_leak == 0:
            log("OK: ninguna especie aparece en más de una partición (sin fuga entre train, val y test).")
        else:
            log(f"[ERROR] {n_leak} especies aparecen en más de una partición. Esto no debería pasar.")
            problems += 1
    elif mode == "random" and len(labeled):
        spread = labeled.groupby(["family", "species_group"])["split"].nunique()
        log(f"Modo random: {int((spread > 1).sum())} de {len(spread)} especies quedaron repartidas entre varias "
            f"particiones. Una evaluación sobre este test será más optimista que una agrupada por especie.")
    small = [n for n in SPLIT_NAMES if glob_counts[n] < 10]
    if small:
        log(f"[AVISO] Las particiones {small} tienen menos de 10 imágenes; las métricas sobre ellas serán muy ruidosas.")
    return problems


# ----------------------------------------------------------------------
# Copia de archivos y salida
# ----------------------------------------------------------------------

def guard_output_dir(out_dir: Path, input_dirs: list):
    out = out_dir.resolve()
    for d in input_dirs:
        d = Path(d).resolve()
        if d == out or out in d.parents:
            print(f"[ERROR] La carpeta de salida ({out}) contiene o coincide con una carpeta de entrada ({d}).")
            print("Elige una carpeta de salida distinta, para no mezclar ni borrar tus datos de origen.")
            sys.exit(1)


def copy_split_files(df: pd.DataFrame, kinds: list, out_dir: Path, original_index: dict, log):
    for kind in kinds:
        missing = 0
        copied = 0
        for r in df.itertuples():
            if kind == "segmented":
                src = Path(r.seg_path)
            elif kind == "masks":
                src = Path(r.mask_path) if r.mask_path else None
            else:
                src = original_index.get((r.family, Path(r.filename).stem))
            if src is None or not Path(src).exists():
                missing += 1
                continue
            dst = out_dir / kind / r.split / r.family / Path(src).name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
        log(f"Copiadas {copied} imágenes de tipo '{kind}'" + (f" ({missing} sin archivo correspondiente)" if missing else ""))


def write_feature_splits(df: pd.DataFrame, features_df, out_dir: Path, log):
    if features_df is None:
        return
    key = df[["family", "filename", "split"]]
    merged = features_df.merge(key, on=["family", "filename"], how="left")
    no_image = int(merged["split"].isna().sum())
    no_features = len(df) - int(merged["split"].notna().sum())
    log("\n=== FEATURES POR PARTICIÓN ===")
    if no_image:
        log(f"{no_image} filas de features no tienen imagen asignada (por ejemplo, filas low_quality) y se omiten.")
    if no_features:
        log(f"{no_features} imágenes no tienen fila en el CSV de features.")
    for s in SPLIT_NAMES:
        part = merged[merged["split"] == s].drop(columns=["family", "split"])
        part.to_csv(out_dir / f"features_{s}.csv", index=False)
        log(f"  features_{s}.csv: {len(part)} filas")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=str(SEGMENTATION_ROOT), help="Carpeta raíz con los datasets segmentados")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                        help="Subcarpetas de --root a dividir (una versión por familia)")
    parser.add_argument("--features-csv", default=str(DEFAULT_FEATURES_CSV),
                        help="CSV de features para dividir también y leer labels ('none' para omitirlo)")
    parser.add_argument("--labels", nargs="*", default=[], help="CSV(s) opcionales (filename,label)")
    parser.add_argument("--original-root", default=str(DEFAULT_ORIGINAL_ROOT),
                        help="Carpeta que contiene <Familia>/ con las fotos originales (solo con --also original)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR), help="Carpeta de salida")
    parser.add_argument("--ratios", nargs=3, type=float, default=list(DEFAULT_RATIOS), metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--split-mode", choices=["group", "random"], default="group",
                        help="'group' (por especie, recomendado) o 'random' (imagen por imagen)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--also", nargs="*", choices=["masks", "original"], default=[],
                        help="Copiar además las máscaras y/o las fotos originales")
    parser.add_argument("--keep-low-quality", action="store_true",
                        help="Incluir imágenes marcadas low_quality en features_csv (por defecto se excluyen)")
    parser.add_argument("--no-copy", action="store_true", help="Solo manifiesto y reporte, sin copiar imágenes")
    parser.add_argument("--overwrite", action="store_true",
                        help="Si la salida ya tiene imágenes, borrarlas y volver a generarlas")
    args = parser.parse_args()

    out_dir = Path(args.output)
    lines = []

    def log(msg=""):
        print(msg)
        lines.append(str(msg))

    ratios = list(args.ratios)
    if any(r <= 0 for r in ratios):
        print("[ERROR] Las tres proporciones deben ser mayores que 0.")
        sys.exit(1)
    if abs(sum(ratios) - 1.0) > 1e-6:
        total = sum(ratios)
        ratios = [r / total for r in ratios]
        log(f"[AVISO] Las proporciones no sumaban 1; se normalizaron a {[round(r, 3) for r in ratios]}.")

    root = Path(args.root)
    df = build_inventory(root, args.datasets)
    log(f"Imágenes encontradas: {len(df)} en {len(args.datasets)} datasets")

    # --- CSV de features (opcional) ---
    features_df = None
    if str(args.features_csv).lower() != "none":
        fpath = Path(args.features_csv)
        if fpath.exists():
            features_df = pd.read_csv(fpath)
            if {"dataset", "filename"} <= set(features_df.columns):
                features_df["family"] = features_df["dataset"].apply(derive_family)
                log(f"Features cargados: {len(features_df)} filas ({fpath.name})")
            else:
                log("[AVISO] El CSV de features no tiene 'dataset' y 'filename'; no se usará.")
                features_df = None
        else:
            log(f"[AVISO] No existe {fpath}; se divide solo el conjunto de imágenes.")

    # --- Excluir low_quality ---
    if features_df is not None and "low_quality" in features_df.columns and not args.keep_low_quality:
        lq = features_df["low_quality"].astype(str).str.lower().isin(["true", "1"])
        bad = set(zip(features_df.loc[lq, "family"], features_df.loc[lq, "filename"]))
        mask_bad = [(f, n) in bad for f, n in zip(df["family"], df["filename"])]
        if any(mask_bad):
            log(f"Se excluyen {sum(mask_bad)} imágenes marcadas low_quality (usa --keep-low-quality para incluirlas).")
            df = df[[not b for b in mask_bad]].reset_index(drop=True)

    df = attach_labels(df, features_df, args.labels, log)
    df = assign_splits(df, ratios, args.seed, args.split_mode)
    problems = build_report(df, ratios, args.split_mode, log)

    # --- Salida ---
    kinds = ["segmented"] + list(args.also)
    original_index = {}
    input_dirs = [Path(p) for p in df["seg_path"].map(lambda s: Path(s).parent).unique()]
    if "masks" in kinds:
        input_dirs += [Path(p).parent for p in df["mask_path"].unique() if p]
    if "original" in kinds:
        original_index = index_originals(Path(args.original_root), sorted(df["family"].unique()))
        input_dirs += [Path(args.original_root) / f for f in df["family"].unique()]
        if not original_index:
            log(f"[AVISO] No se encontraron fotos originales en {args.original_root}\\<Familia>. Revisa --original-root.")
    guard_output_dir(out_dir, input_dirs)

    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_copy:
        existing = [out_dir / k for k in kinds if (out_dir / k).exists() and any((out_dir / k).rglob("*"))]
        if existing:
            if not args.overwrite:
                print(f"[ERROR] La salida ya contiene imágenes de una ejecución anterior: {[str(d) for d in existing]}")
                print("Mezclar ejecuciones con semillas o proporciones distintas filtraría imágenes entre train y test.")
                print("Usa otra carpeta con --output, o --overwrite para borrarlas y regenerarlas.")
                sys.exit(1)
            for d in existing:
                shutil.rmtree(d)
        log("")
        copy_split_files(df, kinds, out_dir, original_index, log)

    manifest = df[["dataset", "family", "filename", "label", "label_source", "species_group", "split"]]
    manifest.sort_values(["split", "family", "species_group", "filename"]).to_csv(out_dir / "splits.csv", index=False)
    write_feature_splits(df, features_df, out_dir, log)

    log(f"\nManifiesto: {out_dir / 'splits.csv'}")
    (out_dir / "split_report.txt").write_text("\n".join(lines), encoding="utf-8")
    log(f"Reporte:    {out_dir / 'split_report.txt'}")
    if problems:
        log("\n[ATENCIÓN] Hay avisos en la sección VERIFICACIONES. Revísalos antes de usar esta división.")
    (out_dir / "split_report.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()