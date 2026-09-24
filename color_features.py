from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

# ----------------------------------------------------------------------
# CONFIG (usado si no se pasan argumentos por línea de comandos)
# ----------------------------------------------------------------------
REPO_DIR = Path(r"D:\Trabajos\Universidad\patrones\Proyecto_mariposas\Git_mariposas\Proyecto-clasificacion-mariposas-Reconocimiento-de-patrones")
SEGMENTATION_ROOT = REPO_DIR / "segmentacion"

# Nombres de las subcarpetas de dataset dentro de SEGMENTATION_ROOT. Si se
# deja vacío ([]), el script las autodetecta: cualquier subcarpeta que
# contenga una carpeta "segmented" adentro se trata como un dataset.
DEFAULT_DATASETS = ["Dataset_papilionidae", "Dataset_pieridae"]

DEFAULT_LABELS_CSV = None   # opcional: ruta a un labels.csv con columnas filename,label

ENABLE_HISTOGRAM = True
HIST_BINS = 16              # bins por canal para el histograma HSV
ENABLE_MOMENTS = True

WHITE_BG_THRESHOLD = 245    # si no hay máscara, se asume fondo = píxel con los 3 canales >= este valor
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

HSV_RANGES = {"h": (0, 180), "s": (0, 256), "v": (0, 256)}
BGR_CHANNEL_NAMES = ["b", "g", "r"]
HSV_CHANNEL_NAMES = ["h", "s", "v"]


# ----------------------------------------------------------------------
# Máscara de foreground (misma lógica que texturas.py)
# ----------------------------------------------------------------------

def load_or_derive_mask(img_bgr: np.ndarray, mask_path: Path | None) -> np.ndarray:
    if mask_path is not None and mask_path.exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            return mask

    lower = np.array([WHITE_BG_THRESHOLD] * 3, dtype=np.uint8)
    white = cv2.inRange(img_bgr, lower, np.array([255, 255, 255], dtype=np.uint8))
    mask = cv2.bitwise_not(white)
    return mask


# ----------------------------------------------------------------------
# Descriptores de color
# ----------------------------------------------------------------------

def histogram_features(img_bgr: np.ndarray, mask: np.ndarray) -> dict:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    feats = {}
    for ch_idx, ch_name in enumerate(HSV_CHANNEL_NAMES):
        vals = hsv[:, :, ch_idx][mask > 0]
        lo, hi = HSV_RANGES[ch_name]
        if vals.size == 0:
            hist = np.full(HIST_BINS, np.nan)
        else:
            hist, _ = np.histogram(vals, bins=HIST_BINS, range=(lo, hi), density=True)
        for i, v in enumerate(hist):
            feats[f"hist_{ch_name}_{i:02d}"] = float(v)
    return feats


def _skewness(vals: np.ndarray, mean: float, std: float) -> float:
    if std < 1e-8:
        return 0.0
    return float(np.mean(((vals - mean) / std) ** 3))


def moment_features(img_bgr: np.ndarray, mask: np.ndarray) -> dict:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    feats = {}
    for space_name, img, names in [("bgr", img_bgr, BGR_CHANNEL_NAMES), ("hsv", hsv, HSV_CHANNEL_NAMES)]:
        for ch_idx, ch_name in enumerate(names):
            vals = img[:, :, ch_idx][mask > 0].astype(np.float64)
            prefix = f"moment_{space_name}_{ch_name}"
            if vals.size == 0:
                feats[f"{prefix}_mean"] = np.nan
                feats[f"{prefix}_std"] = np.nan
                feats[f"{prefix}_skew"] = np.nan
                continue
            mean = float(vals.mean())
            std = float(vals.std())
            feats[f"{prefix}_mean"] = mean
            feats[f"{prefix}_std"] = std
            feats[f"{prefix}_skew"] = _skewness(vals, mean, std)
    return feats


# ----------------------------------------------------------------------
# Pipeline principal (misma estructura que texturas.py)
# ----------------------------------------------------------------------

def process_image(img_path: Path, mask_path: Path | None) -> dict | None:
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return None

    mask = load_or_derive_mask(img_bgr, mask_path)
    if (mask > 0).sum() < 50:
        return None  # muy poca área de mariposa detectada, se descarta

    feats = {"filename": img_path.name, "fg_pixels": int((mask > 0).sum())}
    if ENABLE_HISTOGRAM:
        feats.update(histogram_features(img_bgr, mask))
    if ENABLE_MOMENTS:
        feats.update(moment_features(img_bgr, mask))
    return feats


def load_labels(labels_csv: Path) -> dict:
    mapping = {}
    with open(labels_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fname_col = next((c for c in reader.fieldnames if c.lower() in ("filename", "file", "image")), reader.fieldnames[0])
        label_col = next((c for c in reader.fieldnames if c.lower() in ("label", "species", "class")), reader.fieldnames[1])
        for row in reader:
            mapping[row[fname_col]] = row[label_col]
    return mapping


def process_folder(images_dir: Path, masks_dir: Path | None, output_csv: Path,
                    labels_map: dict, dataset_tag: str = "") -> list:
    if not images_dir.exists():
        print(f"[ERROR] No se encontró la carpeta de imágenes: {images_dir}")
        return []

    files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in VALID_EXTS)
    print(f"Procesando {len(files)} imágenes de {images_dir} ...")

    rows = []
    skipped = []
    for i, path in enumerate(files):
        mask_path = (masks_dir / path.name) if masks_dir else None
        feats = process_image(path, mask_path)
        if feats is None:
            skipped.append(path.name)
            continue
        if labels_map:
            feats["label"] = labels_map.get(path.name, "")
        if dataset_tag:
            feats["dataset"] = dataset_tag
        rows.append(feats)

        if (i + 1) % 25 == 0 or (i + 1) == len(files):
            print(f"  {i + 1}/{len(files)}")

    if not rows:
        print(f"[AVISO] No se generó ninguna fila de features para {images_dir}.")
        return []

    fieldnames = ["filename"]
    if dataset_tag:
        fieldnames.append("dataset")
    if labels_map:
        fieldnames.append("label")
    fieldnames.append("fg_pixels")
    other_cols = sorted(k for k in rows[0].keys() if k not in fieldnames)
    fieldnames += other_cols

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"  -> {len(rows)} imágenes con features, {len(skipped)} omitidas. CSV: {output_csv}")
    if skipped:
        print(f"  [AVISO] Omitidas (sin suficiente área de mariposa detectada): {skipped[:10]}"
              + (" ..." if len(skipped) > 10 else ""))
    return rows


def discover_datasets(root: Path, names: list) -> list:
    result = []
    if names:
        for name in names:
            ds_dir = root / name
            result.append((name, ds_dir / "segmented", ds_dir / "masks"))
    else:
        for ds_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            if (ds_dir / "segmented").exists():
                result.append((ds_dir.name, ds_dir / "segmented", ds_dir / "masks"))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", default=None, help="Carpeta con las imágenes (modo un solo dataset). Si se omite, se autodetectan los datasets dentro de --root")
    parser.add_argument("--masks", default=None, help="Carpeta con las máscaras binarias (modo un solo dataset). Opcional.")
    parser.add_argument("--root", default=str(SEGMENTATION_ROOT), help="Carpeta raíz con subcarpetas por dataset (modo automático, ej. 'segmentacion')")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS_CSV) if DEFAULT_LABELS_CSV else None, help="CSV opcional (filename,label) para agregar la etiqueta a cada fila")
    parser.add_argument("--output", default=None, help="Ruta del CSV de salida (modo un solo dataset)")
    args = parser.parse_args()

    labels_map = {}
    if args.labels:
        labels_path = Path(args.labels)
        if labels_path.exists():
            labels_map = load_labels(labels_path)
            print(f"Labels cargados: {len(labels_map)} desde {labels_path}")
        else:
            print(f"[AVISO] No se encontró el CSV de labels: {labels_path} (se continúa sin labels)")

    if args.images:
        images_dir = Path(args.images)
        masks_dir = Path(args.masks) if args.masks else None
        output_csv = Path(args.output) if args.output else images_dir.parent / "color_features.csv"
        rows = process_folder(images_dir, masks_dir, output_csv, labels_map)
        if not rows:
            sys.exit(1)
        print(f"\nListo. CSV guardado en: {output_csv.resolve()}")
        return

    root = Path(args.root)
    if not root.exists():
        print(f"[ERROR] No se encontró la carpeta raíz: {root}")
        sys.exit(1)

    datasets = discover_datasets(root, DEFAULT_DATASETS)
    if not datasets:
        print(f"[ERROR] No se encontró ningún dataset (subcarpeta con 'segmented' adentro) en: {root}")
        sys.exit(1)

    all_rows = []
    for name, images_dir, masks_dir in datasets:
        print(f"\n=== Dataset: {name} ===")
        output_csv = root / name / "color_features.csv"
        rows = process_folder(images_dir, masks_dir, output_csv, labels_map, dataset_tag=name)
        all_rows.extend(rows)

    if not all_rows:
        print("\n[ERROR] No se generó ninguna fila de features en ningún dataset.")
        sys.exit(1)

    combined_csv = root / "color_features_all.csv"
    fieldnames = ["filename", "dataset"]
    if labels_map:
        fieldnames.append("label")
    fieldnames.append("fg_pixels")
    other_cols = sorted(k for k in all_rows[0].keys() if k not in fieldnames)
    fieldnames += other_cols
    with open(combined_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    print(f"\nTodo listo: {len(all_rows)} imágenes en total.")
    print(f"CSV combinado: {combined_csv.resolve()}")


if __name__ == "__main__":
    main()