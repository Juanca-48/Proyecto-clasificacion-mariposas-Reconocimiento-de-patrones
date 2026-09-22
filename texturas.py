from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import gabor
import mahotas

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

GLCM_LEVELS = 32                       # niveles de gris para cuantizar (sin contar el 0 reservado a fondo)
GLCM_DISTANCES = [1, 2, 3]             # distancias (en píxeles) entre pares de co-ocurrencia
GLCM_ANGLES_DEG = [0, 45, 90, 135]     # ángulos en grados
GLCM_PROPS = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]

WHITE_BG_THRESHOLD = 245   # si no hay máscara, se asume fondo = píxel con los 3 canales >= este valor
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

HARALICK_NAMES = [
    "angular_second_moment", "contrast", "correlation", "variance",
    "inverse_diff_moment", "sum_average", "sum_variance", "sum_entropy",
    "entropy", "difference_variance", "difference_entropy",
    "info_measure_corr_1", "info_measure_corr_2",
]

ENABLE_GABOR = True
GABOR_FREQUENCIES = [0.1, 0.2, 0.3, 0.4]   # ciclos por píxel (más alto = detalle más fino)
GABOR_THETAS_DEG = [0, 45, 90, 135]        # orientaciones del filtro


# ----------------------------------------------------------------------
# Máscara de foreground
# ----------------------------------------------------------------------

def load_or_derive_mask(img_bgr: np.ndarray, mask_path: Path | None) -> np.ndarray:
    """Carga la máscara binaria si existe; si no, la deriva asumiendo que
    el fondo es (casi) blanco puro, como lo dejan los scripts de
    segmentación anteriores."""
    if mask_path is not None and mask_path.exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            return mask

    # Fallback: fondo = casi blanco en los 3 canales
    lower = np.array([WHITE_BG_THRESHOLD] * 3, dtype=np.uint8)
    white = cv2.inRange(img_bgr, lower, np.array([255, 255, 255], dtype=np.uint8))
    mask = cv2.bitwise_not(white)
    return mask


def crop_to_mask_bbox(gray: np.ndarray, mask: np.ndarray, pad: int = 4):
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return gray, mask  # imagen vacía / sin mariposa detectada
    y0, y1 = max(0, ys.min() - pad), min(mask.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(mask.shape[1], xs.max() + pad + 1)
    return gray[y0:y1, x0:x1], mask[y0:y1, x0:x1]


# ----------------------------------------------------------------------
# Método 1: GLCM clásico (scikit-image), excluyendo el fondo
# ----------------------------------------------------------------------

def quantize_with_background(gray: np.ndarray, mask: np.ndarray, levels: int) -> np.ndarray:
    """Cuantiza la imagen a `levels` niveles (1..levels) y reserva el nivel
    0 exclusivamente para el fondo (mask == 0)."""
    bins = np.linspace(0, 256, levels + 1)
    quantized = np.digitize(gray, bins[1:-1]).astype(np.uint8) + 1  # valores en [1, levels]
    quantized[mask == 0] = 0
    return quantized


def glcm_features(gray: np.ndarray, mask: np.ndarray) -> dict:
    quantized = quantize_with_background(gray, mask, GLCM_LEVELS)
    angles_rad = [np.deg2rad(a) for a in GLCM_ANGLES_DEG]

    glcm = graycomatrix(
        quantized,
        distances=GLCM_DISTANCES,
        angles=angles_rad,
        levels=GLCM_LEVELS + 1,   # +1 por el nivel 0 reservado a fondo
        symmetric=True,
        normed=False,
    )
    # Anular toda transición desde/hacia el nivel 0 (fondo). graycoprops
    # renormaliza internamente por la suma de cada matriz, así que el
    # fondo queda completamente excluido del cálculo de propiedades.
    glcm[0, :, :, :] = 0
    glcm[:, 0, :, :] = 0

    feats = {}
    for prop in GLCM_PROPS:
        values = graycoprops(glcm, prop)  # shape (n_distances, n_angles)
        # Promedio y desviación estándar entre ángulos -> invarianza rotacional,
        # una columna por distancia.
        mean_over_angles = values.mean(axis=1)
        std_over_angles = values.std(axis=1)
        for i, d in enumerate(GLCM_DISTANCES):
            feats[f"glcm_{prop}_d{d}_mean"] = float(mean_over_angles[i])
            feats[f"glcm_{prop}_d{d}_std"] = float(std_over_angles[i])
    return feats


# ----------------------------------------------------------------------
# Método 2: Haralick (mahotas), excluyendo el fondo vía ignore_zeros
# ----------------------------------------------------------------------

def haralick_features(gray: np.ndarray, mask: np.ndarray) -> dict:
    # Desplazar el rango real a [1,255] para que el 0 quede reservado
    # exclusivamente al fondo (muchas alas son negras/oscuras: sin este
    # desplazamiter, ignore_zeros también las excluiría por error).
    shifted = 1 + (gray.astype(np.float32) * (254.0 / 255.0))
    shifted = shifted.astype(np.uint8)
    shifted[mask == 0] = 0

    try:
        values = mahotas.features.haralick(shifted, ignore_zeros=True, return_mean=True)
    except ValueError:
        # Puede fallar si, tras excluir el fondo, queda muy poca área útil
        return {f"haralick_{name}": np.nan for name in HARALICK_NAMES}

    return {f"haralick_{name}": float(v) for name, v in zip(HARALICK_NAMES, values)}


# ----------------------------------------------------------------------
# Método 3: Filtros de Gabor (scikit-image), excluyendo el fondo
# ----------------------------------------------------------------------

def gabor_feature_names() -> list:
    names = []
    for freq in GABOR_FREQUENCIES:
        for theta_deg in GABOR_THETAS_DEG:
            names.append(f"gabor_f{freq}_t{theta_deg}_mean")
            names.append(f"gabor_f{freq}_t{theta_deg}_std")
    return names


def gabor_features(gray: np.ndarray, mask: np.ndarray) -> dict:
    fg_vals = gray[mask > 0]
    if fg_vals.size == 0:
        return {name: np.nan for name in gabor_feature_names()}

    # Rellenar el fondo con el gris medio del primer plano: evita un borde
    # artificial de alto contraste entre mariposa y fondo blanco/negro que
    # el filtro interpretaría como textura falsa justo en el contorno.
    fill_value = float(fg_vals.mean())
    gray_filled = gray.astype(np.float64)
    gray_filled[mask == 0] = fill_value
    img_norm = gray_filled / 255.0  # normalizado a [0,1], escala consistente entre imágenes

    feats = {}
    for freq in GABOR_FREQUENCIES:
        for theta_deg in GABOR_THETAS_DEG:
            real, imag = gabor(img_norm, frequency=freq, theta=np.deg2rad(theta_deg))
            magnitude = np.sqrt(real ** 2 + imag ** 2)
            vals = magnitude[mask > 0]  # estadísticas solo sobre píxeles reales de la mariposa
            feats[f"gabor_f{freq}_t{theta_deg}_mean"] = float(vals.mean())
            feats[f"gabor_f{freq}_t{theta_deg}_std"] = float(vals.std())
    return feats


# ----------------------------------------------------------------------
# Pipeline principal
# ----------------------------------------------------------------------

def process_image(img_path: Path, mask_path: Path | None) -> dict | None:
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return None

    mask = load_or_derive_mask(img_bgr, mask_path)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray, mask = crop_to_mask_bbox(gray, mask)

    if (mask > 0).sum() < 50:
        return None  # muy poca área de mariposa detectada, se descarta

    feats = {"filename": img_path.name, "fg_pixels": int((mask > 0).sum())}
    feats.update(glcm_features(gray, mask))
    feats.update(haralick_features(gray, mask))
    if ENABLE_GABOR:
        feats.update(gabor_features(gray, mask))
    return feats


def load_labels(labels_csv: Path) -> dict:
    """Lee un CSV (filename,label,...) y devuelve {filename: label}."""
    mapping = {}
    with open(labels_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        # admite distintos nombres de columna habituales
        fname_col = next((c for c in reader.fieldnames if c.lower() in ("filename", "file", "image")), reader.fieldnames[0])
        label_col = next((c for c in reader.fieldnames if c.lower() in ("label", "species", "class")), reader.fieldnames[1])
        for row in reader:
            mapping[row[fname_col]] = row[label_col]
    return mapping


def process_folder(images_dir: Path, masks_dir: Path | None, output_csv: Path,
                    labels_map: dict, dataset_tag: str = "") -> list:
    """Procesa todas las imágenes de una carpeta y escribe su propio CSV.
    Devuelve la lista de filas (dicts) para poder combinarlas entre datasets."""
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
    """Devuelve [(nombre, images_dir, masks_dir), ...]. Si `names` está
    vacío, autodetecta cualquier subcarpeta de `root` que contenga una
    carpeta 'segmented' adentro."""
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

    # --- Modo un solo dataset: se pasó --images explícitamente ---
    if args.images:
        images_dir = Path(args.images)
        masks_dir = Path(args.masks) if args.masks else None
        output_csv = Path(args.output) if args.output else images_dir.parent / "texture_features.csv"
        rows = process_folder(images_dir, masks_dir, output_csv, labels_map)
        if not rows:
            sys.exit(1)
        print(f"\nListo. CSV guardado en: {output_csv.resolve()}")
        return

    # --- Modo automático: varios datasets dentro de --root ---
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
        output_csv = root / name / "texture_features.csv"
        rows = process_folder(images_dir, masks_dir, output_csv, labels_map, dataset_tag=name)
        all_rows.extend(rows)

    if not all_rows:
        print("\n[ERROR] No se generó ninguna fila de features en ningún dataset.")
        sys.exit(1)

    # CSV combinado con todos los datasets juntos (útil para entrenar un
    # solo clasificador con ambas familias)
    combined_csv = root / "texture_features_all.csv"
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