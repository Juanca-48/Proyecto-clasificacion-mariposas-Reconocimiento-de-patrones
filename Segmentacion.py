import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# ----------------------------------------------------------------------
# CONFIG (usado si no se pasan argumentos por línea de comandos)
# ----------------------------------------------------------------------
PROYECTO_DIR = Path(r"D:\Trabajos\Universidad\patrones\Proyecto_mariposas")

DEFAULT_IMAGES_DIRS = [
    PROYECTO_DIR / "Dataset_papilionidae" / "images",
    PROYECTO_DIR / "Dataset_pieridae" / "images" / "Dataset_pieridae",
]
DEFAULT_OUTPUT_DIR = PROYECTO_DIR / "segmentacion"

METHOD = "grabcut"          # "grabcut" o "otsu"
BORDER_MARGIN_FRAC = 0.04   # % del ancho/alto que se asume fondo seguro (grabcut)
GRABCUT_ITERS = 5
PREVIEW_COUNT = 30          # cuántas imágenes de muestra guardar en preview/
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def keep_largest_component(mask_bin: np.ndarray) -> np.ndarray:
    """Conserva solo el componente conexo más grande de una máscara 0/255."""
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    if n_labels <= 1:
        return mask_bin
    # label 0 es el fondo; buscamos el label (>0) con mayor área
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))
    out = np.zeros_like(mask_bin)
    out[labels == largest_label] = 255
    return out


def clean_mask(mask_bin: np.ndarray) -> np.ndarray:
    """Limpieza morfológica: cierra huecos pequeños, quita ruido, conserva
    el objeto más grande y rellena huecos internos."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel)
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel)
    mask_bin = keep_largest_component(mask_bin)

    # Rellenar huecos internos (ej. el cuerpo separado de las alas por el flash)
    filled = mask_bin.copy()
    h, w = mask_bin.shape
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(filled, flood_mask, (0, 0), 255)
    filled_inv = cv2.bitwise_not(filled)
    mask_bin = mask_bin | filled_inv
    return mask_bin


def segment_grabcut(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    mx = max(1, int(w * BORDER_MARGIN_FRAC))
    my = max(1, int(h * BORDER_MARGIN_FRAC))
    rect = (mx, my, w - 2 * mx, h - 2 * my)

    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(img_bgr, mask, rect, bgd_model, fgd_model, GRABCUT_ITERS, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        # Imagen demasiado pequeña/uniforme para grabCut -> fallback a Otsu
        return segment_otsu(img_bgr)

    mask_bin = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return clean_mask(mask_bin)


def segment_otsu(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask_bin = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Otsu no sabe cuál de los dos grupos es el fondo: asumimos que el fondo
    # es el que predomina en el borde de la imagen, e invertimos si hace falta.
    h, w = mask_bin.shape
    border = np.concatenate([
        mask_bin[0, :], mask_bin[-1, :], mask_bin[:, 0], mask_bin[:, -1]
    ])
    if np.mean(border) > 127:  # el borde quedó "blanco" -> el fondo es el blanco
        mask_bin = cv2.bitwise_not(mask_bin)

    return clean_mask(mask_bin)


def make_preview(img_bgr, mask_bin, cutout_bgr):
    h, w = img_bgr.shape[:2]
    mask_bgr = cv2.cvtColor(mask_bin, cv2.COLOR_GRAY2BGR)
    collage = np.hstack([img_bgr, mask_bgr, cutout_bgr])
    max_w = 1400
    if collage.shape[1] > max_w:
        scale = max_w / collage.shape[1]
        collage = cv2.resize(collage, None, fx=scale, fy=scale)
    return collage


def dataset_name_from_path(images_dir: Path) -> str:
  
    if images_dir.name.lower() == "images":
        return images_dir.parent.name
    return images_dir.name


def segment_folder(images_dir: Path, output_dir: Path, method: str = METHOD):
    if not images_dir.exists():
        print(f"[ERROR] No se encontró la carpeta de imágenes: {images_dir}")
        return False

    masks_dir = output_dir / "masks"
    segmented_dir = output_dir / "segmented"
    preview_dir = output_dir / "preview"
    for d in (masks_dir, segmented_dir, preview_dir):
        d.mkdir(parents=True, exist_ok=True)

    segment_fn = segment_grabcut if method == "grabcut" else segment_otsu

    files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in VALID_EXTS)
    print(f"Procesando {len(files)} imágenes con método '{method}' ...")

    failed = []
    for i, path in enumerate(files):
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            failed.append(path.name)
            continue

        mask_bin = segment_fn(img_bgr)

        # Recorte: mariposa sobre fondo blanco
        cutout = np.full_like(img_bgr, 255)
        cutout[mask_bin == 255] = img_bgr[mask_bin == 255]

        out_name = path.stem + ".png"
        cv2.imwrite(str(masks_dir / out_name), mask_bin)
        cv2.imwrite(str(segmented_dir / out_name), cutout)

        if i < PREVIEW_COUNT:
            collage = make_preview(img_bgr, mask_bin, cutout)
            cv2.imwrite(str(preview_dir / out_name), collage)

        if (i + 1) % 25 == 0 or (i + 1) == len(files):
            print(f"  {i + 1}/{len(files)}")

    print(f"\nListo. Máscaras en: {masks_dir}")
    print(f"Recortes en:        {segmented_dir}")
    print(f"Preview (primeras {PREVIEW_COUNT}) en: {preview_dir}")
    if failed:
        print(f"\n[AVISO] {len(failed)} imágenes no se pudieron leer:")
        for name in failed[:20]:
            print(f"   - {name}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", nargs="+", default=[str(p) for p in DEFAULT_IMAGES_DIRS],
                         help="Una o varias carpetas con las imágenes originales")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR), help="Carpeta de salida raíz")
    parser.add_argument("--method", default=METHOD, choices=["grabcut", "otsu"], help="Método de segmentación")
    args = parser.parse_args()

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    for images_arg in args.images:
        images_dir = Path(images_arg)
        name = dataset_name_from_path(images_dir)
        print(f"\n=== Dataset: {name} ({images_dir}) ===")
        segment_folder(images_dir, output_root / name, method=args.method)

    print(f"\nTodo listo. Resultados en: {output_root.resolve()}")


if __name__ == "__main__":
    main()