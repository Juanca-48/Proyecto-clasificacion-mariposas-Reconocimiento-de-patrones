import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# ----------------------------------------------------------------------
# CONFIG (usado si no se pasan argumentos por línea de comandos)
# ----------------------------------------------------------------------
REPO_DIR = Path(r"D:\Trabajos\Universidad\patrones\Proyecto_mariposas\Git_mariposas\Proyecto-clasificacion-mariposas-Reconocimiento-de-patrones")

DEFAULT_DATASETS = [
    (REPO_DIR / "Papilionidae", "Dataset_papilionidae_v2"),
    (REPO_DIR / "Pieridae", "Dataset_pieridae_v2"),
]
DEFAULT_OUTPUT_DIR = REPO_DIR / "segmentacion"

METHOD = "grabcut"          # "grabcut" o "otsu"
BORDER_MARGIN_FRAC = 0.04   # % del ancho/alto que se asume fondo seguro
GRABCUT_ITERS = 5
PREVIEW_COUNT = 30          # cuántas imágenes de muestra guardar en preview/
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# --- Filtro de "objetos intrusos" (dedos, flores) usado como SEMILLA ---
ENABLE_INTRUDER_REFINEMENT = True
INTRUDER_MIN_AREA_FRAC = 0.005    # un blob debe cubrir al menos este % del área para considerarse intruso
CORE_ERODE_FRAC = 0.03            # qué tanto se erosiona el "núcleo seguro" de mariposa (ver docstring)
FLOWER_HUE_RANGES = ((125, 179), (0, 10))  # rangos de matiz (H, 0-180) considerados "color de flor"
FLOWER_SAT_THRESH = 90


def keep_largest_component(mask_bin: np.ndarray) -> np.ndarray:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    if n_labels <= 1:
        return mask_bin
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))
    out = np.zeros_like(mask_bin)
    out[labels == largest_label] = 255
    return out


def keep_main_object(mask_bin: np.ndarray, reconnect_frac: float = 0.05) -> np.ndarray:
    """Como keep_largest_component, pero primero 'dilata' la máscara para
    reconectar piezas cercanas que quedaron separadas al quitar un
    intruso (ej. las dos alas, separadas justo donde estaba el dedo).
    Así no se pierde un ala completa por quedarse solo con la pieza más
    grande de las dos."""
    h, w = mask_bin.shape
    gap = max(3, int(reconnect_frac * max(h, w)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap, gap))
    dilated = cv2.dilate(mask_bin, kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    if n_labels <= 1:
        return mask_bin
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))
    roi = (labels == largest_label).astype(np.uint8) * 255
    return cv2.bitwise_and(mask_bin, roi)


def fill_holes(mask_bin: np.ndarray) -> np.ndarray:
    h, w = mask_bin.shape
    filled = mask_bin.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(filled, flood_mask, (0, 0), 255)
    return mask_bin | cv2.bitwise_not(filled)


def clean_mask(mask_bin: np.ndarray) -> np.ndarray:
    """Limpieza morfológica + reconexión de piezas + relleno de huecos."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel)
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel)
    mask_bin = keep_main_object(mask_bin)
    return fill_holes(mask_bin)


# ----------------------------------------------------------------------
# Detección de "intrusos": piel (dedos/mano) y colores de flor
# ----------------------------------------------------------------------

def skin_mask(img_bgr: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    return cv2.inRange(ycrcb, lower, upper)


def flower_color_mask(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, _ = cv2.split(hsv)
    sat_ok = s > FLOWER_SAT_THRESH
    hue_ok = np.zeros_like(h, dtype=bool)
    for lo, hi in FLOWER_HUE_RANGES:
        hue_ok |= (h >= lo) & (h <= hi)
    return (sat_ok & hue_ok).astype(np.uint8) * 255


def large_blobs(mask: np.ndarray, min_area_frac: float) -> np.ndarray:
    h, w = mask.shape
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    min_area = min_area_frac * h * w
    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
            out[labels == lbl] = 255
    return out


# ----------------------------------------------------------------------
# GrabCut en dos pasadas
# ----------------------------------------------------------------------

def segment_grabcut_pass1(img_bgr: np.ndarray) -> np.ndarray:
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
        return segment_otsu(img_bgr)

    mask_bin = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return clean_mask(mask_bin)


def segment_grabcut_pass2_refine(img_bgr: np.ndarray, mask1: np.ndarray, intruder_mask: np.ndarray) -> np.ndarray:
    """Segunda pasada de GrabCut usando el intruso como semilla de fondo
    PROBABLE (no absoluto) y un núcleo interior erosionado como semilla
    de mariposa SEGURA. Ver docstring del módulo para la justificación."""
    h, w = img_bgr.shape[:2]

    gc_mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    gc_mask[mask1 == 255] = cv2.GC_PR_FGD

    # el intruso, donde se solape con la máscara, se degrada a fondo
    # PROBABLE -- GrabCut decide con el contexto completo, no a ciegas
    downgrade = (intruder_mask == 255) & (mask1 == 255)
    gc_mask[downgrade] = cv2.GC_PR_BGD

    # núcleo seguro: erosionar la parte de la máscara que NO es intruso
    core_erode = max(5, int(CORE_ERODE_FRAC * min(h, w)))
    safe_source = np.where((mask1 == 255) & (intruder_mask == 0), 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (core_erode, core_erode))
    core = cv2.erode(safe_source, kernel)
    gc_mask[core == 255] = cv2.GC_FGD

    # borde de la imagen: fondo seguro
    mx = max(1, int(w * BORDER_MARGIN_FRAC))
    my = max(1, int(h * BORDER_MARGIN_FRAC))
    gc_mask[:my, :] = cv2.GC_BGD
    gc_mask[-my:, :] = cv2.GC_BGD
    gc_mask[:, :mx] = cv2.GC_BGD
    gc_mask[:, -mx:] = cv2.GC_BGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img_bgr, gc_mask, None, bgd_model, fgd_model, GRABCUT_ITERS, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return mask1  # si algo sale mal, nos quedamos con el resultado de la pasada 1

    mask_bin = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return clean_mask(mask_bin)


def segment_grabcut(img_bgr: np.ndarray) -> np.ndarray:
    mask1 = segment_grabcut_pass1(img_bgr)
    if not ENABLE_INTRUDER_REFINEMENT:
        return mask1

    intruder = cv2.bitwise_or(skin_mask(img_bgr), flower_color_mask(img_bgr))
    intruder = large_blobs(intruder, INTRUDER_MIN_AREA_FRAC)
    if not intruder.any():
        return mask1  # nada sospechoso, no hace falta la segunda pasada

    return segment_grabcut_pass2_refine(img_bgr, mask1, intruder)


def segment_otsu(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask_bin = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = np.concatenate([mask_bin[0, :], mask_bin[-1, :], mask_bin[:, 0], mask_bin[:, -1]])
    if np.mean(border) > 127:
        mask_bin = cv2.bitwise_not(mask_bin)
    return clean_mask(mask_bin)


def make_preview(img_bgr, mask_bin, cutout_bgr):
    mask_bgr = cv2.cvtColor(mask_bin, cv2.COLOR_GRAY2BGR)
    collage = np.hstack([img_bgr, mask_bgr, cutout_bgr])
    max_w = 1400
    if collage.shape[1] > max_w:
        scale = max_w / collage.shape[1]
        collage = cv2.resize(collage, None, fx=scale, fy=scale)
    return collage


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
    parser.add_argument("--images", nargs="+", default=None,
                         help="Una o varias carpetas con las imágenes originales (si se omite, usa DEFAULT_DATASETS)")
    parser.add_argument("--names", nargs="+", default=None,
                         help="Nombres de salida correspondientes a --images, en el mismo orden")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR), help="Carpeta de salida raíz")
    parser.add_argument("--method", default=METHOD, choices=["grabcut", "otsu"], help="Método de segmentación")
    args = parser.parse_args()

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.images:
        names = args.names if args.names else [Path(p).parent.name for p in args.images]
        datasets = list(zip([Path(p) for p in args.images], names))
    else:
        datasets = DEFAULT_DATASETS

    for images_dir, out_name in datasets:
        print(f"\n=== Dataset: {out_name} ({images_dir}) ===")
        segment_folder(images_dir, output_root / out_name, method=args.method)

    print(f"\nTodo listo. Resultados en: {output_root.resolve()}")


if __name__ == "__main__":
    main()