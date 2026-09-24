from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# CONFIG (usado si no se pasan argumentos por línea de comandos)
# ----------------------------------------------------------------------
REPO_DIR = Path(r"D:\Trabajos\Universidad\patrones\Proyecto_mariposas\Git_mariposas\Proyecto-clasificacion-mariposas-Reconocimiento-de-patrones")
SEGMENTATION_ROOT = REPO_DIR / "segmentacion"

DEFAULT_TEXTURE_CSV = SEGMENTATION_ROOT / "texture_features_all.csv"
DEFAULT_COLOR_CSV = SEGMENTATION_ROOT / "color_features_all.csv"
DEFAULT_LABELS_CSVS: list = []   # opcional: lista de CSV (filename,label) si el label no quedó ya en los CSV de arriba
DEFAULT_OUTPUT_DIR = SEGMENTATION_ROOT

MIN_SAMPLES_PER_CLASS = 5     # clases con menos ejemplos que esto se marcan como "insuficientes"
MIN_FG_PIXELS = 500           # filas con menos píxeles de mariposa que esto se marcan como baja calidad
MAX_NAN_FRAC_COL = 0.2        # columnas con más de este % de NaN se descartan directamente
NEAR_CONSTANT_STD = 1e-8      # columnas con desviación estándar menor a esto se consideran "sin información"

NON_FEATURE_COLS = {"filename", "dataset", "label", "genus", "fg_pixels", "low_quality"}


# ----------------------------------------------------------------------
# Carga y unión
# ----------------------------------------------------------------------

def load_labels(paths: list) -> pd.DataFrame:
    frames = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            print(f"[AVISO] No se encontró el CSV de labels: {p} (se ignora)")
            continue
        df = pd.read_csv(p)
        # admite distintos nombres de columna habituales
        cols_lower = {c.lower(): c for c in df.columns}
        fname_col = cols_lower.get("filename") or cols_lower.get("file") or cols_lower.get("image") or df.columns[0]
        label_col = cols_lower.get("label") or cols_lower.get("species") or cols_lower.get("class") or df.columns[1]
        frames.append(df[[fname_col, label_col]].rename(columns={fname_col: "filename", label_col: "label"}))
    if not frames:
        return pd.DataFrame(columns=["filename", "label"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset="filename")


def merge_features(texture_csv: Path, color_csv: Path, labels_csvs: list) -> pd.DataFrame:
    if not texture_csv.exists():
        print(f"[ERROR] No se encontró el CSV de texturas: {texture_csv}")
        sys.exit(1)
    if not color_csv.exists():
        print(f"[ERROR] No se encontró el CSV de color: {color_csv}")
        sys.exit(1)

    tex = pd.read_csv(texture_csv)
    col = pd.read_csv(color_csv)
    print(f"Textura: {tex.shape[0]} filas, {tex.shape[1]} columnas ({texture_csv.name})")
    print(f"Color:   {col.shape[0]} filas, {col.shape[1]} columnas ({color_csv.name})")

    join_keys = ["filename", "dataset"] if "dataset" in tex.columns and "dataset" in col.columns else ["filename"]

    # evitar columnas duplicadas (fg_pixels, label si vienen en ambos)
    dup_cols = [c for c in col.columns if c in tex.columns and c not in join_keys]
    col_to_merge = col.drop(columns=dup_cols)

    merged = tex.merge(col_to_merge, on=join_keys, how="inner")
    lost_tex = len(tex) - len(merged)
    lost_col = len(col) - len(merged)
    if lost_tex or lost_col:
        print(f"[AVISO] {lost_tex} filas de textura y {lost_col} de color no encontraron pareja en el otro CSV (se excluyeron del merge).")
    print(f"Unido:   {merged.shape[0]} filas, {merged.shape[1]} columnas")

    if "label" not in merged.columns and labels_csvs:
        labels_df = load_labels(labels_csvs)
        before = len(merged)
        merged = merged.merge(labels_df, on="filename", how="left")
        n_missing = merged["label"].isna().sum()
        print(f"Labels unidos desde {len(labels_csvs)} archivo(s): {before - n_missing}/{before} filas con label encontrado.")
    elif "label" not in merged.columns:
        print("[AVISO] No hay columna 'label' en los datos ni se pasó --labels: el dataset quedará sin etiquetas.")

    return merged


# ----------------------------------------------------------------------
# Auditoría de calidad
# ----------------------------------------------------------------------

def audit_quality(df: pd.DataFrame, min_fg_pixels: int, report_lines: list) -> pd.DataFrame:
    df = df.copy()
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

    nan_counts = df[feature_cols].isna().sum()
    cols_with_nan = nan_counts[nan_counts > 0]
    rows_with_nan = df[feature_cols].isna().any(axis=1)

    low_fg = pd.Series(False, index=df.index)
    if "fg_pixels" in df.columns:
        low_fg = df["fg_pixels"] < min_fg_pixels

    df["low_quality"] = rows_with_nan | low_fg

    report_lines.append("=== AUDITORÍA DE CALIDAD ===")
    report_lines.append(f"Total de filas: {len(df)}")
    report_lines.append(f"Filas con al menos un NaN en features: {rows_with_nan.sum()}")
    if len(cols_with_nan):
        report_lines.append("Columnas con NaN (top 10):")
        for c, n in cols_with_nan.sort_values(ascending=False).head(10).items():
            report_lines.append(f"    {c}: {n} filas")
    if "fg_pixels" in df.columns:
        report_lines.append(f"fg_pixels -> min={df['fg_pixels'].min()}, mediana={df['fg_pixels'].median():.0f}, max={df['fg_pixels'].max()}")
        report_lines.append(f"Filas con fg_pixels < {min_fg_pixels} (posible segmentación fallida): {low_fg.sum()}")
    report_lines.append(f"TOTAL marcadas como low_quality (NaN o fg_pixels bajo): {df['low_quality'].sum()} / {len(df)}")

    if df["low_quality"].sum() > 0:
        sample = df.loc[df["low_quality"], ["filename"] + (["dataset"] if "dataset" in df.columns else [])].head(15)
        report_lines.append("Ejemplos (hasta 15):")
        for _, row in sample.iterrows():
            report_lines.append(f"    {row.to_dict()}")

    return df


# ----------------------------------------------------------------------
# Distribución de clases
# ----------------------------------------------------------------------

def derive_genus(label: str) -> str:
    if not isinstance(label, str) or not label.strip():
        return ""
    return re.split(r"[ _]+", label.strip())[0]


def report_class_distribution(df: pd.DataFrame, min_samples: int, report_lines: list, output_dir: Path):
    if "label" not in df.columns or df["label"].isna().all():
        report_lines.append("\n[AVISO] No hay labels -- se omite el reporte de distribución de clases.")
        return

    df["genus"] = df["label"].apply(derive_genus)

    for level in ["label", "genus"]:
        counts = df[level].value_counts().sort_values(ascending=False)
        n_classes = len(counts)
        n_insufficient = (counts < min_samples).sum()
        report_lines.append(f"\n=== DISTRIBUCIÓN POR {level.upper()} ===")
        report_lines.append(f"Número de clases: {n_classes}")
        report_lines.append(f"Clases con menos de {min_samples} ejemplos: {n_insufficient} ({100*n_insufficient/max(n_classes,1):.0f}%)")
        report_lines.append(f"Mediana de ejemplos por clase: {counts.median():.0f}  |  min: {counts.min()}  |  max: {counts.max()}")
        report_lines.append(f"Top 10 clases más frecuentes:\n{counts.head(10).to_string()}")
        if n_insufficient:
            report_lines.append(f"Clases insuficientes (< {min_samples}):\n{counts[counts < min_samples].to_string()}")

    # Gráfico simple de barras (por especie, y por género) para revisión visual
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        for ax, level in zip(axes, ["label", "genus"]):
            counts = df[level].value_counts().sort_values(ascending=False)
            ax.bar(range(len(counts)), counts.values)
            ax.axhline(min_samples, color="red", linestyle="--", linewidth=1, label=f"mínimo sugerido ({min_samples})")
            ax.set_title(f"Distribución de clases por {level} ({len(counts)} clases)")
            ax.set_ylabel("N.º de imágenes")
            ax.set_xticks([])
            ax.legend()
        fig.tight_layout()
        out_path = output_dir / "class_distribution.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        report_lines.append(f"\nGráfico guardado en: {out_path}")
    except Exception as e:
        report_lines.append(f"\n[AVISO] No se pudo generar el gráfico de distribución: {e}")


# ----------------------------------------------------------------------
# Limpieza de la matriz de features
# ----------------------------------------------------------------------

def clean_feature_matrix(df: pd.DataFrame, report_lines: list) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

    nan_frac = df[feature_cols].isna().mean()
    drop_nan_cols = nan_frac[nan_frac > MAX_NAN_FRAC_COL].index.tolist()

    stds = df[feature_cols].std(numeric_only=True)
    drop_const_cols = stds[stds < NEAR_CONSTANT_STD].index.tolist()

    drop_cols = sorted(set(drop_nan_cols) | set(drop_const_cols))
    report_lines.append("\n=== LIMPIEZA DE FEATURES ===")
    report_lines.append(f"Columnas de features totales: {len(feature_cols)}")
    if drop_nan_cols:
        report_lines.append(f"Descartadas por > {MAX_NAN_FRAC_COL*100:.0f}% de NaN: {drop_nan_cols}")
    if drop_const_cols:
        report_lines.append(f"Descartadas por ser casi constantes (sin información): {drop_const_cols}")
    report_lines.append(f"Columnas de features restantes: {len(feature_cols) - len(drop_cols)}")

    return df.drop(columns=drop_cols)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--texture-csv", default=str(DEFAULT_TEXTURE_CSV))
    parser.add_argument("--color-csv", default=str(DEFAULT_COLOR_CSV))
    parser.add_argument("--labels", nargs="*", default=DEFAULT_LABELS_CSVS,
                         help="CSV(s) opcionales (filename,label) si el label no está ya en los features")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR), help="Carpeta donde guardar los resultados")
    parser.add_argument("--min-samples", type=int, default=MIN_SAMPLES_PER_CLASS,
                         help="Mínimo de ejemplos por clase para considerarla 'suficiente'")
    parser.add_argument("--min-fg-pixels", type=int, default=MIN_FG_PIXELS,
                         help="Mínimo de píxeles de mariposa detectados para considerar la segmentación válida")
    parser.add_argument("--drop-low-quality", action="store_true",
                         help="Si se indica, elimina del CSV final las filas marcadas como low_quality (por defecto solo se marcan)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_lines = []

    df = merge_features(Path(args.texture_csv), Path(args.color_csv), args.labels)
    df = audit_quality(df, args.min_fg_pixels, report_lines)
    report_class_distribution(df, args.min_samples, report_lines, output_dir)
    df = clean_feature_matrix(df, report_lines)

    if args.drop_low_quality:
        before = len(df)
        df = df[~df["low_quality"]].drop(columns=["low_quality"])
        report_lines.append(f"\n--drop-low-quality: se eliminaron {before - len(df)} filas. Quedan {len(df)}.")

    out_csv = output_dir / "features_dataset_ready.csv"
    df.to_csv(out_csv, index=False)
    report_lines.append(f"\nDataset final guardado en: {out_csv}  ({df.shape[0]} filas, {df.shape[1]} columnas)")

    report_text = "\n".join(str(l) for l in report_lines)
    report_path = output_dir / "quality_report.txt"
    report_path.write_text(report_text, encoding="utf-8")

    print("\n" + report_text)
    print(f"\nReporte guardado en: {report_path}")


if __name__ == "__main__":
    main()