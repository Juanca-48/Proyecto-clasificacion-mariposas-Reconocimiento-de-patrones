#!/usr/bin/env python3
"""
exportar_app.py
---------------
Copia a app/ lo que necesita el Space de Hugging Face para funcionar por sí solo: los módulos de inferencia (los mismos
que usa el entrenamiento, para que la app calcule exactamente las mismas features) y los modelos. app/app.py,
app/requirements.txt y app/README.md se editan directamente en app/.

  scripts/{rutas,Segmentacion,Segmentacion_rembg,texturas,color_features,inferencia,
           embeddings,inferencia_embeddings}.py                          ->  app/
  resultados/final/final_model.joblib                                    ->  app/modelo/   (modelo clásico)
  resultados/rechazo/rechazo.joblib                                      ->  app/modelo/
  resultados/modelo_embeddings/modelo.joblib (si existe)                 ->  app/modelo/modelo_embeddings.joblib

La app ofrece dos modelos: el clásico (predeterminado) y, si existe el archivo de embeddings, un selector para usarlo.
Con --sin-clasico no se reemplaza el modelo clásico de app/modelo (útil mientras se reentrena o si quieres conservar el
que ya está subido). Al final se comprueba que app/ funciona aislada: un proceso que no ve scripts/ carga los modelos y
clasifica una imagen sintética.

Uso:
    python scripts/despliegue/exportar_app.py
    python scripts/despliegue/exportar_app.py --sin-clasico
"""
from __future__ import annotations

# Localiza rutas.py (en scripts/, o junto a este archivo en la app) y agrega al path la carpeta de cada proceso
import sys
from pathlib import Path
sys.path[:0] = [str(p) for p in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]) if (p / "rutas.py").exists()][:1]
import rutas  # noqa: E402,F401


import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from rutas import APP_DIR, FINAL_DIR, REJECT_DIR, RESULTS_DIR

SCRIPTS = Path(__file__).resolve().parents[1]


def origen(modulo):
    """Ruta del módulo dentro de las carpetas de proceso de scripts/."""
    return next(SCRIPTS.rglob(modulo), SCRIPTS / modulo)
MODULOS = ["rutas.py", "Segmentacion.py", "Segmentacion_rembg.py", "texturas.py", "color_features.py", "inferencia.py",
           "embeddings.py", "inferencia_embeddings.py"]
CLASICO = [FINAL_DIR / "final_model.joblib", REJECT_DIR / "rechazo.joblib"]
EMBEDDINGS = RESULTS_DIR / "modelo_embeddings" / "modelo.joblib"

PRUEBA = """
import sys
sys.path[:] = [p for p in sys.path if 'scripts' not in p]
from pathlib import Path
import numpy as np
from inferencia import Clasificador
img = np.full((200, 300, 3), 128, np.uint8)
print('clásico, imagen gris:', Clasificador(Path('modelo/final_model.joblib'), Path('modelo/rechazo.joblib')).predecir(img).estado)
ruta = Path('modelo/modelo_embeddings.joblib')
if ruta.exists():
    from inferencia_embeddings import ClasificadorEmbeddings
    m = ClasificadorEmbeddings(ruta)
    print('embeddings (' + m.descripcion + '), imagen gris:', m.predecir(img).estado)
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sin-clasico", action="store_true", help="No reemplazar el modelo clásico de app/modelo")
    args = ap.parse_args()

    requeridos = [origen(m) for m in MODULOS] + ([] if args.sin_clasico else CLASICO)
    faltan = [p for p in requeridos if not p.exists()]
    if faltan:
        print("[ERROR] Faltan archivos:", *[str(p) for p in faltan], sep="\n  ")
        print("Genera los modelos con entrenar.py y la calibración con calibrar_rechazo.py.")
        sys.exit(1)
    (APP_DIR / "modelo").mkdir(parents=True, exist_ok=True)
    for m in MODULOS:
        shutil.copy2(origen(m), APP_DIR / m)
    copiados = [] if args.sin_clasico else CLASICO
    for p in copiados:
        shutil.copy2(p, APP_DIR / "modelo" / p.name)
    if EMBEDDINGS.exists():
        shutil.copy2(EMBEDDINGS, APP_DIR / "modelo" / "modelo_embeddings.joblib")
    print(f"Copiados {len(MODULOS)} módulos y modelos: clásico {'(conservado)' if args.sin_clasico else '(actualizado)'}, "
          f"embeddings {'(incluido)' if EMBEDDINGS.exists() else '(no existe, la app solo ofrecerá el clásico)'} -> {APP_DIR}")

    # la app debe funcionar sola: proceso nuevo, sin scripts/ en el path, con carpeta de trabajo en app/
    res = subprocess.run([sys.executable, "-c", PRUEBA], cwd=APP_DIR, capture_output=True, text=True)
    print(res.stdout.strip() or res.stderr.strip()[-800:])
    if res.returncode != 0:
        print("[ERROR] app/ no funciona de forma aislada.")
        sys.exit(1)
    print("app/ funciona de forma aislada.")


if __name__ == "__main__":
    main()
