#!/usr/bin/env python3
"""
exportar_app.py
---------------
Copia a app/ lo que necesita el Space de Hugging Face para funcionar por sí solo: los módulos de
inferencia (los mismos que usa el entrenamiento, para que la app calcule exactamente las mismas
features) y los modelos. app/app.py, app/requirements.txt y app/README.md se editan directamente en app/.

  scripts/{rutas,Segmentacion,texturas,color_features,inferencia}.py  ->  app/
  resultados/final/final_model.joblib                                  ->  app/modelo/
  resultados/rechazo/rechazo.joblib                                    ->  app/modelo/

Ejecútalo después de reentrenar el modelo o de calibrar el rechazo. Al final comprueba que app/ funciona
aislada: importa el clasificador desde un proceso que no ve scripts/ y clasifica una imagen sintética.

Uso:
    python scripts/exportar_app.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from rutas import APP_DIR, FINAL_DIR, REJECT_DIR

SCRIPTS = Path(__file__).resolve().parent
MODULOS = ["rutas.py", "Segmentacion.py", "texturas.py", "color_features.py", "inferencia.py"]
MODELOS = [FINAL_DIR / "final_model.joblib", REJECT_DIR / "rechazo.joblib"]


def main():
    faltan = [p for p in [SCRIPTS / m for m in MODULOS] + MODELOS if not p.exists()]
    if faltan:
        print("[ERROR] Faltan archivos:", *[str(p) for p in faltan], sep="\n  ")
        print("Genera los modelos con entrenar.py y la calibración con calibrar_rechazo.py.")
        sys.exit(1)
    (APP_DIR / "modelo").mkdir(parents=True, exist_ok=True)
    for m in MODULOS:
        shutil.copy2(SCRIPTS / m, APP_DIR / m)
    for p in MODELOS:
        shutil.copy2(p, APP_DIR / "modelo" / p.name)
    print(f"Copiados {len(MODULOS)} módulos y {len(MODELOS)} modelos a {APP_DIR}")

    # la app debe funcionar sola: proceso nuevo, sin scripts/ en el path, con carpeta de trabajo en app/
    prueba = (
        "import sys, numpy as np; sys.path[:] = [p for p in sys.path if 'scripts' not in p];"
        "from pathlib import Path; from inferencia import Clasificador;"
        "c = Clasificador(Path('modelo/final_model.joblib'), Path('modelo/rechazo.joblib'));"
        "r = c.predecir(np.full((200, 300, 3), 128, np.uint8)); print('estado con una imagen gris:', r.estado)"
    )
    res = subprocess.run([sys.executable, "-c", prueba], cwd=APP_DIR, capture_output=True, text=True)
    print(res.stdout.strip() or res.stderr.strip()[-600:])
    if res.returncode != 0:
        print("[ERROR] app/ no funciona de forma aislada.")
        sys.exit(1)
    print("app/ funciona de forma aislada.")


if __name__ == "__main__":
    main()
