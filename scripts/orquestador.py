"""
Motor común de preprocesar.py y entrenar.py: ejecuta una lista de etapas en orden, cada una en
un proceso aparte, con selección de etapas, validación de entradas e instalación de dependencias.

Una etapa es una tupla (nombre, script, argumentos, datos que necesita, datos que produce). Con script None la etapa
está reservada pero todavía sin implementar: se muestra como pendiente y se omite sin fallar.
Sin argumentos, cada script usa sus rutas por defecto.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

from rutas import REPO_DIR

SCRIPTS = Path(__file__).resolve().parent


def asegurar_dependencias(requisitos: dict, dry_run: bool = False):
    """Instala con pip lo que falte (nombre de importación -> paquete de pip). Se detiene si no se puede."""
    faltan = [pip for mod, pip in requisitos.items() if importlib.util.find_spec(mod) is None]
    if not faltan:
        print("Dependencias: todas instaladas.")
        return
    print(f"Dependencias que faltan: {', '.join(faltan)}")
    if dry_run:
        print(f"  (dry-run) se instalarían con: {sys.executable} -m pip install {' '.join(faltan)}")
        return
    if subprocess.run([sys.executable, "-m", "pip", "install", *faltan]).returncode != 0:
        print(f"\n[ERROR] No se pudieron instalar: {', '.join(faltan)}.")
        print(f"Instálalas a mano con:  pip install {' '.join(faltan)}")
        print("(Si mahotas falla al compilar: conda install -c conda-forge mahotas, o pip install mahotas --only-binary :all:)")
        sys.exit(1)
    importlib.invalidate_caches()
    print("Dependencias instaladas.")


def seleccionar(stages: list, solo=None, omitir=None, desde=None, hasta=None) -> list:
    """Etapas a ejecutar, siempre en el orden del pipeline. --solo no se combina con el resto."""
    names = [s[0] for s in stages]
    if solo:
        if omitir or desde or hasta:
            raise ValueError("--solo no se combina con --desde, --hasta ni --omitir.")
        chosen = set(solo)
    else:
        i, j = names.index(desde or names[0]), names.index(hasta or names[-1])
        if i > j:
            raise ValueError(f"--desde ({names[i]}) va después de --hasta ({names[j]}). Orden: {' -> '.join(names)}")
        chosen = set(names[i:j + 1]) - set(omitir or [])
    if not chosen:
        raise ValueError("No queda ninguna etapa por ejecutar.")
    return [s for s in stages if s[0] in chosen]


def _rel(p: Path) -> str:
    return str(p.relative_to(REPO_DIR)) if p.is_relative_to(REPO_DIR) else str(p)


def validar_entradas(a_ejecutar: list, todas: list):
    """Cada etapa necesita sus datos en disco o producidos por una etapa anterior de esta corrida."""
    producido = set()
    for name, _, _, necesita, produce in a_ejecutar:
        faltan = [p for p in necesita if not p.exists() and p not in producido]
        if faltan:
            origen = [s[0] for s in todas if any(p in s[4] for p in faltan)]
            raise ValueError(f"La etapa '{name}' necesita datos que no existen: "
                             + ", ".join(_rel(p) for p in faltan)
                             + f". Los genera la etapa: {', '.join(dict.fromkeys(origen)) or 'ninguna de este script'}"
                             + "; inclúyela con --desde o --solo.")
        producido.update(produce)


def mostrar_estado(stages: list):
    print("Etapa               Estado        Resultado")
    for name, script, _, _, produce in stages:
        if script is None:
            print(f"{name:<19} [-] sin implementar")
            continue
        hecha = all(p.exists() for p in produce)
        print(f"{name:<19} {'[x] hecha    ' if hecha else '[ ] pendiente'} {_rel(produce[-1])}")


def correr(doc: str, stages: list, requisitos: dict, prog: str, cierre: str = ""):
    names = [s[0] for s in stages]
    ap = argparse.ArgumentParser(prog=prog, description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--desde", choices=names, help="Primera etapa a ejecutar (por defecto, la primera)")
    ap.add_argument("--hasta", choices=names, help="Última etapa a ejecutar (por defecto, la última)")
    ap.add_argument("--solo", nargs="+", choices=names, metavar="ETAPA", help="Ejecutar solo estas etapas")
    ap.add_argument("--omitir", nargs="+", choices=names, metavar="ETAPA", help="No ejecutar estas etapas")
    ap.add_argument("--estado", action="store_true", help="Mostrar qué etapas ya tienen sus resultados y salir")
    ap.add_argument("--dry-run", action="store_true", help="Mostrar los comandos sin ejecutarlos")
    ap.add_argument("--selftest", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if args.estado:
        mostrar_estado(stages)
        return
    try:
        a_ejecutar = seleccionar(stages, args.solo, args.omitir, args.desde, args.hasta)
        validar_entradas(a_ejecutar, stages)
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    asegurar_dependencias(requisitos, args.dry_run)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}   # los scripts imprimen tildes
    t0 = time.time()
    for n, (name, script, extra, _, _) in enumerate(a_ejecutar, 1):
        if script is None:
            print(f"\n[{n}/{len(a_ejecutar)}] {name}: pendiente (todavía no está implementada), se omite")
            continue
        cmd = [sys.executable, str(SCRIPTS / script), *extra]
        print(f"\n{'=' * 70}\n[{n}/{len(a_ejecutar)}] {name}: {script} {' '.join(extra)}\n{'=' * 70}", flush=True)
        if args.dry_run:
            print("  (dry-run) " + " ".join(cmd))
            continue
        t = time.time()
        if subprocess.run(cmd, cwd=REPO_DIR, env=env).returncode != 0:
            skipped = [s[0] for s in a_ejecutar[n:]]
            print(f"\n[ERROR] Falló la etapa '{name}'. No se ejecutaron: {skipped or 'ninguna más'}.")
            print(f"Corrige el problema y reanuda con:  python scripts/{prog} --desde {name}")
            sys.exit(1)
        print(f"-> '{name}' terminada en {time.time() - t:.0f} s", flush=True)

    if not args.dry_run:
        print(f"\nTerminado en {time.time() - t0:.0f} s.")
        if cierre:
            print(cierre)


def selftest():
    mk = lambda n, nec, prod: (n, n + ".py", [], nec, prod)
    f, g = Path("/no/existe/f"), Path("/no/existe/g")
    st = [mk("a", [], [f]), mk("b", [f], [g]), mk("c", [g], []), mk("d", [], [])]
    nombres = lambda **k: [s[0] for s in seleccionar(st, **k)]
    assert nombres() == list("abcd")
    assert nombres(desde="b") == list("bcd")
    assert nombres(desde="b", hasta="c") == ["b", "c"]
    assert nombres(solo=["c", "a"]) == ["a", "c"], "debe respetar el orden del pipeline"
    assert nombres(omitir=["a", "b"]) == ["c", "d"]
    for malo in (dict(desde="c", hasta="a"), dict(solo=["a"], omitir=["b"]), dict(omitir=list("abcd"))):
        try:
            seleccionar(st, **malo)
        except ValueError:
            continue
        raise AssertionError(f"debía fallar: {malo}")
    assert [s[0] for s in seleccionar(st + [("p", None, [], [], [])], solo=["p"])] == ["p"]   # etapa pendiente
    validar_entradas(st[:2], st)                 # a produce lo que b necesita
    try:
        validar_entradas(st[1:2], st)            # b sin a: falta
    except ValueError as e:
        assert "necesita datos que no existen" in str(e) and "Los genera la etapa: a" in str(e), e
        print("selftest OK")
        return
    raise AssertionError("debía detectar la entrada faltante")
