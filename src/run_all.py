#!/usr/bin/env python3
"""
run_all.py
──────────
Esegue l'intera pipeline in ordine:
  02a  →  diagnostics prezzi
  02b  →  diagnostics margini
  02c  →  change point detection margini

Uso:
  python3 run_all.py           # esegue tutto
  python3 run_all.py 02b 02c  # esegue solo quelli indicati
"""

import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent

PIPELINE = [
    ("02a", "02a_diagnostics_price.py",       "Diagnostics — Prezzi"),
    ("02b", "02b_diagnostics_margin.py",      "Diagnostics — Margini"),
    ("02c", "02c_change_point_detection.py",  "Change Point Detection — Margini"),
    ("02d", "02d_counterfactual_gains.py",    "Analisi Controfattuale — Guadagni Extra"),
]

# ── Filtra per argomenti CLI (es. python run_all.py 02b 02c) ──────────────────
targets = sys.argv[1:]
steps = [(k, f, label) for k, f, label in PIPELINE
         if not targets or any(t in k for t in targets)]

if not steps:
    print(f"⚠  Nessuno step corrisponde a: {targets}")
    sys.exit(1)

# ── Esecuzione ────────────────────────────────────────────────────────────────
total_start = time.time()
results = []

SEP = "═" * 70

for key, filename, label in steps:
    script = BASE_DIR / filename
    if not script.exists():
        print(f"\n{SEP}")
        print(f"  ⚠  {filename} non trovato — salto.")
        results.append((key, label, "SKIP", 0))
        continue

    print(f"\n{SEP}")
    print(f"  ▶  {key}  |  {label}")
    print(f"{SEP}\n")

    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(BASE_DIR),
    )
    elapsed = time.time() - t0

    if proc.returncode == 0:
        status = "OK"
        print(f"\n  ✓  {key} completato in {elapsed:.1f}s")
    else:
        status = "FAIL"
        print(f"\n  ✗  {key} terminato con errore (exit {proc.returncode}) "
              f"dopo {elapsed:.1f}s")

    results.append((key, label, status, elapsed))

# ── Riepilogo finale ──────────────────────────────────────────────────────────
total_elapsed = time.time() - total_start
print(f"\n{SEP}")
print("  RIEPILOGO PIPELINE")
print(f"{SEP}")
for key, label, status, elapsed in results:
    icon = "✓" if status == "OK" else ("⚠" if status == "SKIP" else "✗")
    print(f"  {icon}  {key}  {label:<40}  {status:<5}  {elapsed:.1f}s")
print(f"{SEP}")
print(f"  Totale: {total_elapsed:.1f}s")
print(f"{SEP}\n")

# Exit con errore se almeno uno step ha fallito
if any(s == "FAIL" for _, _, s, _ in results):
    sys.exit(1)