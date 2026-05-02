#!/usr/bin/env python3
"""
run_all.py
──────────
Esegue l'intera pipeline in ordine.

Pipeline base (senza mode):
  02a     →  diagnostics prezzi
  02b     →  diagnostics margini
  02c     →  change point detection (margin e price)
  02d     →  analisi controfattuale (legacy)

Pipeline ITS (per ciascun mode):
  v1      →  ITS Metodo 1: OLS Naïve           [detection: sliding window naïve]
  v2      →  ITS Metodo 2: OLS HAC Newey-West  [detection: Window L2 Discrepancy]
  v3      →  ITS Metodo 3: ARIMAX              [detection: PELT RBF]
  v4      →  ITS Metodo 4: SARIMAX + CCF       [detection: BOCPD Bayesian Online]
  cmp     →  Confronto metodi ITS

Modalità di break detection (--mode / argomento positivo):
  fixed     : tutti usano la data dello shock hardcodata  [default per ITS]
  detected  : per ogni metodo, il break viene rilevato automaticamente
              con l'algoritmo proprio del metodo (nessuna dipendenza da 02c)

Variante di detection (--detect / argomento positivo, solo per mode=detected):
  margin    : detection sul margine distributore           [default → entrambi]
  price     : detection sul prezzo alla pompa netto

  Di default, se --mode detected viene specificato senza --detect,
  la pipeline gira ENTRAMBE le varianti (margin + price), producendo:
    data/plots/its/detected/margin/{metodo}/
    data/plots/its/detected/price/{metodo}/

Uso:
  python3 run_all.py                            # solo pipeline base (02a→02d)
  python3 run_all.py fixed                      # ITS in modalità fixed
  python3 run_all.py detected                   # ITS detected: margin + price
  python3 run_all.py detected margin            # ITS detected solo margin
  python3 run_all.py detected price             # ITS detected solo price
  python3 run_all.py detected --detect=price    # equivalente a sopra
  python3 run_all.py fixed detected             # entrambe le modalità ITS
  python3 run_all.py v1 v2 fixed                # solo v1/v2 in modalità fixed
  python3 run_all.py v3 cmp detected            # v3 e compare in detected (margin+price)
  python3 run_all.py v3 detected price          # v3 detected solo price
  python3 run_all.py 02a 02b                    # solo step base indicati
"""

import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent

# ── Step base (non richiedono --mode) ────────────────────────────────────────
BASE_STEPS = [
    ("02a", "02a_diagnostics_price.py",       "Diagnostics — Prezzi"),
    ("02b", "02b_diagnostics_margin.py",      "Diagnostics — Margini"),
    ("02c", "02c_change_point_detection.py",  "Change Point Detection — Margini"),
    ("02d", "02d_counterfactual_gains.py",    "Analisi Controfattuale — Legacy"),
]

# ── Step ITS (richiedono --mode fixed|detected) ──────────────────────────────
ITS_STEPS = [
    ("v1",  "02d_v1_naive.py",        "ITS Metodo 1 — OLS Naïve"),
    ("v2",  "02d_v2_intermediate.py", "ITS Metodo 2 — OLS HAC Newey-West"),
    ("v3",  "02d_v3_arimax.py",       "ITS Metodo 3 — ARIMAX (Masena & Shongwe 2024)"),
    ("v3",  "02d_v4_transfer.py",     "ITS Metodo 4 — SARIMAX"),
    ("cmp", "02d_compare.py",         "ITS Confronto — 3 Metodi"),
]

ITS_KEYS  = {k for k, _, _ in ITS_STEPS}
BASE_KEYS = {k for k, _, _ in BASE_STEPS}
MODE_VALS   = {"fixed", "detected"}
DETECT_VALS = {"margin", "price"}  # varianti di detection (solo mode=detected)

# Ogni metodo ITS ha il proprio algoritmo di detection autonomo:
#   v1  → sliding window naïve     (margin o price)
#   v2  → Window L2 Discrepancy    (margin o price)
#   v3  → PELT RBF                 (margin o price)
#   v4  → BOCPD Bayesian Online    (margin o price)
DETECT_ALLOWED: dict[str, set[str]] = {
    "v1":  {"margin", "price"},
    "v2":  {"margin", "price"},
    "v3":  {"margin", "price"},
    "v4":  {"margin", "price"},
    "cmp": {"margin", "price"},
}


def run_step(filename: str, extra_args: list[str]) -> tuple[str, float]:
    script = BASE_DIR / filename
    if not script.exists():
        print(f"\n  ⚠  {filename} non trovato — salto.")
        return "SKIP", 0.0

    cmd  = [sys.executable, str(script)] + extra_args
    t0   = time.time()
    proc = subprocess.run(cmd, cwd=str(BASE_DIR))
    elapsed = time.time() - t0

    if proc.returncode == 0:
        print(f"\n  ✓  completato in {elapsed:.1f}s")
        return "OK", elapsed
    else:
        print(f"\n  ✗  errore (exit {proc.returncode}) dopo {elapsed:.1f}s")
        return "FAIL", elapsed


def main() -> None:
    raw_args = sys.argv[1:]

    # ── Parsing argomenti ─────────────────────────────────────────────────────
    modes_requested:   list[str] = []
    detects_requested: list[str] = []
    step_targets:      list[str] = []

    i = 0
    while i < len(raw_args):
        a = raw_args[i]
        if a == "--mode" and i + 1 < len(raw_args):
            modes_requested.append(raw_args[i + 1])
            i += 2
        elif a.startswith("--detect="):
            detects_requested.append(a.split("=", 1)[1])
            i += 1
        elif a == "--detect" and i + 1 < len(raw_args):
            detects_requested.append(raw_args[i + 1])
            i += 2
        elif a in MODE_VALS:
            modes_requested.append(a)
            i += 1
        elif a in DETECT_VALS:
            detects_requested.append(a)
            i += 1
        else:
            step_targets.append(a)
            i += 1

    # Deduplica mantenendo l'ordine
    def _dedup(lst: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for x in lst:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out

    modes   = _dedup(modes_requested)
    detects = _dedup(detects_requested)

    # ── Determina cosa eseguire ───────────────────────────────────────────────
    targets_are_base = step_targets and all(t in BASE_KEYS or t.startswith("02") for t in step_targets)
    targets_are_its  = step_targets and all(t in ITS_KEYS for t in step_targets)
    no_targets       = not step_targets

    if no_targets and not modes:
        run_base_steps = BASE_STEPS
        run_its_steps  = []
        modes          = []
    elif no_targets and modes:
        run_base_steps = []
        run_its_steps  = ITS_STEPS
    elif targets_are_its:
        run_base_steps = []
        run_its_steps  = [(k, f, l) for k, f, l in ITS_STEPS
                          if any(t in k for t in step_targets)]
        if not modes:
            modes = ["fixed"]
    elif targets_are_base:
        run_base_steps = [(k, f, l) for k, f, l in BASE_STEPS
                          if any(t in k for t in step_targets)]
        run_its_steps  = []
    else:
        run_base_steps = [(k, f, l) for k, f, l in BASE_STEPS
                          if any(t in k for t in step_targets)]
        run_its_steps  = [(k, f, l) for k, f, l in ITS_STEPS
                          if any(t in k for t in step_targets)]
        if run_its_steps and not modes:
            modes = ["fixed"]

    SEP = "═" * 70
    total_start = time.time()
    results: list[tuple] = []

    # ── Step base ─────────────────────────────────────────────────────────────
    for key, filename, label in run_base_steps:
        print(f"\n{SEP}")
        print(f"  ▶  {key}  |  {label}")
        print(f"{SEP}\n")
        status, elapsed = run_step(filename, [])
        results.append((key, label, "–", "–", status, elapsed))

    # ── Step ITS per ogni mode (e variante detect se mode=detected) ───────────
    for mode in modes:
        if mode == "detected":
            # Default: entrambe le varianti; altrimenti filtra
            detect_variants = detects if detects else ["margin", "price"]
        else:
            detect_variants = ["–"]  # fixed non usa --detect

        for detect in detect_variants:
            for key, filename, label in run_its_steps:
                # Salta la combinazione se il metodo non supporta questa variante di detect
                if mode == "detected" and detect in DETECT_VALS:
                    allowed = DETECT_ALLOWED.get(key, {"margin", "price"})
                    if detect not in allowed:
                        print(f"\n  ⚙  {key}[detected/{detect}]  →  skipped "
                              f"({detect} non supportato da {key})")
                        continue
                if mode == "detected":
                    tag       = f"{key}[detected/{detect}]"
                    extra_args = ["--mode", "detected", "--detect", detect]
                    detect_str = detect
                else:
                    tag       = f"{key}[fixed]"
                    extra_args = ["--mode", "fixed"]
                    detect_str = "–"

                print(f"\n{SEP}")
                print(f"  ▶  {tag}  |  {label}")
                print(f"{SEP}\n")
                status, elapsed = run_step(filename, extra_args)
                results.append((tag, label, mode, detect_str, status, elapsed))

    # ── Riepilogo ─────────────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    print(f"\n{SEP}")
    print("  RIEPILOGO PIPELINE")
    print(f"{SEP}")
    for key, label, mode_tag, detect_tag, status, elapsed in results:
        icon = "✓" if status == "OK" else ("⚠" if status == "SKIP" else "✗")
        mode_str   = f"[{mode_tag}]"   if mode_tag   != "–" else "          "
        detect_str = f"[{detect_tag}]" if detect_tag != "–" else "        "
        print(f"  {icon}  {key:<30} {mode_str:<12} {detect_str:<10}"
              f"  {label:<38}  {status:<5}  {elapsed:.1f}s")
    print(f"{SEP}")
    print(f"  Totale: {total_elapsed:.1f}s")
    print(f"{SEP}\n")

    if any(s == "FAIL" for *_, s, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    main()