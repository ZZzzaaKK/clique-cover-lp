"""
utils_metrics.py
Zentrale Helfer für Evaluations-Metriken, Plots (Datenhygiene), Reproduzierbarkeit
und simple Heuristiken. Ziel: konsistente Auswertungen in WP1–WP5.
"""

from typing import Iterable, Sequence, Optional, Tuple, List
import math
import random
import numpy as np
import pandas as pd

# ---------------------------------------------------------------
# Reproduzierbarkeit
# ------------------------------------------------------------

def set_global_seeds(seed: int = 33) -> None:
    """
    Setzt globale Zufalls-Seeds für reproducible Runs.
    Wirkung:
      - random.seed(seed)
      - numpy.random.seed(seed)
    Call an beliebiger Stelle im Programmstart (z. B. main()).
    """
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------
# Numerik-Helfer: stabile Verhältnisse & Änderungen
# -------------------------------------------------------------

def safe_ratio(num: Optional[float],
               den: Optional[float],
               *,
               both_zero_value: float = 1.0,
               eps: float = 1e-12) -> float:
    """
    Stabile Division num/den mit sinnvoller Behandlung von Randfällen:
      - num oder den ist None/NaN  → NaN
      - |den| < eps und |num| < eps (≈0/0) → both_zero_value (Default 1.0)
      - |den| < eps und |num| >= eps      → NaN (statt inf, bessere Downstream-Stabilität)
      - sonst: num/den

    Typische Anwendung:
      - Speedup: t_baseline / t_kernel
      - Qualitätsquotient: cost_kernel / cost_baseline
    """
    if num is None or den is None:
        return np.nan
    if isinstance(num, float) and math.isnan(num):
        return np.nan
    if isinstance(den, float) and math.isnan(den):
        return np.nan
    if abs(den) < eps:
        return both_zero_value if abs(num) < eps else np.nan
    return num / den


def rel_change(new: Optional[float],
               base: Optional[float],
               *,
               denom_offset: float = 1.0,
               eps: float = 1e-12) -> float:
    """
    Relative Änderung: (new - base) / max(denom_offset + base, eps)
    Vorteile gegenüber Quotienten:
      - Immer endlich (kein inf), selbst wenn base = 0.
      - 0.0  → gleich
      - <0.0 → Verbesserung (z. B. niedrigere Kosten)
      - >0.0 → Verschlechterung
    """
    if new is None or base is None:
        return np.nan
    if isinstance(new, float) and math.isnan(new):
        return np.nan
    if isinstance(base, float) and math.isnan(base):
        return np.nan
    denom = max(denom_offset + base, eps)
    return (new - base) / denom


# ---------------------------------------------------------------
# DataFrame-Hygiene für Plots
# ------------------------------------------------------------

def clean_for_plot(df: pd.DataFrame,
                   cols: Sequence[str],
                   *,
                   dropna: bool = True) -> pd.DataFrame:
    """
    Ersetzt ±inf → NaN in den relevanten Spalten und droppt (optional) Zeilen,
    die in diesen Spalten NaN enthalten. Verhindert Matplotlib-Warnings wie
    'invalid value encountered in dot' und instabile Achsenlimits.
    """
    out = df.copy()
    out[list(cols)] = out[list(cols)].replace([np.inf, -np.inf], np.nan)
    if dropna:
        out = out.dropna(subset=list(cols))
    return out


def ensure_finite_array(a: Iterable[float]) -> np.ndarray:
    """
    Wandelt in numpy-Array, ersetzt ±inf → NaN. Praktisch für Vorverarbeitung
    vor Aggregationen.
    """
    arr = np.asarray(list(a), dtype=float)
    arr[np.isinf(arr)] = np.nan
    return arr


def nanmean(x: Iterable[float]) -> float:
    """
    Bequeme Kurzform für np.nanmean mit ensure_finite_array.
    """
    arr = ensure_finite_array(x)
    if arr.size == 0:
        return np.nan
    return float(np.nanmean(arr))


def safe_idxmax(s: pd.Series) -> Optional[int]:
    """
    idxmax, aber robust: ignoriert NaN, gibt None zurück, wenn alles NaN oder Serie leer.
    """
    if s is None or len(s) == 0:
        return None
    s2 = s.replace([np.inf, -np.inf], np.nan).dropna()
    if s2.empty:
        return None
    return int(s2.idxmax())


# ---------------------------------------------------------------
# Graph-Heuristiken (für Kernelization etc.)
# ------------------------------------------------------------

def graph_density(n: int, m: int) -> float:
    """
    Dichte eines einfachen, ungerichteten Graphen mit n Knoten und m Kanten.
    Formel: density = 2*m / (n*(n-1)) für n >= 2, sonst 0
    """
    if n < 2:
        return 0.0
    return (2.0 * float(m)) / float(n * (n - 1))


def should_kernelize(G,
                     *,
                     min_n: int = 80,
                     density_min: float = 0.05,
                     density_max: float = 0.65) -> bool:
    """
    Einfache, schnelle Heuristik: Kernelization nur, wenn sie sich lohnen dürfte.
      - min_n: Graphgröße ab der sich der Overhead typischerweise amortisiert
      - density_min/max: zu dünn oder zu dicht → Regeln greifen oft wenig

    Rückgabe:
      - True  → Kernelization aktivieren
      - False → lieber ohne Kernelization (Overhead sparen)

    Abhängigkeit networkx: absichtlich per Duck-Typing (G muss .number_of_nodes und .number_of_edges haben).
    """
    try:
        n = int(G.number_of_nodes())
        m = int(G.number_of_edges())
    except Exception:
        # Falls G diese Methoden nicht hat, Kernelization deaktivieren.
        return False

    d = graph_density(n, m)
    return (n >= min_n) and (density_min <= d <= density_max)


# ---------------------------------------------------------------
# Robust: Log-Log-Steigung (z. B. für Schätzung 'O(n^alpha)')
# ------------------------------------------------------------

def estimate_loglog_slope(x: Iterable[float],
                          y: Iterable[float],
                          *,
                          min_points: int = 2) -> float:
    """
    Schätzt die Steigung in loglog-Skala (alpha in O(n^alpha)):
      - Filtert nicht-positive oder nicht-endliche Werte.
      - Nutzt numpy.polyfit auf (log(x), log(y)).
      - Gibt bei zu wenig gültigen Punkten einen konservativen Default 2.0 zurück.

    Rückgabe:
      - float (auf 1 Nachkommastelle gerundet)
    """
    x_arr = ensure_finite_array(x)
    y_arr = ensure_finite_array(y)

    # Nur positive Werte sind für log sinnvoll
    mask = (x_arr > 0) & (y_arr > 0) & ~np.isnan(x_arr) & ~np.isnan(y_arr)
    if mask.sum() < min_points:
        return 2.0

    lx = np.log(x_arr[mask])
    ly = np.log(y_arr[mask])

    if lx.size < min_points:
        return 2.0

    # Grad-1-Fit: ly ≈ a*lx + b → a ist die Steigung
    a, b = np.polyfit(lx, ly, deg=1)
    # kosmetisch runden (vergleichbar mit vorherigem round(..., 1))
    return float(np.round(a, 1))
