'''
EDIT suggestion:
Add ground truth (Clique Cover Number θ(G)) to graph instance files.

Berechne θ(G) (θ(G) = χ(Ḡ)) per ILP auf dem Komplementgraphen Ḡ (kalt, ohne Warmstart) und schreibe eine
standardisierte Ground‑Truth‑Zeile in eine Begleitdatei.

Ausgabezeile (exakt):
    "Clique Cover Number θ(G): K (Calculated by ILP on complement graph Ḡ)."

Beispiel:
    python -m add_ground_truth path/to/graph.txt --overwrite


Why this matters:
- We solve Clique Cover by coloring the **complement graph** (θ(G) = χ(Ḡ)).
- Writing the line as **"Clique Cover Number θ(G): …"** avoids the previous
  ambiguity (which imo incorrectly suggested χ(G)).
- The note **"Calculated by ILP on complement graph"** documents the method,
  making downstream evaluation (WP1/WP2) unambiguous and reproducible.
'''
from pathlib import Path
import sys
from typing import Optional

# ----------------------
# Import-Handling: funktioniert sowohl mit direktem Aufruf als auch als Modul
# ----------------------
try:
    # Relativer Import, wenn als Modul gestartet
    from .wrapperV2 import ilp_wrapper
    from .utils import get_value
except ImportError:
    # Fallback: Projekt-Root zum sys.path hinzufügen und absolute Imports nutzen
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.wrapperV2 import ilp_wrapper
    from src.utils import get_value

def _append_theta_line(txt_path: Path, theta: int) -> None:
    #Append the standardized θ(G) line to the given .txt file.
    with open(txt_path, 'a', encoding='utf-8') as f:
        # Kommentarzeile als Kontext-Hinweis (bleibt in der Datei)
        f.write("# Calculated by ILP on complement graph Ḡ\n")
        # Exakt geforderte Ausgabezeile
        f.write(f"Clique Cover Number θ(G): {theta} (Calculated by ILP on complement graph Ḡ).\n")

def add_ground_truth_if_missing(directory: str, verbose: bool = True) -> None:
    """
    Durchsuche `directory` rekursiv nach .txt-Dateien und füge die θ(G)-Zeile an,
    falls sie fehlt.

    Hinweise
    --------
    - `ilp_wrapper` berechnet θ(G) exakt via χ(Ḡ)-ILP (warmstart standardmäßig deaktiviert).
    - Die Prüfung nutzt `get_value(..., attribute_name="Clique Cover Number θ(G)")`.
    """
    path = Path(directory)
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {directory}")

    for txt_file in path.glob("**/*.txt"):
        # Prüfe, ob die neue Ground-Truth-Zeile bereits existiert
        existing_theta = get_value(str(txt_file), "Clique Cover Number θ(G)")
        if existing_theta is not None:
            if verbose:
                print(f"Ground truth already exists for {txt_file.name}: {existing_theta}")
            continue

        if verbose:
            print(f"→ Computing ground truth for {txt_file.name} …")

        # Kalter ILP-Lauf (use_warmstart=False) für faire Ground-Truth
        res = ilp_wrapper(str(txt_file), use_warmstart=False, time_limit=600, mip_gap=0.0, verbose=False, return_assignment=False)
        if isinstance(res, dict):
            status = res.get('status')
            theta = res.get('theta')
            if status not in ("optimal", "time_limit", "suboptimal", "interrupted") or theta is None:
                print(f"Failed to compute θ(G) for {txt_file.name} (status={status})")
                continue
        else:
                # Falls ilp_wrapper aus Kompatibilitätsgründen nur eine Zahl liefert
            theta = int(res)

        _append_theta_line(txt_file, int(theta))
        if verbose:
            print(f"  + Added: Clique Cover Number θ(G): {theta}")


def _cli() -> None:
    """CLI-Einstieg: `python -m add_ground_truth_dir <dir>`"""
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "test_graphs/generated/perturbed"
    add_ground_truth_if_missing(target_dir)


if __name__ == "__main__":
    _cli()
