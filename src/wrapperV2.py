"""
WRAPPERS for running reduction, heuristic and ILP on clique-cover Instances...
THOUGHTS
--------------------- How to handle UB for best ILP Performance? ------------------------
Using the upper bound (UB) strictly is optional and depends on our goals:
- In the OG code (wrappers.py), `interactive_reduced_ilp_wrapper` uses UB from the Chalupa heuristic to guide reduction rounds, stopping when no improvement is made.
  This makes sense if we want to minimize problem size before ILP and avoid wasting effort on non-promising reductions.
- Strictly enforcing UB inside the ILP (e.g., as a constraint on the number of colors) could speed up solving by pruning solutions worse than the heuristic.
Here “hard equality” would mean forcing the ILP to *exactly* match the UB value (`sum(colors) == UB`). This is risky because if the heuristic UB is not optimal,
we might exclude the true optimal solution entirely.
- A safer approach imo is to use UB as a **hard inequality** (`sum(colors) ≤ UB`), which never eliminates the optimal solution but can still prune the search space.
- For simplified `reduced_ilp_wrapper`, we could pass UB into the ILP to potentially reduce runtime. This would involve adding an argument to the ILP solver
to accept a UB and incorporate it into the model.

In short: I would suggest to use UB as an upper limit (≤ UB) to contribute to better performance and avoid enforcing UB as an exact target (== UB) unless
we are certain it is optimal.

----------------------- Theory interlude --------------------------------
 χ(G) — Chromatic Number
    Definition: Die minimale Anzahl an Farben, die benötigt wird, um die Knoten von G so zu färben,
                dass keine zwei benachbarten Knoten dieselbe Farbe haben.

    Interpretation: Jede Farbe steht für eine unabhängige Menge (kein Kantenpaar innerhalb einer Farbe).
                    Färben = Partition von V(G) in minimale Anzahl unabhängiger Mengen.

    χ(G) = Größe der kleinsten Partition von V(G) in independent sets.

 θ(G) — Clique Cover Number
    Definition: Die minimale Anzahl an Cliquen, deren Vereinigung alle Knoten von G überdeckt.

    Interpretation: Jede Clique kann isoliert werden und deckt einen Teil der Knoten ab, so dass am Ende alle Knoten abgedeckt sind.
    Clique Cover = Partition von V(G) in minimale Anzahl vollständiger Teilgraphen.

    θ(G) = Größe der kleinsten Partition von V(G) in cliques.

 why important?
    - Cliquen in G ↔ unabhängige Mengen im Komplementgraphen Ḡ
    - Eine Clique in G ist eine independent set in Ḡ.
    - Eine independent set in G ist eine Clique in Ḡ.
    - im OG Code ist, so wie ich das verstanden habe, folgendes passiert: ILP hat χ(G) berechnet (Färbung von G) → und das war somit nicht die gesuchte Zahl für Clique Cover.
    - Bsp:
        - G: Dreieck (3 Knoten, alle verbunden): χ(G) = 3 (jeder Knoten andere Farbe), θ(G) = 1 (das Dreieck selbst ist eine Clique, deckt alle Knoten),
        - Ḡ: 3 isolierte Knoten: χ(Ḡ) = 1 → stimmt mit θ(G) überein.


    - ILP sollte jetzt mit dem folgenden Code Ḡ färben, wonach dann gilt: θ(G) = χ(Ḡ)
        (Clique Cover von G = Färbung des Komplements Ḡ).

------------------------------ θ(G) vs χ(Ḡ) ----------------------------------

What’s different now:
    - all wrappers operate on the complement graph Ḡ (consistent Clique Cover via θ(G)=χ(Ḡ)). should lead to correct results for Cloque Cover.?
    - reduced_ilp_wrapper applies reductions on Ḡ before ILP interactive_reduced_ilp_wrapper fixes the loop logic: keeps improving UB on Ḡ until it stops decreasing,
        then ILP on reduced Ḡ
    - interactive_reduced_ilp_wrapper fixes the loop logic: keeps improving UB on Ḡ until it stops decreasing, then ILP on reduced Ḡ
    - Added a chalupa_wrapper that also works on Ḡ (UB for θ(G))
    - ilp_wrapper(..., use_warmstart: bool)` toggles whether a Chalupa-based warmstart
        is computed and passed into the ILP solver. Default is False (fair ILP baseline).
    - Interoperability:
        - script uses `solve_ilp_clique_cover` from `algorithms.ilp_solver` which computes θ(G)
            by coloring the complement graph Ḡ (χ(Ḡ) = θ(G)).
        - If `use_warmstart=True` and Chalupa is available, a warmstart assignment is built.

Dependencies:
- utils.txt_to_networkx (reads graph from .txt)
- algorithms.ilp_solver.solve_ilp_clique_cover
- algorithms.chalupa.ChalupaHeuristic (optional; only if use_warmstart=True)

"""
import networkx as nx
from typing import Optional, Dict, Any, List
import os
import time

from utils import txt_to_networkx
from algorithms.ilp_solver import solve_ilp_clique_cover
try:
    from algorithms.chalupa import ChalupaHeuristic
except Exception:
    ChalupaHeuristic = None
from reductions.reductions import apply_all_reductions


# Kompakt-Relabelling, damit alle Knoten 0..n-1 sind (keine String-Labels etc.) ---
def _compact_int_labels(G: nx.Graph) -> nx.Graph:
    """
    brings node labels safely to 0...n-1 (stable sorting)
    important after all reductions/foldings in order to hold Array/ILP indizierung on a constant level
    (sonst index out of bounds error, da Labels wie 29 Knoten erhalten bleiben, aber nur nach Reduktion nur noch viel weniger da sind und
    faelschlicherweise altes Label abgegriffen wird und zusätzlich per Array neue Length -> crash)
    """
    mapping = {old: i for i, old in enumerate(G.nodes())}
    return nx.relabel_nodes(G, mapping, copy=True)


def chalupa_wrapper(txt_filepath: str) -> Optional[int]:
    """Upper Bound für θ(G) per Chalupa auf Ḡ (Komplement)."""
    try:
        print(f"{txt_filepath}")
        G = txt_to_networkx(txt_filepath)
        G = _compact_int_labels(G)
        Gc = nx.complement(G)
        Gc = _compact_int_labels(Gc)
        if ChalupaHeuristic is None:
            return None
        chalupa = ChalupaHeuristic(Gc)
        # robust: unterschiedliche API-Namen unterstützen
        if hasattr(chalupa, 'iterated_greedy_clique_covering'):
            covering = chalupa.iterated_greedy_clique_covering()
            return len(covering) if covering is not None else None
        elif hasattr(chalupa, 'run'):
            res = chalupa.run()
            # viele Implementationen liefern upper_bound o. Ä.
            return res.get('upper_bound') if isinstance(res, dict) else int(res)
        else:
            return None
    except Exception as e:
        print(f"Chalupa failed on {txt_filepath}: {e}")
        return None


def _chalupa_warmstart(G: nx.Graph) -> Optional[Dict[Any, int]]:
    """
    Liefert eine Knoten->Farbe-Zuordnung als Warmstart, berechnet über χ(Ḡ).
    Wichtig: Hier **immer** auf dem Komplement desjenigen Graphen rechnen, den man dem ILP übergibt.
    """
    if ChalupaHeuristic is None:
        return None
    try:
        Gc = nx.complement(G)
        Gc = _compact_int_labels(Gc)
        heuristic = ChalupaHeuristic(Gc)
        if hasattr(heuristic, 'coloring'):
            col = heuristic.coloring()  # Dict node->color
        elif hasattr(heuristic, 'iterated_greedy_clique_covering'):
            # evtl. Liste von Cliquen/Independent Sets -> in Farben umwandeln
            cover = heuristic.iterated_greedy_clique_covering()
            if cover is None:
                return None
            col = {}
            for c_idx, bucket in enumerate(cover):
                for v in bucket:
                    col[v] = c_idx
        elif hasattr(heuristic, 'run'):
            res = heuristic.run()
            col = res.get('coloring') if isinstance(res, dict) else None
        else:
            return None
        # Nur Knoten behalten, die im (ggf. reduzierten) G existieren
        return {v: int(c) for v, c in col.items() if v in G.nodes}
    except Exception:
        return None

def _is_valid_clique_cover(G: nx.Graph, color_of: dict) -> bool:
    """
    Prüft, ob die FarbkKlassen in G Cliquen bilden (d.h. gültige Clique-Cover).
    color_of: dict[node] -> color_id  (aus Chalupa auf Ḡ abgeleitet)
    """
    if not color_of:
        return False
    if set(color_of.keys()) != set(G.nodes()):
        return False
    by_color = {}
    for v, c in color_of.items():
        by_color.setdefault(c, []).append(v)
    for nodes in by_color.values():
        # Jede Farbe muss in G eine Clique sein
        for i, u in enumerate(nodes):
            for v in nodes[i+1:]:
                if not G.has_edge(u, v):
                    return False
    return True


def ilp_wrapper(txt_filepath: str, use_warmstart: bool = False, **kwargs):
    G = txt_to_networkx(txt_filepath)
    G = _compact_int_labels(G)  # <<< NEU: kompakte Labels 0..n-1
    warm = _chalupa_warmstart(G) if use_warmstart else None
    t0 = time.time()
    res = solve_ilp_clique_cover(G, warmstart=warm, **kwargs)
    if isinstance(res, dict) and ('time' not in res or res['time'] is None):
        res['time'] = time.time() - t0
    return res



def reduced_ilp_wrapper(txt_filepath: str, use_warmstart: bool = False, **kwargs) -> Dict[str, Any]:
    """
    Apply reductions on the complement graph Ḡ, then solve χ(Ḡ_red) via ILP
    to obtain θ(G) (since θ(G) = χ(Ḡ)).

    Pipeline:
      1) Einlesen G
      2) Ḡ = complement(G)
      3) Reduktionen auf Ḡ  →  Ḡ_red
      4) G_red_back = complement(Ḡ_red)
      5) ILP auf G_red_back (intern wird wieder Ḡ_red gefärbt)

    Hinweis:
      - Warmstart wird (falls aktiviert) aus G erzeugt (_chalupa_warmstart(G)).
      - Rückgabewert entspricht dem von solve_ilp_clique_cover (meist dict).

    Returns
    -------
    dict | int | None
        θ(G) auf der reduzierten Instanz bzw. Solver-Response.

    ****WICHTIG**** Jede Reduktionsroutine (und später ILP-Solver) sieht jetzt nur Graphen mit Labels (0...n-1).
        -> Behebung der OutofBounds-Zugriffe, auch wenn vorher Knoten mit hohen IDs weggefallen sind
    """

    if not os.path.exists(txt_filepath):
            raise FileNotFoundError(f'Input file not found: {txt_filepath}')

    G = txt_to_networkx(txt_filepath)
    G = _compact_int_labels(G)
    Gc = nx.complement(G)
    Gc = _compact_int_labels(Gc)

    Gc_red, _meta = apply_all_reductions(Gc, verbose=False, timing=False)
    Gc_red = _compact_int_labels(Gc_red)  #(nach Reduktion sicherheitshalber)

    G_red_back = nx.complement(Gc_red)
    G_red_back = _compact_int_labels(G_red_back)

    # Warmstart MUSS zu G_red_back passen, sonst inkonsistente Keys
    warm = _chalupa_warmstart(G_red_back) if use_warmstart else None
    return solve_ilp_clique_cover(G_red_back, warmstart=warm, **kwargs)

def interactive_reduced_ilp_wrapper(
            txt_filepath: str,
            use_warmstart: bool = False,
            max_rounds: int = 10,
            **kwargs
    ) -> Dict[str, Any]:
        if not os.path.exists(txt_filepath):
            raise FileNotFoundError(f'Input file not found: {txt_filepath}')
        """
        Iteriere: UB per Chalupa auf Ḡ(G_curr) verbessern; Reduktionen auf **G_curr** anwenden;
        stoppe, wenn UB nicht mehr sinkt; dann ILP auf finalem **G_curr**.
        - vor UB-Berechnung: Chalupa bekommt sauber beschriftetes Komplement
        - nach Reduktion sofort neues Labelling 0...n-1
        - vor ILP: nochmal Sicherheitscheck, damit Arraylaengen mit Knotenzahl übereinstimmen
        """
        G_curr = txt_to_networkx(txt_filepath)
        G_curr = _compact_int_labels(G_curr)  # Früh kompaktes Labeling

        prev_ub = float('inf')
        rounds = 0
        while rounds < max_rounds:
            # 1) UB per Chalupa auf Komplement von G_curr
            if ChalupaHeuristic is None:
                break
            Gc = nx.complement(G_curr)
            Gc = _compact_int_labels(Gc)  # <-- hier ebenfalls kompakten!
            chal = ChalupaHeuristic(Gc)
            if hasattr(chal, 'iterated_greedy_clique_covering'):
                cover = chal.iterated_greedy_clique_covering()
                ub = len(cover) if cover is not None else float('inf')
            elif hasattr(chal, 'run'):
                res = chal.run()
                ub = res.get('upper_bound', float('inf')) if isinstance(res, dict) else float('inf')
            else:
                ub = float('inf')

            if ub >= prev_ub:
                break
            prev_ub = ub

            # 2) Reduktionen auf G_curr
            G_curr, _ = apply_all_reductions(G_curr, verbose=False, timing=False)
            G_curr = _compact_int_labels(G_curr)  # <-- hier sofort relabeln
            rounds += 1

        # Final: kompaktes Labeling vor ILP
        G_curr = _compact_int_labels(G_curr)

        warm = _chalupa_warmstart(G_curr) if use_warmstart else None
        return solve_ilp_clique_cover(G_curr, warmstart=warm, **kwargs)

def batch_ilp(file_list: List[str], use_warmstart: bool = False, **kwargs) -> List[Dict[str, Any]]:
    results = []
    for path in file_list:
        try:
            res = ilp_wrapper(path, use_warmstart=use_warmstart, **kwargs)
            results.append({'file': path, **res})
        except Exception as e:
            results.append({'file': path, 'error': str(e)})
    return results

""" old Version with conflicts
from typing import Optional, Dict, Any, List
import os
import networkx as nx

from utils import txt_to_networkx

from algorithms.ilp_solver import solve_ilp_clique_cover
try:
    from algorithms.chalupa import ChalupaHeuristic
except Exception:
    ChalupaHeuristic = None
from reductions.reductions import apply_all_reductions

def chalupa_wrapper(txt_filepath):
    #Wrapper for Chalupa algorithm
    try:
        print(f"{txt_filepath}")
        G = txt_to_networkx(txt_filepath)
        chalupa = ChalupaHeuristic(nx.complement(G))
        result = chalupa.run()
        return result['upper_bound']
    except Exception as e:
        print(f"Chalupa failed on {txt_filepath}: {e}")
        return None

def _chalupa_warmstart(G: nx.Graph) -> Optional[Dict[Any, int]]:
    if ChalupaHeuristic is None:
        return None
    Gc = nx.complement(G)
    try:
        heuristic = ChalupaHeuristic(Gc)
        if hasattr(heuristic, 'coloring'):
            col = heuristic.coloring()
        elif hasattr(heuristic, 'iterated_greedy_clique_covering'):
            col = heuristic.iterated_greedy_clique_covering()
        else:
            return None
        return {v: int(c) for v, c in col.items() if v in G.nodes}
    except Exception:
        return None


def ilp_wrapper(txt_filepath: str, use_warmstart: bool = False, **kwargs) -> Dict[str, Any]:
    if not os.path.exists(txt_filepath):
        raise FileNotFoundError(f'Input file not found: {txt_filepath}')
    G = txt_to_networkx(txt_filepath)
    warm = _chalupa_warmstart(G) if use_warmstart else None
    return solve_ilp_clique_cover(G, warmstart=warm, **kwargs)
    # Compute θ(G) exactly by solving χ(Ḡ) with the ILP solver.
    #
    #    Parameters
    #    ----------
    #    txt_filepath : str
    #        Path to a graph instance in the repo's text format.
    #   
    #       Returns
    #       -------
    #       int or None
    #           Clique cover number θ(G). None if ILP failed.
    

def reduced_ilp_wrapper(txt_filepath: str, use_warmstart: bool = False, **kwargs) -> Dict[str, Any]:
    if not os.path.exists(txt_filepath):
        raise FileNotFoundError(f'Input file not found: {txt_filepath}')
    G = txt_to_networkx(txt_filepath)
    warm = _chalupa_warmstart(G) if use_warmstart else None
    G_reduced, _ = apply_all_reductions(G)
    return solve_ilp_clique_cover(G_reduced, warmstart=warm, **kwargs)
    # Apply reductions on Ḡ and then solve χ(Ḡ_red) via ILP to get θ(G).
    #
    #Note
    #----
    #This function currently does **not** pass a UB into the ILP. If wanted for pruning, a constraint like `sum(y_c) ≤ UB`
    #(where y_c indicates color c is used) should be added.
    #
    #Returns
    #-------
    #int or None
    #    Clique cover number θ(G) on the reduced instance, or None on failure.
    
    try:
        print(f"{txt_filepath}")
        G = txt_to_networkx(txt_filepath)
        Gc = nx.complement(G)
        G_red, _meta = apply_all_reductions(Gc, verbose=False, timing=False)
        result = solve_ilp_clique_cover(G_red)
        if isinstance(result, dict) and 'error' in result:
            print(f"Reduced ILP failed on {txt_filepath}: {result['error']}")
            return None
        chromatic_number = result['chromatic_number'] if isinstance(result, dict) else int(result)
        return chromatic_number
    except Exception as e:
        print(f"Reduced ILP failed on {txt_filepath}: {e}")
        return None


def interactive_reduced_ilp_wrapper(txt_filepath: str, use_warmstart: bool = False, max_rounds: int = 10, **kwargs) -> Dict[str, Any]:
    if not os.path.exists(txt_filepath):
        raise FileNotFoundError(f'Input file not found: {txt_filepath}')
    G = txt_to_networkx(txt_filepath)
    Gc = nx.complement(G)
    prev_ub = float('inf')
    rounds = 0
    while rounds < max_rounds:
        if ChalupaHeuristic is None:
            break
        heuristic = ChalupaHeuristic(Gc)
        covering = heuristic.iterated_greedy_clique_covering()
        ub = len(covering) if covering is not None else float('inf')
        if ub >= prev_ub:
            break
        prev_ub = ub
        Gc, _ = apply_all_reductions(Gc)
        rounds += 1
    warm = _chalupa_warmstart(G) if use_warmstart else None
    return solve_ilp_clique_cover(Gc, warmstart=warm, **kwargs)


def batch_ilp(file_list: List[str], use_warmstart: bool = False, **kwargs) -> List[Dict[str, Any]]:
    results = []
    for path in file_list:
        try:
            res = ilp_wrapper(path, use_warmstart=use_warmstart, **kwargs)
            results.append({'file': path, **res})
        except Exception as e:
            results.append({'file': path, 'error': str(e)})
    return results
"""