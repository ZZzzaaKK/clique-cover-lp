"""
wrapperV2.py
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
    - Eine Clique in G ist ein independent set in Ḡ.
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
"""
wrapperV2.py
WRAPPERS for running reduction, heuristic and ILP on clique-cover instances.

Kernpunkte dieser Version:
- Einheitlicher Metadaten-Contract fuer alle Wrapper -> Evaluations-CSV ist konsistent befuellt.
- Warmstart fuer Coloring auf dem Komplement: korrekt via Clique-Cover auf G = complement(Gc).
- Standard-ILP validiert optional die Zuordnung (falls vom Solver geliefert).
- Reduced/Interactive-Wrapper liefern Runden-/Kernel-Serien und round_logs.
"""

import os
import time
from typing import Optional, Dict, Any, List

import networkx as nx

# Robuste Imports (Projekt / Fallback)
try:
    from src.utils import txt_to_networkx
except ImportError:
    from utils import txt_to_networkx

try:
    from src.algorithms.ilp_solver import solve_ilp_clique_cover
except ImportError:
    from algorithms.ilp_solver import solve_ilp_clique_cover

try:
    from src.algorithms.chalupa import ChalupaHeuristic
except ImportError:
    from algorithms.chalupa import ChalupaHeuristic

try:
    from src.reductions.reductions import apply_all_reductions
except ImportError:
    from reductions.reductions import apply_all_reductions

# Config flags (optional via ENV ueberschreibbar)
INTERACTIVE_MAX_ROUNDS = int(os.getenv("INTERACTIVE_MAX_ROUNDS", "10"))
USE_WARMSTART_DEFAULT = bool(int(os.getenv("USE_WARMSTART", "0")))

# Metadaten-Contract fuer alle Wrapper
EXPECTED_META_DEFAULTS: Dict[str, Any] = {
    'theta': None,
    'time': None,
    'status': 'unknown',
    'gap': None,
    'kernel_nodes': None,
    'kernel_edges': None,
    'used_warmstart': False,
    'start_objective': None,
    'improvement_over_start': None,
    'rounds': None,
    'k_start': None,
    'k_final': None,
    'k_rounds': None,
    'kernel_nodes_rounds': None,
    'kernel_edges_rounds': None,
    'round_logs': None,
    'assignment': None,  # falls Solver eine Zuordnung liefert
}


def _with_defaults(d: Optional[Dict[str, Any]], **overrides: Any) -> Dict[str, Any]:
    """
    Vereinheitlicht Rueckgaben der Wrapper: fuellt alle erwarteten Keys.
    'overrides' ueberschreibt sowohl Defaults als auch d (None wird ignoriert).
    """
    out = EXPECTED_META_DEFAULTS.copy()
    if d:
        out.update(d)
    for k, v in overrides.items():
        if v is not None:
            out[k] = v
    return out


def _compact_int_labels(G: nx.Graph) -> nx.Graph:
    """
    Bringt Node-Labels stabil auf 0..n-1. Wichtig nach Reduktionen
    fuer konsistente Array-/ILP-Indizierung.
    """
    if G is None or G.number_of_nodes() == 0:
        return G
    mapping = {old: i for i, old in enumerate(list(G.nodes()))}
    return nx.relabel_nodes(G, mapping, copy=True)


def _is_valid_clique_cover(G: nx.Graph, color_of: Dict[Any, int]) -> bool:
    """
    Prueft, ob die Farbklassen in G Cliquen bilden (valide Clique-Cover).
    color_of: dict[node] -> color_id
    """
    if not color_of:
        return False
    if set(color_of.keys()) != set(G.nodes()):
        return False

    by_color: Dict[int, List[Any]] = {}
    for v, c in color_of.items():
        by_color.setdefault(c, []).append(v)

    for _, nodes in by_color.items():
        for i, u in enumerate(nodes):
            for v in nodes[i + 1:]:
                if not G.has_edge(u, v):
                    return False
    return True


def _validate_result(G: nx.Graph, result: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
    """
    Validiert das Ergebnis anhand der Zuordnung (falls vorhanden) auf dem ORIGINAL-Graphen G.
    """
    validated = result.copy()
    validated['validation'] = {
        'is_valid': False,
        'error': None,
        'num_cliques': None
    }

    try:
        assignment = result.get('assignment')
        if not assignment:
            validated['validation']['error'] = "No assignment found in result"
            return validated

        # assignment: color_id -> [nodes] -> in node->color umwandeln
        color_of: Dict[Any, int] = {}
        for color_id, nodes in assignment.items():
            for node in nodes:
                color_of[node] = color_id

        is_valid = _is_valid_clique_cover(G, color_of)
        validated['validation']['is_valid'] = is_valid
        validated['validation']['num_cliques'] = len(assignment)
        if not is_valid:
            validated['validation']['error'] = "Assignment does not form a valid clique cover"

        if verbose:
            if is_valid:
                print("Validation OK: clique cover with {} cliques".format(len(assignment)))
            else:
                print("Validation FAILED:", validated['validation']['error'])
    except Exception as e:
        validated['validation']['error'] = "Validation failed: {}".format(str(e))
        if verbose:
            print("Validation error:", str(e))

    return validated


# Heuristik / Warmstarts

def chalupa_wrapper(txt_filepath: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Upper bound fuer theta(G) via Chalupa-Heuristik (direkt auf G).
    Liefert konsistent befuellte Metadaten.
    """
    try:
        if verbose:
            print("Running Chalupa on:", txt_filepath)

        t0 = time.time()
        G = txt_to_networkx(txt_filepath)
        G = _compact_int_labels(G)

        if ChalupaHeuristic is None:
            if verbose:
                print("ChalupaHeuristic not available")
            return _with_defaults({'theta': None, 'time': None, 'status': 'failed'})

        chalupa = ChalupaHeuristic(G)
        theta = None
        cover = None

        if hasattr(chalupa, 'iterated_greedy_clique_covering'):
            cover = chalupa.iterated_greedy_clique_covering()
            theta = len(cover) if cover is not None else None
        elif hasattr(chalupa, 'run'):
            res = chalupa.run()
            if isinstance(res, dict):
                theta = res.get('upper_bound')
                cover = res.get('cover')
            else:
                theta = int(res) if res is not None else None

        elapsed = time.time() - t0

        meta = {
            'theta': theta,
            'time': elapsed,
            'status': 'heuristic' if theta is not None else 'failed',
            'gap': None,
            'kernel_nodes': G.number_of_nodes(),
            'kernel_edges': G.number_of_edges(),
            'used_warmstart': False,
            'start_objective': None,
            'improvement_over_start': None,
            'rounds': 0,
            'k_start': None,
            'k_final': None,
            'k_rounds': None,
            'kernel_nodes_rounds': None,
            'kernel_edges_rounds': None,
            'round_logs': [{
                'stage': 'heuristic',
                'kernel_nodes': G.number_of_nodes(),
                'kernel_edges': G.number_of_edges(),
            }],
        }
        # Optional: assignment aus Cover rekonstruieren
        if cover:
            assignment = {i: list(c) for i, c in enumerate(cover)}
            meta['assignment'] = assignment

        return _with_defaults(meta)

    except Exception as e:
        if verbose:
            print("Chalupa failed:", e)
        return _with_defaults({'theta': None, 'time': None, 'status': 'failed'})


def _chalupa_warmstart(G: nx.Graph) -> Optional[Dict[Any, int]]:
    """
    Warmstart fuer ILP auf G (Clique-Cover direkt auf G).
    Rueckgabe: node -> clique_id (als Farben fuer Cover-ILP interpretierbar)
    """
    if ChalupaHeuristic is None:
        return None
    try:
        heuristic = ChalupaHeuristic(G)
        cover = None
        if hasattr(heuristic, 'iterated_greedy_clique_covering'):
            cover = heuristic.iterated_greedy_clique_covering()
        elif hasattr(heuristic, 'run'):
            r = heuristic.run()
            if isinstance(r, dict):
                cover = r.get('cover')

        if not cover:
            return None

        col: Dict[Any, int] = {}
        for c_idx, clique in enumerate(cover):
            for v in clique:
                col[v] = c_idx
        return col
    except Exception:
        return None


def _chalupa_warmstart_for_coloring(Gc: nx.Graph) -> Optional[Dict[Any, int]]:
    """
    Warmstart fuer Coloring auf Gc: benutze eine Clique-Cover auf G = complement(Gc),
    denn Cliquen in G entsprechen Unabhaengigkeitsmengen (Farben) in Gc.
    Rueckgabe: node -> color_id (Farben fuer Coloring auf Gc)
    """
    if ChalupaHeuristic is None:
        return None
    try:
        G = nx.complement(Gc)
        heuristic = ChalupaHeuristic(G)
        cover = None
        if hasattr(heuristic, 'iterated_greedy_clique_covering'):
            cover = heuristic.iterated_greedy_clique_covering()
        elif hasattr(heuristic, 'run'):
            _r = heuristic.run()
            if isinstance(_r, dict):
                cover = _r.get('cover')

        if not cover:
            return None

        col: Dict[Any, int] = {}
        for c_idx, clique in enumerate(cover):
            for v in clique:
                col[v] = c_idx  # Cliquen in G = unabhaengige Mengen in Gc -> Farben
        return col
    except Exception:
        return None


# ILP-Wrapper

def ilp_wrapper(
    txt_filepath: str,
    use_warmstart: bool = False,
    validate: bool = True,
    verbose: bool = False,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Standard-ILP-Wrapper (exakt).
    Berechnet theta(G) (Solver kann intern chi(G complement) loesen).
    """
    if verbose:
        print("Running ILP on:", txt_filepath)

    G = txt_to_networkx(txt_filepath)
    G_original = G.copy()
    G = _compact_int_labels(G)

    warm = _chalupa_warmstart(G) if use_warmstart else None
    start_objective = len(set(warm.values())) if warm else None
    if verbose and use_warmstart:
        if warm:
            print("Warmstart available ({} colors)".format(start_objective))
        else:
            print("Warmstart not available")

    t0 = time.time()
    res = solve_ilp_clique_cover(G, is_already_complement=False, warmstart=warm, **kwargs)
    elapsed = time.time() - t0

    if not isinstance(res, dict):
        res = {'theta': res}

    res_time = res.get('time', elapsed)
    res_status = res.get('status', 'optimal')
    res_gap = res.get('gap', 0.0)

    used_warm = bool(warm) if use_warmstart else False
    improvement = (start_objective - res['theta']) if (start_objective is not None and res.get('theta') is not None) else None

    meta = _with_defaults(
        res,
        time=res_time,
        status=res_status,
        gap=res_gap,
        kernel_nodes=G.number_of_nodes(),
        kernel_edges=G.number_of_edges(),
        used_warmstart=used_warm,
        start_objective=start_objective,
        improvement_over_start=improvement,
        rounds=0,
        k_start=None,
        k_final=res.get('theta'),
        k_rounds=None,
        kernel_nodes_rounds=None,
        kernel_edges_rounds=None,
        round_logs=None,
    )

    if validate and isinstance(meta, dict):
        meta = _validate_result(G_original, meta, verbose=verbose)

    return meta


def reduced_ilp_wrapper(
    txt_filepath: str,
    use_warmstart: bool = False,
    validate: bool = True,
    verbose: bool = False,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Reduzierte exakte Loesung:
    - Arbeitet auf dem Komplement Gc = G complement
    - Wendet Reduktionen auf Gc an
    - Loest chi(Gc_red) via ILP (-> theta(G))
    - Liefert Runden-/Kernel-Serien (als Single-Pass: rounds=1)
    """
    if not os.path.exists(txt_filepath):
        raise FileNotFoundError('Input file not found: {}'.format(txt_filepath))

    if verbose:
        print("Running Reduced ILP on:", txt_filepath)

    t0_total = time.time()

    G = txt_to_networkx(txt_filepath)
    G_original = G.copy()
    G = _compact_int_labels(G)

    original_nodes = G.number_of_nodes()
    original_edges = G.number_of_edges()

    # Komplement bilden und reduzieren
    Gc = nx.complement(G)
    Gc = _compact_int_labels(Gc)

    if verbose:
        print("Original: |V|={}, |E|={}".format(original_nodes, original_edges))
        print("Complement: |V|={}, |E|={}".format(Gc.number_of_nodes(), Gc.number_of_edges()))

    Gc_red, _meta, _add = apply_all_reductions(Gc, verbose=verbose, timing=verbose)
    Gc_red = _compact_int_labels(Gc_red)
    reduced_nodes = Gc_red.number_of_nodes()
    reduced_edges = Gc_red.number_of_edges()

    if verbose:
        print("After reductions: |V|={}, |E|={}".format(reduced_nodes, reduced_edges))

    # Optional: UB via Chalupa auf G (entspricht UB fuer chi(Gc))
    ub_cc = None
    try:
        if ChalupaHeuristic is not None:
            _chal = ChalupaHeuristic(G)
            if hasattr(_chal, 'iterated_greedy_clique_covering'):
                _cov = _chal.iterated_greedy_clique_covering()
                ub_cc = len(_cov) if _cov is not None else None
            elif hasattr(_chal, 'run'):
                _r = _chal.run()
                if isinstance(_r, dict) and 'upper_bound' in _r:
                    ub_cc = _r['upper_bound']
    except Exception:
        ub_cc = None

    # Warmstart fuer Coloring auf Gc_red
    warm = _chalupa_warmstart_for_coloring(Gc_red) if use_warmstart else None
    start_objective = len(set(warm.values())) if warm else None

    # Solve
    t0_solve = time.time()
    res = solve_ilp_clique_cover(Gc_red, is_already_complement=True, warmstart=warm, **kwargs)
    solve_time = time.time() - t0_solve
    total_time = time.time() - t0_total

    if not isinstance(res, dict):
        res = {'theta': res}

    used_warm = bool(warm) if use_warmstart else False
    improvement = (start_objective - res['theta']) if (start_objective is not None and res.get('theta') is not None) else None

    meta = _with_defaults(
        res,
        time=total_time if total_time is not None else solve_time,
        status=res.get('status', 'optimal'),
        gap=res.get('gap', 0.0),
        kernel_nodes=reduced_nodes,
        kernel_edges=reduced_edges,
        used_warmstart=used_warm,
        start_objective=start_objective,
        improvement_over_start=improvement,
        rounds=1,  # Single-Pass-Reduktion
        k_start=ub_cc,  # sinnvoller als original_nodes
        k_final=res.get('theta'),
        k_rounds=([ub_cc, res.get('theta')] if (ub_cc is not None and res.get('theta') is not None) else None),
        kernel_nodes_rounds=[original_nodes, reduced_nodes],
        kernel_edges_rounds=[original_edges, reduced_edges],
        round_logs=[{
            'round': 1,
            'ub': ub_cc,
            'reduction_removed_nodes': original_nodes - reduced_nodes,
            'kernel_nodes': reduced_nodes,
            'kernel_edges': reduced_edges
        }],
    )

    # Validierung auf reduzierten Kernen nicht 1:1 moeglich -> Hinweis
    if validate and isinstance(meta, dict):
        meta.setdefault('validation', {})
        meta['validation'].update({
            'note': 'Reduced instance on complement; validation on reduced kernel only (no back-mapping).',
            'final_reduced_nodes': reduced_nodes,
            'original_nodes': original_nodes,
            'rounds': 1
        })

    return meta


def interactive_reduced_ilp_wrapper(
    txt_filepath: str,
    use_warmstart: bool = False,
    max_rounds: int = INTERACTIVE_MAX_ROUNDS,
    validate: bool = True,
    verbose: bool = False,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Interaktive Reduktion:
    - Iterativ UB auf chi(Gc) verbessern (via theta(G) mit G = complement(Gc))
    - Reduktionen anwenden, bis keine Verbesserung mehr erfolgt
    - Finalen reduzierten Komplement-Graph exakt via ILP loesen
    - Liefert umfassende Runden-/Kernel-Logs
    """
    if not os.path.exists(txt_filepath):
        raise FileNotFoundError('Input file not found: {}'.format(txt_filepath))

    if verbose:
        print("Running Interactive Reduced ILP on:", txt_filepath)

    t0_total = time.time()

    G = txt_to_networkx(txt_filepath)
    G_original = G.copy()
    G = _compact_int_labels(G)

    Gc_curr = nx.complement(G)
    Gc_curr = _compact_int_labels(Gc_curr)

    rounds = 0
    prev_ub = float('inf')
    k_rounds: List[Optional[int]] = []                 # UB je Runde (Surrogat fuer k)
    kn_rounds: List[int] = [Gc_curr.number_of_nodes()]  # Kernel-Knoten je Stufe (Start inkl.)
    ke_rounds: List[int] = [Gc_curr.number_of_edges()]  # Kernel-Kanten je Stufe
    round_logs: List[Dict[str, Any]] = []

    if verbose:
        print("Starting interactive reduction (max rounds = {})".format(max_rounds))
        print("Initial complement: |V|={}, |E|={}".format(Gc_curr.number_of_nodes(), Gc_curr.number_of_edges()))

    while rounds < max_rounds:
        # UB auf chi(Gc_curr) via theta(G) mit G = complement(Gc_curr)
        if ChalupaHeuristic is None:
            if verbose:
                print("ChalupaHeuristic not available -> stop interactive loop")
            break

        G_curr = nx.complement(Gc_curr)
        chal = ChalupaHeuristic(G_curr)
        ub = float('inf')
        if hasattr(chal, 'iterated_greedy_clique_covering'):
            cover = chal.iterated_greedy_clique_covering()
            ub = len(cover) if cover is not None else float('inf')
        elif hasattr(chal, 'run'):
            _r = chal.run()
            if isinstance(_r, dict) and 'upper_bound' in _r:
                ub = _r['upper_bound']

        k_rounds.append(None if (ub == float('inf')) else int(ub))

        if verbose:
            print("Round {}: UB={} (prev={})".format(rounds + 1, ub, prev_ub))

        if ub >= prev_ub:
            if verbose:
                print("No UB improvement -> stop")
            break
        prev_ub = ub

        # Reduktionen anwenden
        nodes_before = Gc_curr.number_of_nodes()
        edges_before = Gc_curr.number_of_edges()

        Gc_next, _meta, _add = apply_all_reductions(Gc_curr, verbose=False, timing=False)
        Gc_next = _compact_int_labels(Gc_next)

        nodes_after = Gc_next.number_of_nodes()
        edges_after = Gc_next.number_of_edges()

        round_logs.append({
            'round': rounds + 1,
            'ub': None if (ub == float('inf')) else int(ub),
            'kernel_nodes_before': nodes_before,
            'kernel_edges_before': edges_before,
            'kernel_nodes': nodes_after,
            'kernel_edges': edges_after,
            'reduction_removed_nodes': nodes_before - nodes_after
        })

        if verbose:
            print("Reduction: |V| {} -> {}  |E| {} -> {}".format(
                nodes_before, nodes_after, edges_before, edges_after))

        # Abbruch, wenn strukturell nichts mehr passiert
        if nodes_after == nodes_before and edges_after == edges_before:
            if verbose:
                print("No further reductions possible -> stop")
            Gc_curr = Gc_next
            kn_rounds.append(nodes_after)
            ke_rounds.append(edges_after)
            rounds += 1
            break

        Gc_curr = Gc_next
        kn_rounds.append(nodes_after)
        ke_rounds.append(edges_after)
        rounds += 1

    if verbose:
        print("Finished after {} round(s). Final kernel: |V|={}, |E|={}".format(
            rounds, Gc_curr.number_of_nodes(), Gc_curr.number_of_edges()))

    # Warmstart auf finalem Kernel (Coloring auf Gc_curr)
    warm = _chalupa_warmstart_for_coloring(Gc_curr) if use_warmstart else None
    start_objective = len(set(warm.values())) if warm else None
    if verbose and use_warmstart:
        if warm:
            print("Warmstart generated ({} colors)".format(start_objective))
        else:
            print("Warmstart not available")

    # Exakte Loesung
    t0_solve = time.time()
    res = solve_ilp_clique_cover(Gc_curr, is_already_complement=True, warmstart=warm, **kwargs)
    t_solve = time.time() - t0_solve

    # Normalisieren / anreichern
    res = dict(res) if isinstance(res, dict) else {'theta': res}
    res_status = res.get('status', 'optimal')
    total_time = time.time() - t0_total
    res_time = res.get('time', total_time)

    used_warm = bool(warm) if use_warmstart else False
    improvement = (start_objective - res['theta']) if (start_objective is not None and res.get('theta') is not None) else None

    meta = _with_defaults(
        res,
        time=res_time,
        status=res_status,
        gap=res.get('gap', 0.0),
        used_warmstart=used_warm,
        start_objective=start_objective,
        improvement_over_start=improvement,
        rounds=rounds,
        k_start=(k_rounds[0] if k_rounds and k_rounds[0] is not None else None),
        k_final=(k_rounds[-1] if k_rounds and k_rounds[-1] is not None else None),
        k_rounds=(k_rounds if k_rounds else None),
        kernel_nodes=Gc_curr.number_of_nodes(),
        kernel_edges=Gc_curr.number_of_edges(),
        kernel_nodes_rounds=(kn_rounds if kn_rounds else None),
        kernel_edges_rounds=(ke_rounds if ke_rounds else None),
        round_logs=(round_logs if round_logs else None),
    )

    if validate and isinstance(meta, dict):
        meta.setdefault('validation', {})
        meta['validation'].update({
            'note': 'Interactive reduction on complement; validation on reduced kernels only.',
            'final_reduced_nodes': Gc_curr.number_of_nodes(),
            'original_nodes': G_original.number_of_nodes(),
            'rounds': rounds,
            'reduction_efficiency': '{} nodes removed in {} rounds'.format(
                G_original.number_of_nodes() - Gc_curr.number_of_nodes(), rounds)
        })

    return meta


# Batch & Debug

def batch_ilp(
    file_list: List[str],
    use_warmstart: bool = False,
    validate: bool = True,
    verbose: bool = False,
    **kwargs: Any
) -> List[Dict[str, Any]]:
    """Batchverarbeitung mehrerer Graphdateien (mit optionaler Validierung)."""
    results: List[Dict[str, Any]] = []
    for i, path in enumerate(file_list):
        if verbose:
            print("--- Processing {}/{}: {} ---".format(i + 1, len(file_list), path))
        try:
            res = ilp_wrapper(path, use_warmstart=use_warmstart, validate=validate, verbose=verbose, **kwargs)
            results.append({'file': path, **res})
            if verbose and isinstance(res, dict) and 'validation' in res:
                valid = res['validation'].get('is_valid', False)
                print("Result: theta(G) = {}, valid = {}".format(res.get('theta', 'N/A'), valid))
        except Exception as e:
            results.append({'file': path, 'error': str(e)})
            if verbose:
                print("Error:", str(e))
    return results


def debug_clique_cover(txt_filepath: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Debug: fuehrt Heuristik und mehrere exakte Varianten aus und vergleicht Ergebnisse.
    """
    print("=== DEBUGGING CLIQUE COVER for {} ===".format(txt_filepath))

    results: Dict[str, Any] = {}

    # 1. Chalupa Upper Bound
    print("1) Chalupa Upper Bound (Clique Covering):")
    chalupa_ub = chalupa_wrapper(txt_filepath, verbose=verbose)
    results['chalupa_ub'] = chalupa_ub

    # 2. Standard ILP
    print("2) Standard ILP:")
    ilp_res = ilp_wrapper(txt_filepath, use_warmstart=False, validate=True, verbose=verbose)
    results['ilp_standard'] = ilp_res

    # 3. ILP with Warmstart
    print("3) ILP with Warmstart:")
    ilp_warm_res = ilp_wrapper(txt_filepath, use_warmstart=True, validate=True, verbose=verbose)
    results['ilp_warmstart'] = ilp_warm_res

    # 4. Reduced ILP
    print("4) Reduced ILP:")
    red_ilp_res = reduced_ilp_wrapper(txt_filepath, use_warmstart=False, validate=True, verbose=verbose)
    results['reduced_ilp'] = red_ilp_res

    # Summary
    print("=== SUMMARY for {} ===".format(txt_filepath))
    if isinstance(chalupa_ub, dict):
        print("Chalupa UB (theta(G) upper bound):", chalupa_ub.get('theta'))

    for method, res in results.items():
        if isinstance(res, dict) and 'theta' in res:
            theta = res['theta']
            valid = res.get('validation', {}).get('is_valid', 'N/A')
            time_taken = res.get('time', 'N/A')
            print("{}: theta(G) = {}, valid = {}, time = {} s".format(method, theta, valid, time_taken))

    # Sanity check: Chalupa sollte >= ILP sein
    if isinstance(chalupa_ub, dict) and isinstance(results.get('ilp_standard'), dict):
        ilp_theta = results['ilp_standard'].get('theta')
        ch_theta = chalupa_ub.get('theta')
        if ilp_theta is not None and ch_theta is not None and ch_theta < ilp_theta:
            print("Warning: Heuristic upper bound smaller than exact solution (unexpected)")

    return results
