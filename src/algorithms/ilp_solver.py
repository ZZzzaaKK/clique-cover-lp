"""
Integer Linear Programming (ILP) formulation for the vertex clique coloring problem.

kurzer Hinweis zur Integration:

- Aufruf: res = solve_ilp_clique_cover(G, warmstart=heur_cover_dict, time_limit=300, mip_gap=0.0)
- Ergebnis: res['theta'] ist θ(G); identisch zu res['chi_complement'].
- Für add_ground_truth.py: Zeile Clique Cover Number θ(G): {res['theta']} (Calculated by ILP on complement graph Ḡ).
- Für Warmstart: akzeptiert {node: color} oder {color: [nodes...]} oder Liste in G.nodes()-Reihenfolge.
"""


"""
ILP solver for the Clique Cover Number θ(G) via Coloring on the complement graph Ḡ.

Core idea
---------
θ(G) = χ(Ḡ). We therefore color the complement graph Gc = Ḡ with as few colors as possible.
Each color class in Ḡ is an independent set in Ḡ, i.e., a clique in G, which forms one clique
in a vertex clique cover of G.

Features
--------
- Computes θ(G) by solving χ(Ḡ) (coloring the complement graph).
- Gurobi-based assignment model (binary x[v,i], y[i])
- Symmetry breaking: y_i ≥ y_{i+1}, and fix the highest-degree node to color 0
- Optional warmstart from a given clique cover (or coloring) assignment
- Clear return dictionary with θ(G), χ(Ḡ), assignment (colors → original node labels), status, time, gap

API
---
solve_ilp_clique_cover(
    G: nx.Graph,
    time_limit: int = 300,
    mip_gap: float | None = None,
    threads: int | None = None,
    max_colors: int | None = None,
    warmstart: dict | None = None,   # node → color OR {color: [nodes]} OR list[int] aligned to nodes
    verbose: bool = False,
    return_assignment: bool = True,
) -> dict

Notes on warmstart
------------------
- You can pass a clique-cover assignment for G (θ-labeling). The same labels serve as a valid coloring
  for Ḡ, so we can use them directly to set .Start values for x and y.
- Accepted forms:
    1) {node: color}
    2) {color: [nodes, ...]}
    3) list/tuple of colors ordered as nodelist (G.nodes())

Return value (clear)
--------------------
{
  'status': 'optimal'|'time_limit'|'infeasible'|...,  # Gurobi status as string
  'theta': int | None,             # Clique Cover Number θ(G)
  'chi_complement': int | None,    # Chromatic number of Ḡ (same number)
  'assignment': {                  # Only if return_assignment and a feasible solution exists
      color_id: [original_node_labels,...],
      ...
  },
  'time': float | None,            # seconds
  'gap': float | None,             # relative MIP gap
  'n_nodes': int,
  'n_edges': int,
  'n_edges_complement': int,
}
"""

"""
Integer Linear Programming (ILP) solver for the Clique Cover Number θ(G).

Core idea
---------
θ(G) = χ(Ḡ). We therefore color the complement graph Gc = Ḡ with as few colors as possible.
Each color class in Ḡ is an independent set in Ḡ, i.e., a clique in G.

IMPORTANT: Fixed complement handling
------------------------------------
This solver now has a clear API regarding complement graphs:
- By default (is_already_complement=False): Takes G, computes Gc internally, colors Gc
- With is_already_complement=True: Takes Gc directly, colors it without another complement

Features
--------
- Computes θ(G) by solving χ(Ḡ) (coloring the complement graph)
- Gurobi-based assignment model with binary variables x[v,i] and y[i]
- Symmetry breaking: y_i ≥ y_{i+1}, fix highest-degree node to color 0
- Optional warmstart from given clique cover assignment
- Clear return dictionary with all relevant metrics

API
---
solve_ilp_clique_cover(
    G: nx.Graph,
    is_already_complement: bool = False,  # NEW: Indicates if G is already Ḡ
    time_limit: int = 300,
    mip_gap: float | None = None,
    threads: int | None = None,
    max_colors: int | None = None,
    warmstart: dict | None = None,
    verbose: bool = False,
    return_assignment: bool = True,
) -> dict
"""

import sys
import os
import time
from typing import Dict, List, Any, Optional, Union
import networkx as nx

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    gp = None
    GRB = None

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def _parse_warmstart(
        G: nx.Graph,
        nodelist: List[Any],
        warmstart: Any,
) -> Optional[Dict[Any, int]]:
    """
    Normalize warmstart into {node: color}, or None if invalid.

    Accepted formats:
    1. {node: color}
    2. {color: [nodes, ...]}
    3. list/tuple of colors ordered as nodelist
    """
    if warmstart is None:
        return None

    # Case 1: dict node->color or color->list[nodes]
    if isinstance(warmstart, dict):
        keys = set(warmstart.keys())
        node_keys = set(nodelist)

        # Check if it's node->color mapping
        if keys & node_keys:
            return {node: int(warmstart[node]) for node in nodelist if node in warmstart}

        # Otherwise assume it's color->list[nodes]
        assignment: Dict[Any, int] = {}
        for c, nodes in warmstart.items():
            try:
                color = int(c)
            except Exception:
                continue
            for node in nodes:
                if node in node_keys:
                    assignment[node] = color
        return assignment or None

    # Case 2: list/tuple aligned with nodelist
    if isinstance(warmstart, (list, tuple)):
        if len(warmstart) != len(nodelist):
            return None
        return {node: int(color) for node, color in zip(nodelist, warmstart)}

    return None


def solve_ilp_clique_cover(
        G: nx.Graph,
        is_already_complement: bool = False,
        time_limit: int = 300,
        mip_gap: Optional[float] = None,
        threads: Optional[int] = None,
        max_colors: Optional[int] = None,
        warmstart: Optional[Union[Dict[Any, int], Dict[int, List[Any]], List[int]]] = None,
        verbose: bool = False,
        return_assignment: bool = True,
) -> Dict[str, Any]:
    """
    Compute θ(G) by solving a graph coloring ILP using Gurobi.

    Parameters
    ----------
    G : nx.Graph
        Input graph. Either original graph G (default) or already the complement Ḡ.
    is_already_complement : bool, default False
        If False: G is the original graph, solver computes Gc = complement(G) internally
        If True: G is already the complement graph Ḡ, solver uses it directly
    time_limit : int, default 300
        Gurobi time limit in seconds
    mip_gap : float | None
        Relative MIP gap (e.g., 0.0 for optimality, 0.01 for 1%)
    threads : int | None
        Number of threads for Gurobi
    max_colors : int | None
        Optional upper bound on number of colors
    warmstart : dict | list | None
        Optional warmstart assignment (see _parse_warmstart for formats)
    verbose : bool
        Enable/disable Gurobi output
    return_assignment : bool
        Whether to return the color assignment

    Returns
    -------
    dict
        Dictionary containing:
        - 'status': Solver status string
        - 'theta': θ(G) = minimum clique cover number
        - 'chi_complement': χ(Ḡ) (same value as theta)
        - 'assignment': Color assignment if requested {color: [nodes]}
        - 'time': Solve time in seconds
        - 'gap': MIP gap achieved
        - 'n_nodes': Number of nodes
        - 'n_edges': Number of edges in original graph
        - 'n_edges_complement': Number of edges in complement
        - 'error': Error message if applicable
    """

    # Check for Gurobi availability
    if gp is None or GRB is None:
        return {
            'status': 'error',
            'theta': None,
            'chi_complement': None,
            'assignment': None,
            'time': None,
            'gap': None,
            'n_nodes': G.number_of_nodes(),
            'n_edges': G.number_of_edges(),
            'n_edges_complement': None,
            'error': 'gurobipy not available',
        }

    try:
        # Basic properties
        n = G.number_of_nodes()

        # Handle empty graph
        if n == 0:
            return {
                'status': 'optimal',
                'theta': 0,
                'chi_complement': 0,
                'assignment': {} if return_assignment else None,
                'time': 0.0,
                'gap': 0.0,
                'n_nodes': 0,
                'n_edges': 0,
                'n_edges_complement': 0,
            }

        nodelist = list(G.nodes())

        # Determine which graph to color based on is_already_complement flag
        if is_already_complement:
            # G is already the complement, use it directly
            Gc = G
            m_original = None  # We don't know the original edge count
            m = G.number_of_edges()
        else:
            # G is the original graph, compute complement
            m = G.number_of_edges()
            Gc = nx.complement(G)
            m_original = m
            m = Gc.number_of_edges()

        mc = Gc.number_of_edges()

        # Determine upper bound on colors
        if max_colors is not None:
            H = max(1, min(n, max_colors))
        else:
            # Use degree in Gc + 1 as upper bound (Brooks' theorem)
            degmax_c = max((d for _, d in Gc.degree()), default=0)
            H = min(n, degmax_c + 1)

        # Node index mapping
        idx_of = {v: i for i, v in enumerate(nodelist)}

        # Parse warmstart
        ws_map = _parse_warmstart(Gc, nodelist, warmstart)

        # Select anchor node for symmetry breaking (highest degree in Gc)
        v0 = None
        if n > 0 and mc > 0:
            v0 = max(Gc.degree, key=lambda x: x[1])[0]
        elif n > 0:
            v0 = nodelist[0]

        # Create Gurobi model
        model = gp.Model('theta_via_coloring_complement')

        # Set parameters
        if not verbose:
            model.Params.OutputFlag = 0
        if time_limit is not None:
            model.Params.TimeLimit = time_limit
        if mip_gap is not None:
            model.Params.MIPGap = mip_gap
        if threads is not None:
            model.Params.Threads = threads

        # Variables
        # x[v,i] = 1 if node v has color i
        # y[i] = 1 if color i is used
        x = model.addVars(n, H, vtype=GRB.BINARY, name="x")
        y = model.addVars(H, vtype=GRB.BINARY, name="y")

        # Objective: minimize number of colors used
        model.setObjective(gp.quicksum(y[i] for i in range(H)), GRB.MINIMIZE)

        # Constraint 1: Each node gets exactly one color
        for v_idx in range(n):
            model.addConstr(
                gp.quicksum(x[v_idx, i] for i in range(H)) == 1,
                name=f"one_color_{v_idx}"
            )

        # Constraint 2: Adjacent nodes in Gc cannot have the same color
        for u, v in Gc.edges():
            u_idx = idx_of[u]
            v_idx = idx_of[v]
            for i in range(H):
                model.addConstr(
                    x[u_idx, i] + x[v_idx, i] <= 1,
                    name=f"edge_{u_idx}_{v_idx}_color_{i}"
                )

        # Constraint 3: Color i can only be assigned if y[i] = 1
        for v_idx in range(n):
            for i in range(H):
                model.addConstr(
                    x[v_idx, i] <= y[i],
                    name=f"use_color_{v_idx}_{i}"
                )

        # Symmetry breaking 1: y[i] >= y[i+1]
        for i in range(H - 1):
            model.addConstr(y[i] >= y[i + 1], name=f"symmetry_{i}")

        # Symmetry breaking 2: Fix highest-degree node to color 0
        if v0 is not None:
            v0_idx = idx_of[v0]
            model.addConstr(x[v0_idx, 0] == 1, name="anchor_color_0")

        # Apply warmstart if provided
        if ws_map:
            for v in nodelist:
                if v in ws_map:
                    v_idx = idx_of[v]
                    color = ws_map[v]
                    if 0 <= color < H:
                        x[v_idx, color].Start = 1.0
                        # Set other colors to 0
                        for i in range(H):
                            if i != color:
                                x[v_idx, i].Start = 0.0

            # Set y variables based on used colors
            used_colors = set(ws_map.values())
            for i in range(H):
                y[i].Start = 1.0 if i in used_colors else 0.0

        # Optimize
        start_time = time.time()
        model.optimize()
        solve_time = time.time() - start_time

        # Extract results
        status_map = {
            GRB.OPTIMAL: 'optimal',
            GRB.TIME_LIMIT: 'time_limit',
            GRB.INFEASIBLE: 'infeasible',
            GRB.INF_OR_UNBD: 'inf_or_unbounded',
            GRB.UNBOUNDED: 'unbounded',
            GRB.USER_OBJ_LIMIT: 'obj_limit',
            GRB.INTERRUPTED: 'interrupted',
        }
        status = status_map.get(model.Status, f'unknown_{model.Status}')

        # Get solution if feasible
        theta = None
        assignment = None
        gap = None

        if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.USER_OBJ_LIMIT]:
            if model.SolCount > 0:
                theta = int(round(model.ObjVal))
                gap = model.MIPGap if hasattr(model, 'MIPGap') else None

                if return_assignment:
                    # Extract color assignment
                    assignment = {}
                    for v_idx in range(n):
                        v = nodelist[v_idx]
                        for i in range(H):
                            if x[v_idx, i].X > 0.5:
                                if i not in assignment:
                                    assignment[i] = []
                                assignment[i].append(v)
                                break

        return {
            'status': status,
            'theta': theta,
            'chi_complement': theta,  # Same value
            'assignment': assignment,
            'time': solve_time,
            'gap': gap,
            'n_nodes': n,
            'n_edges': m_original if not is_already_complement else None,
            'n_edges_complement': mc,
            'is_already_complement': is_already_complement,
        }

    except Exception as e:
        return {
            'status': 'error',
            'theta': None,
            'chi_complement': None,
            'assignment': None,
            'time': None,
            'gap': None,
            'n_nodes': G.number_of_nodes() if G else 0,
            'n_edges': G.number_of_edges() if G else 0,
            'n_edges_complement': None,
            'error': str(e),
        }


# Convenience function for backwards compatibility
def solve_ilp_direct_on_complement(
        Gc: nx.Graph,
        **kwargs
) -> Dict[str, Any]:
    """
    Convenience function that directly colors the given graph Gc (assumed to be complement).

    This is equivalent to calling solve_ilp_clique_cover(Gc, is_already_complement=True, **kwargs)
    """
    return solve_ilp_clique_cover(Gc, is_already_complement=True, **kwargs)

"""
#OLD Version von Til, hat folgende Probleme: 
# Bezeichner‑Konflikt / Zielgröße unklar
#      Rückgabe/Feld chromatic_number, während die WP‑Aufgabe θ(G) (Clique‑Cover‑Zahl) verlangt.
#      Keine klare Note, ob auf G (Cover) oder auf Ḡ (Coloring) optimiert wird → Verwechslungsgefahr zwischen χ(Ḡ) und θ(G).
# Komplement-Behandlung inkonsistent
#      An einer Stelle wird Ḡ extern (im Wrapper/anderen Skripten) gebildet, an anderer Stelle nicht → Doppelarbeit und Risiken für falsche Ergebnisse.
# Konflikt‑Constraints potenziell falsch/uneindeutig
#     In Coloring‑Modellen müssen für Kanten in Ḡ die Gleichfarbigkeit verboten werden:
#     xu,i+xv,i≤1
#     Im alten Code gab es Hinweise auf xu,i+xv,i≤yi/ fehlende klare Trennung → kann zulässige Färbungen erlauben, die eigentlich verboten sind.
# Symmetrie nicht gebrochen
#     Keine Ordnung der Farben yi≥yi+1, kein „Anker“‑Knoten → unnötiger Suchraum, längere Laufzeiten.
# Rückgabe nicht „klar“
#      Nur eine Zahl/Feld (z.B chromatic_number), keine saubere Struktur mit theta, assignment, gap, time etc.
#      Erschwert Integration in add_ground_truth.py/Auswertung.
# Fehlende/harte Solver‑Parameter
#     Kein (oder fest verdrahteter) TimeLimit, MIPGap, Threads, keine kontrollierte Verbosität → schlechter reproduzierbar, schwer zu debuggen.
# 
# Kein Warmstart‑Interface
#     Heuristische Lösunge like our Chalupa approach werden nicht als Startlösung gesetzt werden → verschenktes Performance‑Potenzial.
#     ist für den Vergleich zwischen Chalupa und ILP auch großer Quatsch, aber möglicherweise für später 1 gutes Tool for best Performance 
# Label‑Stabilität
#      Kein sauberer Umgang mit Original‑Knotenlabels in der Rückgabe (Color→Knotenlisten) → frickelige Nachnutzung.

# ----------------------------------------------------------
# Hauptfunktion: Löse Vertex Clique Coloring mit ILP
# (Assignment Model gemäß Mutzel, Folie 24)
# ----------------------------------------------------------
def solve_ilp_clique_cover(G):
    V = list(G.nodes())           # Liste aller Knoten
    E = list(G.edges())           # Liste aller Kanten
    n = len(V)                    # Anzahl Knoten
    H = n                         # Obergrenze für Farben (max. eine pro Knoten)

    # Neues Gurobi-Modell erstellen
    model = gp.Model("clique_coloring")
    model.setParam("OutputFlag", 0)  # Keine Konsolenausgabe von Gurobi

    # Binärvariablen: x[v, i] = 1, wenn Knoten v Farbe i erhält
    x = model.addVars(V, range(H), vtype=GRB.BINARY)

    # Binärvariablen: w[i] = 1, wenn Farbe i überhaupt verwendet wird
    w = model.addVars(range(H), vtype=GRB.BINARY)

    # (1) Jeder Knoten bekommt genau eine Farbe
    for v in V:
        model.addConstr(gp.quicksum(x[v, i] for i in range(H)) == 1)

    # (2) Benachbarte Knoten dürfen nicht dieselbe Farbe haben
    for (u, v) in E:
        for i in range(H):
            model.addConstr(x[u, i] + x[v, i] <= w[i])

    # (3) Farbe i darf nur verwendet werden (w[i] = 1), wenn sie auch zugewiesen wird
    for i in range(H):
        model.addConstr(w[i] <= gp.quicksum(x[v, i] for v in V))

    # (4) Symmetriebrechung: wenn Farbe i verwendet wird, muss auch Farbe i-1 verwendet worden sein
    for i in range(1, H):
        model.addConstr(w[i] <= w[i - 1])

    # Ziel: Minimale Anzahl verwendeter Farben (entspricht Cliqueanzahl)
    model.setObjective(gp.quicksum(w[i] for i in range(H)), GRB.MINIMIZE)

    # Optimierung starten
    model.optimize()

    # Wenn optimale Lösung gefunden, extrahiere Resultat
    if model.status == GRB.OPTIMAL:
        # chromatic_number = Anzahl verwendeter Farben (d.h. Cliquen)
        chromatic_number = int(sum(w[i].X for i in range(H)))

        # Zuweisung der Farbe (bzw. Clique) für jeden Knoten
        coloring = {v: [i for i in range(H) if x[v, i].X > 0.5][0] for v in V}

        return {
            "chromatic_number": chromatic_number,
            "coloring": coloring,
            "n_nodes": len(V),
            "n_edges": len(E),
        }
    else:
        return {"error": "Keine optimale Lösung gefunden."}

# ----------------------------------------------------------
# Beispiel: Graph laden und ILP lösen
# ----------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_chalupa.py <path_to_graph_file>")
        print("Example: python test_chalupa.py test_graphs/curated/graph_50593.txt")
        sys.exit()

    graph_path = sys.argv[1]

    try:
        G = txt_to_networkx(graph_path)
        result = solve_ilp_clique_cover(G)

        print("Anzahl Knoten:", result["n_nodes"])
        print("Anzahl Kanten:", result["n_edges"])
        print(result)
    except Exception as e:
        print(f"An exception occured: {e}")
"""