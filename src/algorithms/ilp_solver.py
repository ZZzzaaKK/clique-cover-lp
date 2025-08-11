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


import sys
import os
import time
from typing import Dict, Iterable, List, Tuple, Any, Optional

import gurobipy as gp
from gurobipy import GRB
import networkx as nx

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import txt_to_networkx


from typing import Dict, List, Any, Optional, Union
import time
import networkx as nx

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:
    gp = None
    GRB = None


def _parse_warmstart(
    G: nx.Graph,
    nodelist: List[Any],
    warmstart: Any,
) -> Optional[Dict[Any, int]]:
    # Normalize warmstart into {node: color}, or None if invalid
    if warmstart is None:
        return None
    # Case 1: dict node->color or color->list[nodes]
    if isinstance(warmstart, dict):
        keys = set(warmstart.keys())
        node_keys = set(nodelist)
        if keys & node_keys:
            return {node: int(warmstart[node]) for node in nodelist if node in warmstart}
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

"""
    Compute θ(G) by solving a graph coloring ILP on the complement graph Ḡ using Gurobi.

    Parameters
    ----------
    G : nx.Graph
        Original graph whose clique cover number θ(G) is sought.
    time_limit : int, default 300
        Gurobi time limit (seconds).
    mip_gap : float | None
        Optional relative MIP gap target (e.g., 0.0 for optimality, 0.01 for 1%).
    threads : int | None
        Limit threads Gurobi uses. None = solver default.
    max_colors : int | None
        Optional hard upper bound H on the number of colors (cover sets).
        If None, use Δ(Ḡ)+1 (safe upper bound by Brooks/trivial bound) and at most n.
    warmstart : dict | list | None
        Optional warmstart assignment; see module docstring.
    verbose : bool
        If True, Gurobi output is enabled; else suppressed.
    return_assignment : bool
        If True and a feasible solution exists, return color → node-list mapping.

    Returns
    -------
    dict
        See module docstring for schema.
"""

def solve_ilp_clique_cover(
    G: nx.Graph,
    time_limit: int = 300,
    mip_gap: Optional[float] = None,
    threads: Optional[int] = None,
    max_colors: Optional[int] = None,
    warmstart: Optional[Union[Dict[Any, int], Dict[int, List[Any]], List[int]]] = None,
    verbose: bool = False,
    return_assignment: bool = True,
) -> Dict[str, Any]:
    # Handle missing gurobipy
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
    # Basic properties
    n = G.number_of_nodes()
    m = G.number_of_edges()
    nodelist = list(G.nodes())
    Gc = nx.complement(G)
    mc = Gc.number_of_edges()
    # Colors upper bound H
    degmax_c = max((d for _, d in Gc.degree()), default=0)
    default_H = min(n, degmax_c + 1)
    H = max(1, min(n, max_colors)) if max_colors is not None else default_H
    # Map node->index
    idx_of = {v: i for i, v in enumerate(nodelist)}
    # Warmstart mapping
    ws_map = _parse_warmstart(G, nodelist, warmstart)
    # Anchor node selection
    v0 = None
    if n > 0:
        try:
            v0 = max(Gc.degree, key=lambda x: x[1])[0] if mc > 0 else nodelist[0]
        except Exception:
            v0 = nodelist[0]
    # Build model
    model = gp.Model('theta_via_coloring_on_complement')
    if not verbose:
        model.Params.OutputFlag = 0
    if time_limit is not None:
        model.Params.TimeLimit = int(time_limit)
    if mip_gap is not None:
        model.Params.MIPGap = float(mip_gap)
    if threads is not None:
        model.Params.Threads = int(threads)
    # Variables: x[v,i] = 1 if node v has color i; y[i] = 1 if color i is used
    x = model.addVars(n, H, vtype=GRB.BINARY, name="x")
    y = model.addVars(H, vtype=GRB.BINARY, name="y")
    # Each node gets exactly one color
    for v in range(n):
        model.addConstr(gp.quicksum(x[v, i] for i in range(H)) == 1)
    # Linking x to y
    for v in range(n):
        for i in range(H):
            model.addConstr(x[v, i] <= y[i])
    # Conflict constraints on complement edges
    for u, v in Gc.edges():
        iu = idx_of[u]
        iv = idx_of[v]
        for i in range(H):
            model.addConstr(x[iu, i] + x[iv, i] <= 1)
    # Symmetry breaking
    for i in range(H - 1):
        model.addConstr(y[i] >= y[i + 1])
    if v0 is not None:
        model.addConstr(x[idx_of[v0], 0] == 1)
    # Objective: minimize number of colors
    model.setObjective(gp.quicksum(y[i] for i in range(H)), GRB.MINIMIZE)
    # Warmstart assignment
    if ws_map is not None:
        for node, color in ws_map.items():
            if node in idx_of and 0 <= color < H:
                v = idx_of[node]
                x[v, color].Start = 1.0
                for i in range(H):
                    if i != color:
                        x[v, i].Start = 0.0
                y[color].Start = 1.0
        used = set(c for c in ws_map.values() if 0 <= c < H)
        for i in range(H):
            if i not in used:
                y[i].Start = 0.0
    # Solve
    t0 = time.time()
    model.optimize()
    t1 = time.time()
    # Status mapping
    status_code = model.Status
    status_map = {
        GRB.OPTIMAL: 'optimal',
        GRB.TIME_LIMIT: 'time_limit',
        GRB.INFEASIBLE: 'infeasible',
        GRB.UNBOUNDED: 'unbounded',
        GRB.INTERRUPTED: 'interrupted',
    }
    status = status_map.get(status_code, str(status_code))
    # Prepare result dict
    result: Dict[str, Any] = {
        'status': status,
        'theta': None,
        'chi_complement': None,
        'assignment': None,
        'time': getattr(model, 'Runtime', t1 - t0),
        'gap': None,
        'n_nodes': n,
        'n_edges': m,
        'n_edges_complement': mc,
    }
    feasible_statuses = {getattr(GRB, 'OPTIMAL', -1), getattr(GRB, 'SUBOPTIMAL', -2), getattr(GRB, 'TIME_LIMIT', -3)}
    if status_code in feasible_statuses and model.SolCount > 0:
        obj = model.ObjVal if model.SolCount > 0 else None
        if obj is not None:
            theta_val = int(round(obj))
            result['theta'] = theta_val
            result['chi_complement'] = theta_val
        try:
            result['gap'] = float(model.MIPGap)
        except Exception:
            result['gap'] = None
        if return_assignment:
            used_colors: Dict[int, List[Any]] = {i: [] for i in range(H)}
            for node in nodelist:
                v = idx_of[node]
                chosen = None
                for i in range(H):
                    if x[v, i].X > 0.5:
                        chosen = i
                        break
                if chosen is None:
                    chosen = max(range(H), key=lambda i: x[v, i].X)
                used_colors[chosen].append(node)
            assignment: Dict[int, List[Any]] = {}
            for i in range(H):
                try:
                    yi = y[i].X
                except Exception:
                    yi = 0.0
                if yi > 0.5:
                    assignment[i] = used_colors[i]
            result['assignment'] = assignment
    return result


if __name__ == "__main__":
    # Example graph: two triangles connected by one edge
    G = nx.Graph()
    G.add_nodes_from(range(6))
    G.add_edges_from([(0, 1), (1, 2), (0, 2)])
    G.add_edges_from([(3, 4), (4, 5), (3, 5)])
    G.add_edge(2, 3)
    print(solve_ilp_clique_cover(G, time_limit=30, verbose=True))



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