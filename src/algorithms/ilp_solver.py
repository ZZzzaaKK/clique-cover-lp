"""
Integer Linear Programming (ILP) formulation for the vertex clique coloring problem.
"""

import gurobipy as gp
from gurobipy import GRB
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import txt_to_networkx

# ----------------------------------------------------------
# Hauptfunktion: Löse Vertex Clique Coloring mit ILP
# (Assignment Model gemäß Mutzel, Folie 24)
# ----------------------------------------------------------
def solve_ilp_clique_cover(G, time_limit=60, require_optimal=False):
    V = list(G.nodes())           # Liste aller Knoten
    E = list(G.edges())           # Liste aller Kanten
    n = len(V)                    # Anzahl Knoten
    H = n                         # Obergrenze für Farben (max. eine pro Knoten)

    # Neues Gurobi-Modell erstellen
    model = gp.Model("clique_coloring")
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", time_limit)

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
            model.addConstr(x[u, i] + x[v, i] <= 1)

    # (3) Wenn ein Knoten eine Farbe erhält, muss diese Farbe als verwendet markiert werden
    for v in V:
        for i in range(H):
            model.addConstr(x[v, i] <= w[i])

    # (4) Symmetriebrechung: wenn Farbe i verwendet wird, muss auch Farbe i-1 verwendet worden sein
    for i in range(1, H):
        model.addConstr(w[i] <= w[i - 1])

    # Ziel: Minimale Anzahl verwendeter Farben (entspricht Cliqueanzahl)
    model.setObjective(gp.quicksum(w[i] for i in range(H)), GRB.MINIMIZE)

    # Optimierung starten
    model.optimize()

    # Resultat auswerten
    solution_found = model.solCount > 0
    is_optimal = model.status == GRB.OPTIMAL

    if is_optimal or (solution_found and not require_optimal):
        chromatic_number = int(round(model.ObjVal))
        coloring = {v: [i for i in range(H) if x[v, i].X > 0.5][0] for v in V}

        return {
            "chromatic_number": chromatic_number,
            "coloring": coloring,
            "n_nodes": len(V),
            "n_edges": len(E),
            "optimal": is_optimal
        }
    else:
        return {"error": f"Solver finished with status {model.status} and found {model.solCount} solutions."}

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
