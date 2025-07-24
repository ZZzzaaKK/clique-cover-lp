"""
Integer Linear Programming (ILP) formulation for the vertex clique coloring problem.
"""

import networkx as nx
import gurobipy as gp
from gurobipy import GRB
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# ----------------------------------------------------------
# Funktion zum Laden eines Graphen aus einer Pickle-Datei
# ----------------------------------------------------------
def load_graph(path):
    if path.endswith(".g6"):
        return nx.read_graph6(path)
    else:
        raise ValueError("Unsupported file format")

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
# (erzeugt mit WP0-Simulator... Pickle)
# ----------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_chalupa.py <path_to_graph_file>")
        print("Example: python test_chalupa.py test_cases/curated/graph_50593.g6")
        sys.exit()

    graph_path = sys.argv[1]

    try:
        G = load_graph(graph_path)
        result = solve_ilp_clique_cover(G)

        print("Anzahl Knoten:", result["n_nodes"])
        print("Anzahl Kanten:", result["n_edges"])
        print(result)
    except Exception as e:
        print(f"An exception occured: {e}")
