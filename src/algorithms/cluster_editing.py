import networkx as nx
import gurobipy as gp
from gurobipy import GRB
from itertools import combinations


def solve_cluster_editing_ilp(
    graph: nx.Graph, weights: dict[tuple[int, int], int] | None
) -> tuple[set[tuple[int, int]], float]:
    """
    Args:
        graph: Input graph
        weights: Edge weights for pairs of nodes (positive = edge exists, negative = non-edge). If None, edges are assigned +1 and non-edges -1 weight

    Returns:
        modifications: set of edge modifications on the original graph
        cost: cost of the edge modifications
    """

    nodes = list(graph.nodes())
    n = len(nodes)
    if n <= 1:
        return set(), 0.0

    model = gp.Model("cluster_editing")
    model.Params.OutputFlag = 0
    modifications = set()

    # Create all pairs
    pairs = {(min(u, v), max(u, v)) for u, v in combinations(nodes, 2)}

    # Decision variable: x[i, j] = 1 if edge (i, j) exists in solution
    x = model.addVars(pairs, vtype=GRB.BINARY, name="x")
    obj_expr = gp.LinExpr()

    print("Weights:", weights)

    for i, j in pairs:
        if weights is None:
            w = 1 if graph.has_edge(i, j) else -1
        else:
            w = weights.get((i, j), 1 if graph.has_edge(i, j) else -1)

        # Cost of modification
        if w > 0:
            obj_expr += w * (1 - x[(i, j)])
        else:
            obj_expr += (-w) * x[(i, j)]

    model.setObjective(obj_expr, GRB.MINIMIZE)

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                u, v, w = nodes[i], nodes[j], nodes[k]
                pair_uv = (min(u, v), max(u, v))
                pair_vw = (min(v, w), max(v, w))
                pair_uw = (min(u, w), max(u, w))
                model.addConstr(x[pair_uv] + x[pair_vw] - x[pair_uw] <= 1)
                model.addConstr(x[pair_uv] - x[pair_vw] + x[pair_uw] <= 1)
                model.addConstr(-x[pair_uv] + x[pair_vw] + x[pair_uw] <= 1)
    model.optimize()

    if model.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
        raise RuntimeError(f"Optimization failed with status {model.Status}")

    # Extract modifications from solution
    modifications = set()
    for u, v in pairs:
        solution_has_edge = x[(u, v)].X > 0.5
        original_has_edge = graph.has_edge(u, v)

        if solution_has_edge != original_has_edge:
            modifications.add((u, v))

    return modifications, model.ObjVal
