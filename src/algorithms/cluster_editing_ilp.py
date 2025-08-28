# src/algorithms/cluster_editing_ilp.py
from typing import Dict, Tuple, List, Set, Any, Optional
import math
import networkx as nx


def solve_cluster_editing_ilp(
    graph: nx.Graph,
    weights: Dict[Tuple[int, int], float],
    time_limit: Optional[float] = None,
    mip_gap: Optional[float] = None,
    threads: Optional[int] = None,
    gurobi_params: Optional[Dict[str, Any]] = None,
):
    """
    Solve (weighted) Cluster Editing via correlation clustering ILP using Gurobi.

    Variable s_ij in {0,1}: 1 = u,v getrennt (cut), 0 = gleiche Cluster.
    Ziel: Minimiere "Disagreements":
        - Für positive Gewichte (Kante):     bezahlt man bei Trennung (s_ij = 1)
        - Für negative Gewichte (Nicht-Kante): bezahlt man bei Zusammenlegen (s_ij = 0)

    Objekt: sum_{i<j} [ pos_w*s_ij + neg_w*(1 - s_ij) ]
            = sum neg_w + sum (pos_w - neg_w) * s_ij
    Nebenbedingungen: Dreiecksungleichungen (Transitivität)

    Returns:
      clusters: List[Set[int]]
      stats: dict (obj_value, runtime, gap, status, num_vars, num_constr)
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as e:
        raise RuntimeError(
            "gurobipy ist nicht installiert oder keine Lizenz aktiv. "
            "Installiere 'gurobipy' und aktiviere die Gurobi-Lizenz."
        ) from e

    nodes = list(graph.nodes())
    n = len(nodes)

    pairs: List[Tuple[int, int]] = []
    for i in range(n):
        u = nodes[i]
        for j in range(i + 1, n):
            v = nodes[j]
            a, b = (min(u, v), max(u, v))
            pairs.append((a, b))

    # ILP-Modell
    m = gp.Model("cluster_editing_ilp")
    if time_limit is not None:
        m.Params.TimeLimit = float(time_limit)
    if mip_gap is not None:
        m.Params.MIPGap = float(mip_gap)
    if threads is not None:
        m.Params.Threads = int(threads)
    m.Params.OutputFlag = 0
    if gurobi_params:
        for k, v in gurobi_params.items():
            setattr(m.Params, k, v)

    s = m.addVars(pairs, vtype=GRB.BINARY, name="s")

    # --- Objective ---
    # Robustheit: clippe nicht-finite Gewichte
    WMAX = 1e6

    const_term = 0.0
    lin_expr = gp.LinExpr()

    for (a, b) in pairs:
        # Default-Gewicht, falls nicht hinterlegt
        default_w = 1.0 if graph.has_edge(a, b) else -1.0
        w = float(weights.get((a, b), default_w))

        # Sanitize: ersetze NaN/Inf durch große, endliche Werte
        if not math.isfinite(w):
            w = WMAX if w > 0 else -WMAX

        # Clipping zur Sicherheit
        if w > WMAX:
            w = WMAX
        elif w < -WMAX:
            w = -WMAX

        pos_w = max(0.0, w)
        neg_w = max(0.0, -w)

        const_term += neg_w
        lin_expr += (pos_w - neg_w) * s[(a, b)]

    m.setObjective(const_term + lin_expr, GRB.MINIMIZE)

    # Dreiecksungleichungen
    for i in range(n):
        ui = nodes[i]
        for j in range(i + 1, n):
            uj = nodes[j]
            aij = (min(ui, uj), max(ui, uj))
            for k in range(j + 1, n):
                uk = nodes[k]
                aik = (min(ui, uk), max(ui, uk))
                ajk = (min(uj, uk), max(uj, uk))
                m.addConstr(s[aij] <= s[aik] + s[ajk])
                m.addConstr(s[aik] <= s[aij] + s[ajk])
                m.addConstr(s[ajk] <= s[aij] + s[aik])

    m.optimize()

    status = m.Status
    from gurobipy import GRB  # type: ignore
    if status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED):
        raise RuntimeError(f"Gurobi endete mit Status {status}")

    # Cluster konstruieren: s_ij = 0 => gleiches Cluster
    H = nx.Graph()
    H.add_nodes_from(nodes)
    for (a, b) in pairs:
        if s[(a, b)].X <= 0.5:
            H.add_edge(a, b)

    clusters = [set(c) for c in nx.connected_components(H)]

    stats = {
        "obj_value": m.ObjVal if m.SolCount > 0 else None,
        "runtime": m.Runtime,
        "gap": getattr(m, "MIPGap", None) if status != GRB.OPTIMAL else 0.0,
        "status": status,
        "num_vars": m.NumVars,
        "num_constr": m.NumConstrs,
    }
    return clusters, stats
