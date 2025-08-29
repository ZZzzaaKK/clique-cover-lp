# src/algorithms/cluster_editing_ilp_fixed.py
from typing import Dict, Tuple, List, Set, Any, Optional
import math
import logging
import networkx as nx

logger = logging.getLogger(__name__)


def solve_cluster_editing_ilp(
        graph: nx.Graph,
        weights: Dict[Tuple[int, int], float],
        time_limit: Optional[float] = 300,
        mip_gap: Optional[float] = None,
        threads: Optional[int] = None,
        gurobi_params: Optional[Dict[str, Any]] = None,
        use_2partition: bool = True,
        max_cutting_planes: int = 1000,
):
    """
    Solve (weighted) Cluster Editing via correlation clustering ILP using Gurobi.
    Implements the complete approach from Grötschel & Wakabayashi including 2-partition inequalities.

    Variable x_ij in {0,1}: 1 = edge exists in solution, 0 = edge does not exist

    Objective: Minimize disagreements with input graph

    Constraints:
    - Triangle inequalities (ensure transitivity)
    - 2-partition inequalities (strengthen LP relaxation)

    Args:
        graph: Input graph
        weights: Edge weights (positive = edge exists, negative = non-edge)
        time_limit: Maximum time in seconds
        mip_gap: MIP gap tolerance
        threads: Number of threads
        gurobi_params: Additional Gurobi parameters
        use_2partition: Whether to use 2-partition cutting planes
        max_cutting_planes: Maximum number of cutting planes per iteration

    Returns:
        clusters: List[Set[int]] - Computed clustering
        stats: dict - Solver statistics
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as e:
        raise RuntimeError(
            "gurobipy is not installed or no license is active. "
            "Install 'gurobipy' and activate the Gurobi license."
        ) from e

    nodes = list(graph.nodes())
    n = len(nodes)

    # Validate input
    if n == 0:
        return [], {'obj_value': 0, 'runtime': 0, 'status': 'OPTIMAL'}

    # Create all pairs
    pairs: List[Tuple[int, int]] = []
    for i in range(n):
        u = nodes[i]
        for j in range(i + 1, n):
            v = nodes[j]
            pairs.append((min(u, v), max(u, v)))

    # Initialize model
    m = gp.Model("cluster_editing_ilp_complete")

    # Set parameters
    if time_limit is not None:
        m.Params.TimeLimit = float(time_limit)
    if mip_gap is not None:
        m.Params.MIPGap = float(mip_gap)
    if threads is not None:
        m.Params.Threads = int(threads)
    m.Params.OutputFlag = 1 #wirft Gurobi-Outputstatusmeldungen oder eben auch nicht

    # Apply additional parameters
    if gurobi_params:
        for k, v in gurobi_params.items():
            setattr(m.Params, k, v)

    # Create variables
    x = m.addVars(pairs, vtype=GRB.BINARY, name="x")

    # Build objective function with weight validation
    WMAX = 1e6  # Maximum allowed weight magnitude
    obj_const = 0.0
    obj_expr = gp.LinExpr()

    for (a, b) in pairs:
        # Get weight with default
        if graph.has_edge(a, b):
            default_w = 1.0  # Edge exists, deletion cost
        else:
            default_w = -1.0  # Edge doesn't exist, insertion cost

        w = float(weights.get((a, b), default_w))

        # Validate and clip weight
        if not math.isfinite(w):
            logger.warning(f"Non-finite weight {w} for edge ({a},{b}), using default {default_w}")
            w = default_w
        w = max(-WMAX, min(WMAX, w))  # Clip to reasonable range

        # Convert to minimization of disagreements
        if w > 0:  # Edge exists in input
            obj_expr += w * (1 - x[(a, b)])  # Pay cost if we delete it
        else:  # Edge doesn't exist in input
            obj_expr += (-w) * x[(a, b)]  # Pay cost if we add it

    m.setObjective(obj_expr, GRB.MINIMIZE)

    # Add triangle inequalities (transitivity constraints)
    for i in range(n):
        ui = nodes[i]
        for j in range(i + 1, n):
            uj = nodes[j]
            pair_ij = (min(ui, uj), max(ui, uj))
            for k in range(j + 1, n):
                uk = nodes[k]
                pair_ik = (min(ui, uk), max(ui, uk))
                pair_jk = (min(uj, uk), max(uj, uk))

                # For any three nodes, we can't have exactly two edges
                # (would create a P3 path, violating transitivity)
                m.addConstr(x[pair_ij] + x[pair_jk] - x[pair_ik] <= 1)
                m.addConstr(x[pair_ij] - x[pair_jk] + x[pair_ik] <= 1)
                m.addConstr(-x[pair_ij] + x[pair_jk] + x[pair_ik] <= 1)

    # Implement cutting plane approach if enabled
    if use_2partition:
        _solve_with_cutting_planes(m, x, nodes, pairs, max_cutting_planes)
    else:
        m.optimize()

    # Check status
    status = m.Status
    if status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED, GRB.SUBOPTIMAL):
        logger.warning(f"Gurobi terminated with status {status}")
        if m.SolCount == 0:
            raise RuntimeError(f"No solution found, status {status}")

    # Extract solution - build graph from edges with x_ij = 1
    H = nx.Graph()
    H.add_nodes_from(nodes)
    for (a, b) in pairs:
        if x[(a, b)].X > 0.5:  # Variable is 1
            H.add_edge(a, b)

    # Connected components form clusters
    clusters = [set(c) for c in nx.connected_components(H)]

    # Collect statistics
    stats = {
        "obj_value": m.ObjVal if m.SolCount > 0 else None,
        "runtime": m.Runtime,
        "gap": m.MIPGap if hasattr(m, 'MIPGap') and status != GRB.OPTIMAL else 0.0,
        "status": status,
        "num_vars": m.NumVars,
        "num_constr": m.NumConstrs,
        "num_clusters": len(clusters),
    }

    return clusters, stats


def _solve_with_cutting_planes(model, x, nodes, pairs, max_cuts_per_round):
    """
    Implement cutting plane approach with 2-partition inequalities.
    Based on Grötschel & Wakabayashi approach.
    """
    from gurobipy import GRB
    import gurobipy as gp

    iteration = 0
    max_iterations = 100

    while iteration < max_iterations:
        # Solve LP relaxation
        model.optimize()

        if model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            break

        # Get current solution
        x_val = {}
        for pair in pairs:
            x_val[pair] = x[pair].X

        # Check if solution is integral
        is_integral = all(abs(v - round(v)) < 1e-6 for v in x_val.values())
        if is_integral:
            logger.info(f"Found integral solution after {iteration} cutting plane iterations")
            break

        # Find violated 2-partition inequalities
        cuts_added = _find_2partition_cuts(model, x, nodes, x_val, max_cuts_per_round)

        if cuts_added == 0:
            # No more cuts found, solve as MIP
            logger.info(f"No violated cuts found after {iteration} iterations, solving as MIP")
            for pair in pairs:
                x[pair].VType = GRB.BINARY
            model.optimize()
            break

        logger.debug(f"Added {cuts_added} cutting planes in iteration {iteration}")
        iteration += 1


def _find_2partition_cuts(model, x, nodes, x_val, max_cuts):
    """
    Find violated 2-partition inequalities.

    2-partition inequality: For disjoint S, T ⊆ V:
    sum_{i∈S, j∈T} x_ij - sum_{i,j∈S} x_ij - sum_{i,j∈T} x_ij ≤ min(|S|, |T|)
    """
    import gurobipy as gp

    cuts_added = 0
    n = len(nodes)

    # Heuristic separation: for each node, try to find violated cuts
    for i_idx, i in enumerate(nodes):
        if cuts_added >= max_cuts:
            break

        # Look at nodes connected to i with fractional values
        W = []
        for j_idx, j in enumerate(nodes):
            if i_idx != j_idx:
                pair = (min(i, j), max(i, j))
                if 0 < x_val[pair] < 1:
                    W.append(j)

        if len(W) < 2:
            continue

        # Try different partitions
        for start_node in W[:min(5, len(W))]:  # Limit trials
            T = {start_node}

            # Greedily build T
            for k in W:
                if k in T or k == i:
                    continue

                # Check if k should be added to T
                score = 0
                for t in T:
                    pair = (min(k, t), max(k, t))
                    score += x_val.get(pair, 0)

                pair_ik = (min(i, k), max(i, k))
                score -= x_val.get(pair_ik, 0)

                if score > 0:
                    T.add(k)

            if len(T) < 2:
                continue

            # Check violation
            S = {i}

            # Calculate LHS of inequality
            lhs = 0

            # sum_{i∈S, j∈T} x_ij
            for s in S:
                for t in T:
                    pair = (min(s, t), max(s, t))
                    lhs += x_val.get(pair, 0)

            # -sum_{i,j∈S} x_ij (only relevant if |S| > 1)
            s_list = list(S)
            for idx1 in range(len(s_list)):
                for idx2 in range(idx1 + 1, len(s_list)):
                    pair = (min(s_list[idx1], s_list[idx2]), max(s_list[idx1], s_list[idx2]))
                    lhs -= x_val.get(pair, 0)

            # -sum_{i,j∈T} x_ij
            t_list = list(T)
            for idx1 in range(len(t_list)):
                for idx2 in range(idx1 + 1, len(t_list)):
                    pair = (min(t_list[idx1], t_list[idx2]), max(t_list[idx1], t_list[idx2]))
                    lhs -= x_val.get(pair, 0)

            # RHS of inequality
            rhs = min(len(S), len(T))

            # Check violation
            if lhs > rhs + 1e-4:  # Small tolerance
                # Add cutting plane
                expr = gp.LinExpr()

                # sum_{i∈S, j∈T} x_ij
                for s in S:
                    for t in T:
                        pair = (min(s, t), max(s, t))
                        expr += x[pair]

                # -sum_{i,j∈S} x_ij
                for idx1 in range(len(s_list)):
                    for idx2 in range(idx1 + 1, len(s_list)):
                        pair = (min(s_list[idx1], s_list[idx2]), max(s_list[idx1], s_list[idx2]))
                        expr -= x[pair]

                # -sum_{i,j∈T} x_ij
                for idx1 in range(len(t_list)):
                    for idx2 in range(idx1 + 1, len(t_list)):
                        pair = (min(t_list[idx1], t_list[idx2]), max(t_list[idx1], t_list[idx2]))
                        expr -= x[pair]

                model.addConstr(expr <= rhs)
                cuts_added += 1

                logger.debug(f"Added 2-partition cut: |S|={len(S)}, |T|={len(T)}, violation={lhs - rhs:.4f}")

    return cuts_added


# Additional utility functions for validation

def validate_clustering(graph: nx.Graph, clusters: List[Set[int]]) -> bool:
    """Validate that clustering is valid (disjoint and covers all nodes)."""
    all_nodes = set(graph.nodes())
    covered = set()

    for cluster in clusters:
        if covered & cluster:
            logger.error("Clusters are not disjoint")
            return False
        covered.update(cluster)

    if covered != all_nodes:
        logger.error("Clusters do not cover all nodes")
        return False

    return True


def calculate_clustering_cost(graph: nx.Graph,
                              weights: Dict[Tuple[int, int], float],
                              clusters: List[Set[int]]) -> float:
    """Calculate the cost of a clustering."""
    cost = 0.0

    # Cost of missing edges within clusters
    for cluster in clusters:
        nodes_in_cluster = list(cluster)
        for i in range(len(nodes_in_cluster)):
            for j in range(i + 1, len(nodes_in_cluster)):
                u, v = nodes_in_cluster[i], nodes_in_cluster[j]
                if not graph.has_edge(u, v):
                    # Need to add edge
                    pair = (min(u, v), max(u, v))
                    cost += abs(weights.get(pair, 1.0))

    # Cost of edges between clusters
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            for u in clusters[i]:
                for v in clusters[j]:
                    if graph.has_edge(u, v):
                        # Need to remove edge
                        pair = (min(u, v), max(u, v))
                        cost += abs(weights.get(pair, 1.0))

    return cost