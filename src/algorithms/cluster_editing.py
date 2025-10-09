import networkx as nx
import gurobipy as gp
from gurobipy import GRB
from itertools import combinations


def kernelize_edge_cuts(
    graph: nx.Graph,
    weights: dict[tuple[int, int], int] | None = None,
    k: int | None = None,
) -> tuple[nx.Graph, dict[tuple[int, int], int], int | float, set[tuple[int, int]]]:
    """
    Apply edge cuts kernelization for cluster editing.

    Let v be a vertex satisfying 2δ(v) + γ(N[v]) < |N[v]|, then do the following:
    1. add edges to connect N(v) into a clique and decrease k accordingly;
    2. for each neighbor x of N[v], if w(E(x, N[v])) ≤ |N[v]|/2, remove edges E(x, N[v]) and decrease k accordingly;
    3. if one neighbor x of N[v] survives, merge N[v] appropriately and decrease k accordingly.

    Args:
        graph: Input graph
        weights: Edge weights for pairs of nodes (positive = edge exists, negative = non-edge)
        k: Budget for modifications (if None, unlimited)

    Returns:
        reduced_graph: Graph after kernelization
        reduced_weights: Updated weights
        remaining_k: Remaining budget after reductions
        applied_modifications: Set of modifications applied during kernelization
    """
    if weights is None:
        weights = {}
        for u, v in combinations(graph.nodes(), 2):
            edge_key = (min(u, v), max(u, v))
            weights[edge_key] = 1 if graph.has_edge(u, v) else -1

    # Create a copy of the graph and weights to work with
    working_graph = graph.copy()
    working_weights = weights.copy()
    remaining_k = k if k is not None else float("inf")
    applied_modifications = set()

    def get_weight(u: int, v: int) -> int:
        """Get weight between two vertices"""
        edge_key = (min(u, v), max(u, v))
        return working_weights.get(edge_key, 1 if working_graph.has_edge(u, v) else -1)

    def compute_cut_cost(vertex_set: set[int]) -> int:
        """Compute γ(X): total cost of the cut of vertex set X"""
        complement = set(working_graph.nodes()) - vertex_set
        cut_cost = 0

        for u in vertex_set:
            for v in complement:
                weight = get_weight(u, v)
                if weight > 0 and working_graph.has_edge(u, v):
                    cut_cost += weight  # Cost of removing existing positive edge
                elif weight < 0 and not working_graph.has_edge(u, v):
                    cut_cost += abs(weight)  # Cost of adding missing negative edge
        return cut_cost
    """
    #problematisch, weil: summiert werden Beträge aller Paare, also auch Nicht-Kanten (negative Gewichte) als positive Beiträge.
    def compute_edge_weight_sum(x: int, vertex_set: set[int]) -> int:
        #Compute w(E(x, N[v])): sum of weights of edges between x and vertex_set
        weight_sum = 0
        for v in vertex_set:
            if x != v:
                weight_sum += abs(get_weight(x, v))
        return weight_sum
    """
    
    def compute_edge_weight_sum(x: int, vertex_set: set[int]) -> int:
        # Summe der Gewichte nur über EXISTIERENDE Kanten von x in vertex_set, Nicht-Kanten bleiben außen vor
        s = 0
        for u in vertex_set:
            if u != x and working_graph.has_edge(x, u):
                w = get_weight(x, u)
                # Bei Deiner Konvention: vorhandene Kante => w > 0
                if w > 0:
                    s += w  # i.d.R. 1
        return s
   
    
    def add_edge_with_cost(u: int, v: int) -> int:
        """Add edge and return cost"""
        edge_key = (min(u, v), max(u, v))
        if working_graph.has_edge(u, v):
            return 0  # No cost if edge already exists

        weight = get_weight(u, v)
        cost = abs(weight) if weight < 0 else 0
        working_graph.add_edge(u, v)
        applied_modifications.add(edge_key)
        working_weights[edge_key] = abs(weight)
        return cost

    def remove_edge_with_cost(u: int, v: int) -> int:
        """Remove edge and return cost"""
        edge_key = (min(u, v), max(u, v))
        if not working_graph.has_edge(u, v):
            return 0  # No cost if edge doesn't exist

        weight = get_weight(u, v)
        cost = abs(weight) if weight > 0 else 0
        working_graph.remove_edge(u, v)
        applied_modifications.add(edge_key)
        working_weights[edge_key] = -abs(weight)
        return cost

    progress_made = True
    while progress_made:
        progress_made = False

        for v in list(working_graph.nodes()):
            # Get neighborhoods
            open_neighbors = set(working_graph.neighbors(v))  # N(v)
            closed_neighbors = open_neighbors | {v}  # N[v]

            # Calculate δ(v): degree of v
            delta_v = len(open_neighbors)

            # Calculate γ(N[v]): total cost of cut of closed neighborhood
            gamma_closed = compute_cut_cost(closed_neighbors)
            print(f"Gamme_closed: {gamma_closed}")
            print(f"Delta_v: {delta_v}")
            print(f"Length of closed_neighbors: {len(closed_neighbors)}")

            # Check kernelization condition: 2δ(v) + γ(N[v]) < |N[v]|
            if 2 * delta_v + gamma_closed >= len(closed_neighbors):
                continue

            progress_made = True

            # Step 1: Add edges to connect N(v) into a clique
            clique_cost = 0
            for u1 in open_neighbors:
                for u2 in open_neighbors:
                    if u1 < u2:
                        clique_cost += add_edge_with_cost(u1, u2)

            if remaining_k != float("inf"):
                remaining_k -= clique_cost
                if remaining_k < 0:
                    break

            # Step 2: For each neighbor x of N[v], check if w(E(x, N[v])) ≤ |N[v]|/2
            neighbors_of_closed = set()
            for node in closed_neighbors:
                neighbors_of_closed.update(working_graph.neighbors(node))

            # Remove vertices that are already in N[v]
            external_neighbors = neighbors_of_closed - closed_neighbors

            surviving_neighbors = set()
            for x in external_neighbors:
                edge_weight_sum = compute_edge_weight_sum(x, closed_neighbors)

                if edge_weight_sum <= len(closed_neighbors) / 2:
                    # Remove edges E(x, N[v])
                    removal_cost = 0
                    for node in closed_neighbors:
                        if x != node:
                            removal_cost += remove_edge_with_cost(x, node)

                    if remaining_k != float("inf"):
                        remaining_k -= removal_cost
                        if remaining_k < 0:
                            break
                else:
                    surviving_neighbors.add(x)

            # Step 3: If one neighbor x of N[v] survives, merge N[v] appropriately
            if len(surviving_neighbors) == 1:
                x = next(iter(surviving_neighbors))

                # Merge: connect x to all vertices in N(v) (making x part of the clique)
                merge_cost = 0
                for node in open_neighbors:
                    merge_cost += add_edge_with_cost(x, node)

                if remaining_k != float("inf"):
                    remaining_k -= merge_cost
                    if remaining_k < 0:
                        break

            # Break after processing one vertex to restart the search
            break

    print(f"Applied modifications through edge cut reductions: {applied_modifications}")
    return working_graph, working_weights, remaining_k, applied_modifications


def solve_cluster_editing_ilp(
    graph: nx.Graph, weights: dict[tuple[int, int], int] | None, time_limit=60
) -> tuple[set, float, set[tuple[int, int]]]:
    """
    Args:
        graph: Input graph
        weights: Edge weights for pairs of nodes (positive = edge exists, negative = non-edge). If None, edges are assigned +1 and non-edges -1 weight

    Returns:
        clusters: clusters in the final solutions
        cost: cost of the edge modifications
        modifications: set of edge modifications on the original graph
    """

    nodes = list(graph.nodes())
    n = len(nodes)
    if n <= 1:
        return set(nodes), 0.0, set()

    model = gp.Model("cluster_editing")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit
    clusters = set()

    # Create all pairs
    pairs = {(min(u, v), max(u, v)) for u, v in combinations(nodes, 2)}

    # Decision variable: x[i, j] = 1 if edge (i, j) exists in solution
    x = model.addVars(pairs, vtype=GRB.BINARY, name="x")
    obj_expr = gp.LinExpr()

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

    # Extract modifications and clusters from solution
    modifications = set()
    solution_graph = nx.Graph()
    solution_graph.add_nodes_from(nodes)

    for u, v in pairs:
        solution_has_edge = x[(u, v)].X > 0.5
        original_has_edge = graph.has_edge(u, v)

        if solution_has_edge != original_has_edge:
            modifications.add((u, v))

        # Build solution graph for cluster extraction
        if solution_has_edge:
            solution_graph.add_edge(u, v)

    # Extract clusters as connected components
    clusters = {
        frozenset(component) for component in nx.connected_components(solution_graph)
    }

    return clusters, model.ObjVal, modifications
