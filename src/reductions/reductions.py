from typing import Tuple, List, Union, Any
import time
import logging
import networkx as nx
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def apply_isolated_vertex_reduction(G) -> Tuple[nx.Graph, bool, list, int]:
    """
    Removes isolated vertices from the graph G.
    Returns:
        - Modified graph (in-place)
        - Boolean indicating if any change occurred
        - List of removed isolated vertices
        - Addition to the vertex clique cover count due to this reduction
    """
    changed = False
    VCC_addition = 0
    removed = []
    for v in list(G.nodes()):
        if G.degree(v) == 0:
            G.remove_node(v)
            removed.append(v)
            VCC_addition += 1
            changed = True
    return G, changed, removed, VCC_addition


def neighbourhood_is_crossing_independent(G: nx.Graph, v) -> bool:
    """
    Checks if the external neighborhoods of all pairs of neighbors of v
    are crossing-independent.

    That is, for every pair of neighbors (u, w) of v, and for every
    a ∈ N(u)\\{v}, b ∈ N(w)\\{v}, the edge (a, b) must exist.
    """
    neighbors = list(G.neighbors(v))

    # check all unordered pairs of neighbors
    for i, u in enumerate(neighbors):
        u_ext = set(G.neighbors(u)) - {v}
        for w in neighbors[i+1:]:
            w_ext = set(G.neighbors(w)) - {v}

            # check crossing-independence condition
            for a in u_ext:
                for b in w_ext:
                    if not G.has_edge(a, b):
                        return False

    return True



def apply_degree_two_folding(G: nx.Graph) -> Tuple[nx.Graph, bool, List[Tuple[str, str, str]], int]:
    """
    Applies degree-2 folding reduction to the graph G.
    Returns:
        - Reduced graph G
        - Boolean flag indicating if any folding happened
        - A list of folds: (v, u, w) tuples to help with solution reconstruction
        - Addition to the vertex clique cover count due to this reduction
    """
    VCC_addition = 0
    changed = False
    folds = []
    for v in list(G.nodes()):
        if G.degree(v) == 2:
            neighbors = list(G.neighbors(v))
            if len(neighbors) != 2:
                continue
            u, w = neighbors
            if G.has_edge(u, w):
                continue  # Folding only applies if u and w are not connected

            if not neighbourhood_is_crossing_independent(G, v):
                continue

            # Get external neighbors of u and w (excluding v)
            u_neighbors = set(G.neighbors(u)) - {v}
            w_neighbors = set(G.neighbors(w)) - {v}
            new_neighbors = (u_neighbors | w_neighbors)

            # Determine VCC addition based on neighborhood structure
            u_private_neighbors = u_neighbors - w_neighbors
            w_private_neighbors = w_neighbors - u_neighbors

            if not new_neighbors:
                # Case: Isolated P3 (u-v-w). VCC(P3)=2, VCC(G')=0. Diff=2.
                VCC_addition = 2
            elif u_private_neighbors and w_private_neighbors:
                # Case: Both u and w have private external neighbors, creating a "shortcut". Diff=2.
                VCC_addition = 2
            else:
                # Default case (e.g., P4, diamond graph). Diff=1.
                VCC_addition = 1

            # Add new vertex x representing folded structure (if it's not an isolated P3)
            if new_neighbors:
                x = f"fold_{v}"
                G.add_node(x)
                for n in new_neighbors:
                    G.add_edge(x, n)

            # Remove v, u, w
            G.remove_nodes_from([v, u, w])

            folds.append((v, u, w))
            changed = True
            return G, changed, folds, VCC_addition # Only one fold per call for consistency


    return G, changed, folds, VCC_addition



def apply_twin_folding_or_removal(G: nx.Graph) -> Tuple[nx.Graph, bool, List[Tuple[str, str, str, List[str]]], int]:
    """
    Applies the Twin Folding Reduction for foldable twins or the Twin Removal Reduction (false twins with independent neighborhood).
    Returns:
        - Modified graph
        - Boolean indicating if any change occurred
        - List of (u, v, w, x, y, new_node/None (in case of removal)) tuples for reconstruction
        - Addition to the vertex clique cover count due to this reduction
    """
    VCC_addition = 0
    changed = False
    folded_twins = []
    nodes = list(G.nodes())

    for i in range(len(nodes)):
        u = nodes[i]
        if G.degree(u) != 3:
            continue
        w, x, y = set(G.neighbors(u))
        neighbours_to_check = (set(G.neighbors(w)) & set(G.neighbors(x)) & set(G.neighbors(y))) - {u}
        v_found = None
        for v in neighbours_to_check:
            if G.degree(v) == 3:
                v_found = v
                break
        if v_found is None:
            continue
        v = v_found

        if not neighbourhood_is_crossing_independent(G, u):
            continue # Ensure the crossing independent condition

        if G.has_edge(w, x) or G.has_edge(w, y) or G.has_edge(x, y):
            nodes_to_remove = {u, v, w, x, y}
            G.remove_nodes_from(nodes_to_remove)
            folded_twins.append((u, v, w, x, y, "removal"))  # Mark as removed, no folding
            VCC_addition = 2
            changed = True
            return G, changed, folded_twins, VCC_addition # Only apply one per call for consistency

        # Twin folding is safe
        new_node = f"{u}_{v}_twin_folded"
        G.add_node(new_node)
        for neighbor in (set(G.neighbors(w)) | set(G.neighbors(x)) | set(G.neighbors(y))) - {u, v}:
            G.add_edge(new_node, neighbor)
        nodes_to_remove = {u, v, w, x, y}
        G.remove_nodes_from(nodes_to_remove)
        folded_twins.append((u, v, w, x, y, new_node))
        VCC_addition = 3
        changed = True
        return G, changed, folded_twins, VCC_addition  # Only apply one per call for consistency

    return G, changed, folded_twins, VCC_addition


def apply_domination_reduction(G: nx.Graph) -> Tuple[nx.Graph, bool, List[Tuple[str, str]], int]:
    """
    Applies the Domination Reduction.
    If v dominates u (i.e., N[v] ⊇ N[u]), then v can be safely removed.

    Returns:
        - Modified graph (in-place)
        - Whether any change occurred
        - List of (dominator, dominated) pairs removed
        - Addition to the vertex clique cover count due to this reduction
    """
    VCC_addition = 0
    changed = False
    dominated = []
    nodes = list(G.nodes())

    for i in range(len(nodes)):
        u = nodes[i]
        if u not in G:
            continue
        Nu_closed = (set(G.neighbors(u)) | {u})
        for v in list(G.neighbors(u)):
            if v not in G:
                continue
            Nv_closed = (set(G.neighbors(v)) | {v})
            if Nu_closed.issubset(Nv_closed):
                # v dominates u, so remove v (the dominator)
                G.remove_node(v)
                dominated.append((v, u))  # v dominates u
                changed = True
                return G, changed, dominated, VCC_addition  # Only one per call for safety

    return G, changed, dominated, VCC_addition


def maximal_independent_set_from_matching(G: nx.Graph) -> set:
    M = nx.max_weight_matching(G, maxcardinality=True)
    matched_nodes = {u for edge in M for u in edge}

    I = set(G.nodes()) - matched_nodes
    remaining = deque(G.nodes())
    seen = set(I)  # nodes already in I or blocked

    while remaining:
        v = remaining.popleft()
        if v in seen:
            continue
        if not any(nbr in I for nbr in G.neighbors(v)):
            I.add(v)
            seen.add(v)
            for nbr in G.neighbors(v):
                seen.add(nbr)  # block neighbors
    return I


def apply_crown_reduction(G: nx.Graph) -> Tuple[nx.Graph, bool, List[Any], int]:
    """
    Applies the Crown Reduction rule to the graph for the Vertex Clique Cover problem.

    Returns:
        - Modified graph (in-place)
        - Whether a reduction was applied
        - List of tuples (I, H, M, unmatched_I)
        - Addition to the vertex clique cover count due to this reduction
    """
    VCC_addition = 0
    changed = False
    crown_sets = []

    try:
        I = maximal_independent_set_from_matching(G)
        I = {v for v in I if G.degree(v) > 0}  # exclude isolated vertices
        if not I:
            return G, False, [], VCC_addition

        # iteratively prune I to keep only vertices with neighbors in H
        prev_I = None
        while prev_I != I:
            prev_I = set(I)
            H = set()
            for v in I:
                H.update(G.neighbors(v))
            H -= I
            I = {v for v in I if any(nbr in H for nbr in G.neighbors(v))}

        H = set()
        for v in I:
            H.update(G.neighbors(v))
        H -= I
        if not H:
            return G, False, [], VCC_addition

        if not H:
            return G, False, [], VCC_addition

        # build bipartite graph B=(H ∪ I, E')
        B = nx.Graph()
        B.add_nodes_from(H, bipartite=0)
        B.add_nodes_from(I, bipartite=1)
        for h in H:
            for i in I:
                if G.has_edge(h, i):
                    B.add_edge(h, i)

        # maximum matching in between H and I
        M = nx.bipartite.maximum_matching(B, top_nodes=H)
        matched_pairs = [(u, v) for u, v in M.items() if u in H and v in I]

        # check if all vertices in H are matched
        if len(matched_pairs) == len(H):
            matched_I = {v for _, v in matched_pairs}
            unmatched_I = I - matched_I

            # Remove H and I from G
            G.remove_nodes_from(H)
            G.remove_nodes_from(I)

            crown_sets.append((list(I), list(H), matched_pairs, list(I - matched_I)))
            VCC_addition += len(I)  # add full I to clique cover number
            changed = True

    except Exception as e:
        print(f"[Warning] Crown reduction failed: {e}")

    return G, changed, crown_sets, VCC_addition


def apply_all_reductions(G, verbose: bool = True, timing: bool = True) -> Tuple[nx.Graph, List[Tuple[str, Union[list, str]]], int]:
    """
    Applies all reductions iteratively until no further reductions can be applied.
    Returns:
        - Reduced graph G
        - Trace of applied reductions
        - Total addition to the vertex clique cover count due to reductions
    """

    reductions = [
        apply_isolated_vertex_reduction,
        apply_degree_two_folding,
        apply_twin_folding_or_removal,
        apply_domination_reduction,
        apply_crown_reduction
    ]
    trace = []
    VCC_total_addition = 0
    round_number = 1

    outer_did_change = True
    while outer_did_change:
        outer_did_change = False
        for reduction in reductions:
            inner_did_change = True
            while inner_did_change:
                if verbose:
                    logger.info(f"\n--- Reduction Round {round_number} ({reduction.__name__}) ---")
                start = time.time() if timing else None
                G, inner_did_change, details, VCC_addition = reduction(G)
                end = time.time() if timing else None
                if inner_did_change:
                    outer_did_change = True
                    logger.info(f"Applied {reduction.__name__} with details: {details}. VCC addition: {VCC_addition}")
                    if verbose:
                        logger.info(f"Applied {reduction.__name__}: {details}")
                        if timing:
                            if start is not None and end is not None:
                                logger.info(f"Time: {end - start:.4f}s")
                    trace.append((reduction.__name__, details))
                    VCC_total_addition += VCC_addition
                    round_number += 1
    return G, trace, VCC_total_addition
