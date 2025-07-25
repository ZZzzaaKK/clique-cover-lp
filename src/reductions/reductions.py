from typing import Tuple, List, Union, Any
import time
import logging
import networkx as nx

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def is_isolated_vertex(G, v: int) -> bool:
    return G.degree(v) == 0

def apply_isolated_vertex_reduction(G) -> Tuple[nx.Graph, bool, list]:
    changed = False
    removed = []
    for v in list(G.nodes()):
        if is_isolated_vertex(G, v):
            G.remove_node(v)
            removed.append(v)
            changed = True
    return G, changed, removed


def apply_degree_two_folding(G: nx.Graph) -> Tuple[nx.Graph, bool, List[Tuple[str, str, str]]]:
    """
    Applies degree-2 folding reduction to the graph G.
    Returns:
        - Reduced graph G
        - Boolean flag indicating if any folding happened
        - A list of folds: (v, u, w) tuples to help with solution reconstruction
    """
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

            # Get external neighbors of u and w (excluding v)
            u_neighbors = set(G.neighbors(u)) - {v}
            w_neighbors = set(G.neighbors(w)) - {v}
            new_neighbors = (u_neighbors | w_neighbors)

            # Add new vertex x representing folded structure
            x = f"fold_{v}"
            G.add_node(x)
            for n in new_neighbors:
                G.add_edge(x, n)

            # Remove v, u, w
            G.remove_nodes_from([v, u, w])

            folds.append((v, u, w))  # for reconstructing solution
            changed = True
            break  # fold one node at a time for safety

    return G, changed, folds


def apply_twin_removal(G: nx.Graph) -> Tuple[nx.Graph, bool, List[Tuple[str, str, set]]]:
    """
    Applies the Twin Removal Reduction.
    Returns:
        - Modified graph (in-place)
        - Whether any change was made
        - List of removed twin pairs with their neighborhood
    """
    changed = False
    removed_info = []

    nodes = list(G.nodes())
    for i in range(len(nodes)):
        u = nodes[i]
        if G.degree(u) != 3:
            continue
        for j in range(i + 1, len(nodes)):
            v = nodes[j]
            if G.degree(v) != 3:
                continue
            if G.has_edge(u, v):
                continue  # They must be false twins (non-adjacent)

            Nu = set(G.neighbors(u))
            Nv = set(G.neighbors(v))
            if Nu != Nv:
                continue

            # Check if any edge exists between neighbors
            has_internal_edge = any(
                G.has_edge(x, y) for x in Nu for y in Nu if x != y
            )

            if has_internal_edge:
                # Remove u, v, and their neighbors (N[u,v])
                to_remove = Nu | {u, v}
                G.remove_nodes_from(to_remove)
                removed_info.append((u, v, Nu))
                changed = True
                return G, changed, removed_info  # Apply one reduction per call

    return G, changed, removed_info


def apply_twin_folding(G: nx.Graph) -> Tuple[nx.Graph, bool, List[Tuple[str, str, str, List[str]]]]:
    """
    Applies the Twin Folding Reduction for foldable twins (false twins with independent neighborhood).
    Returns:
        - Modified graph
        - Boolean indicating if any change occurred
        - List of (u, v, new_node, neighbors) tuples for reconstruction
    """
    changed = False
    folded_twins = []
    nodes = list(G.nodes())

    for i in range(len(nodes)):
        u = nodes[i]
        if G.degree(u) != 3:
            continue
        for j in range(i + 1, len(nodes)):
            v = nodes[j]
            if G.degree(v) != 3:
                continue
            if G.has_edge(u, v):
                continue  # must be false twins

            neighbors_u = set(G.neighbors(u))
            neighbors_v = set(G.neighbors(v))
            if neighbors_u != neighbors_v:
                continue

            N = list(neighbors_u)
            if any(G.has_edge(x, y) for i, x in enumerate(N) for y in N[i + 1:]):
                continue  # neighbors are not independent

            # Twin folding is safe
            new_node = f"{u}_{v}_folded"
            G.add_node(new_node)
            for neighbor in N:
                G.add_edge(new_node, neighbor)
            G.remove_node(u)
            G.remove_node(v)
            folded_twins.append((u, v, new_node, N))
            changed = True
            return G, changed, folded_twins  # Only apply one per call for consistency

    return G, changed, folded_twins


def apply_domination_reduction(G: nx.Graph) -> Tuple[nx.Graph, bool, List[Tuple[str, str]]]:
    """
    Applies the Domination Reduction.
    If v dominates u (i.e., N[v] ⊇ N[u]), then v can be safely removed.

    Returns:
        - Modified graph (in-place)
        - Whether any change occurred
        - List of (dominated, dominator) pairs removed
    """
    changed = False
    dominated = []
    nodes = list(G.nodes())

    for i in range(len(nodes)):
        u = nodes[i]
        Nu_closed = set(G.neighbors(u)) | {u}
        for j in range(len(nodes)):
            if i == j:
                continue
            v = nodes[j]
            Nv_closed = set(G.neighbors(v)) | {v}
            if Nu_closed.issubset(Nv_closed):
                # v dominates u, so remove v
                G.remove_node(v)
                dominated.append((u, v))  # v dominates u
                changed = True
                return G, changed, dominated  # Only one per call for safety

    return G, changed, dominated


def apply_crown_reduction(G: nx.Graph) -> Tuple[nx.Graph, bool, List[Any]]:
    """
    Applies the Crown Reduction rule to the graph for the Vertex Clique Cover problem.

    Returns:
        - Modified graph (in-place)
        - Whether a reduction was applied
        - List of tuples (I, H, M, unmatched_I)
    """
    changed = False
    crown_sets = []

    try:
        # Try multiple independent sets heuristically
        for _ in range(3):
            I = set(nx.algorithms.approximation.maximum_independent_set(G))
            if not I:
                continue

            H = set()
            for i in I:
                H.update(G.neighbors(i))

            # Only keep neighbors of I (i.e., H)
            H = H - I  # Just in case
            if not H:
                continue

            # Build bipartite graph between H and I
            B = nx.Graph()
            B.add_nodes_from(H, bipartite=0)
            B.add_nodes_from(I, bipartite=1)

            for h in H:
                for i in I:
                    if G.has_edge(h, i):
                        B.add_edge(h, i)

            # Get maximum cardinality matching (from H to I)
            M = list(nx.bipartite.maximum_matching(B, top_nodes=H).items())

            # Filter the actual H→I edges only
            M = [(u, v) for u, v in M if u in H and v in I]

            if len(M) == len(H):
                matched_I = {v for _, v in M}
                unmatched_I = I - matched_I

                # Remove crown nodes
                G.remove_nodes_from(H)
                G.remove_nodes_from(I)

                crown_sets.append((list(I), list(H), M, list(unmatched_I)))
                changed = True
                break  # Only apply one crown reduction at a time
    except Exception as e:
        print(f"[Warning] Crown reduction failed: {e}")

    return G, changed, crown_sets


def apply_all_reductions(G, verbose: bool = True, timing: bool = True) -> Tuple[nx.Graph, List[Tuple[str, Union[list, str]]]]:
    reductions = [
        apply_isolated_vertex_reduction,
        apply_degree_two_folding,
        apply_twin_removal,
        apply_twin_folding,
        apply_domination_reduction,
        apply_crown_reduction
    ]
    trace = []
    changed = True
    round_number = 1
    while changed:
        changed = False
        if verbose:
            logger.info(f"\n--- Reduction Round {round_number} ---")
        for reduction in reductions:
            start = time.time() if timing else None
            G, did_change, details = reduction(G)
            end = time.time() if timing else None
            if did_change:
                if verbose:
                    logger.info(f"Applied {reduction.__name__}: {details}")
                    if timing:
                        if start is not None and end is not None:
                            logger.info(f"Time: {end - start:.4f}s")
                trace.append((reduction.__name__, details))
                changed = True
                break
        round_number += 1
    return G, trace
