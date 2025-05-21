from typing import Tuple, List, Union
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

def apply_degree_two_folding(G) -> Tuple[nx.Graph, bool, list]:
    changed = False
    folded = []
    for v in list(G.nodes()):
        if G.degree(v) == 2:
            neighbors = list(G.neighbors(v))
            if len(neighbors) < 2:
                continue
            u, w = neighbors
            if not G.has_edge(u, w):
                G.add_edge(u, w)
            G.remove_node(v)
            folded.append(v)
            changed = True
    return G, changed, folded

def apply_twin_removal(G) -> Tuple[nx.Graph, bool, list]:
    changed = False
    removed_pairs = []
    seen = set()
    nodes = list(G.nodes())
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            u, v = nodes[i], nodes[j]
            if (u, v) in seen:
                continue
            if G.degree(u) == 3 and G.degree(v) == 3 and set(G.neighbors(u)) == set(G.neighbors(v)):
                N_uv = set(G.neighbors(u))
                if any(G.has_edge(x, y) for x in N_uv for y in N_uv if x != y):
                    G.remove_node(u)
                    G.remove_node(v)
                    removed_pairs.append((u, v))
                    changed = True
                    seen.add((u, v))
    return G, changed, removed_pairs

def apply_domination_reduction(G) -> Tuple[nx.Graph, bool, list]:
    changed = False
    dominated = []
    nodes = list(G.nodes())
    for i in range(len(nodes)):
        u = nodes[i]
        for j in range(len(nodes)):
            v = nodes[j]
            if u == v:
                continue
            if set(G.neighbors(u)).issubset(set(G.neighbors(v))):
                G.remove_node(v)
                dominated.append((u, v))
                changed = True
                break
    return G, changed, dominated

def apply_twin_folding(G) -> Tuple[nx.Graph, bool, list]:
    changed = False
    folded_twins = []
    seen = set()
    nodes = list(G.nodes())
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            u, v = nodes[i], nodes[j]
            if (u, v) in seen:
                continue
            if G.degree(u) == 3 and G.degree(v) == 3 and set(G.neighbors(u)) == set(G.neighbors(v)):
                N = list(G.neighbors(u))
                if len(N) == 3:
                    w, x, y = N
                    new_node = f"{u}_{v}_folded"
                    common_neighbors = list(set(G.neighbors(w)) & set(G.neighbors(x)) & set(G.neighbors(y)))
                    G.add_node(new_node)
                    for cn in common_neighbors:
                        G.add_edge(new_node, cn)
                    G.remove_node(u)
                    G.remove_node(v)
                    folded_twins.append((u, v, new_node))
                    changed = True
                    seen.add((u, v))
    return G, changed, folded_twins

def apply_crown_reduction(G) -> Tuple[nx.Graph, bool, list]:
    changed = False
    crown_sets = []
    try:
        maximal_independent_sets = [set(nx.algorithms.approximation.maximum_independent_set(G)) for _ in range(3)]
    except Exception as e:
        logger.warning(f"Failed to compute maximal independent set: {e}")
        return G, False, []
    for I in maximal_independent_sets:
        N_I = set()
        for i in I:
            N_I.update(G.neighbors(i))
        H = G.subgraph(N_I).copy()
        M = list(nx.algorithms.matching.max_weight_matching(H, maxcardinality=True))
        if len(M) >= len(I):
            unmatched = I - set(u for u, _ in M) - set(v for _, v in M)
            G.remove_nodes_from(I)
            G.remove_nodes_from(N_I)
            crown_sets.append((list(I), list(N_I), list(M), list(unmatched)))
            changed = True
            break
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
