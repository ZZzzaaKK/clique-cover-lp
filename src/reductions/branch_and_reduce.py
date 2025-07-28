import networkx as nx
from reductions.reductions import apply_all_reductions
from lower_bound_linear import compute_lower_bound

def branch_and_reduce(G: nx.Graph, depth=0, best=float('inf')):
    G, trace = apply_all_reductions(G, verbose=False, timing=False)

    if G.number_of_nodes() == 0:
        return 0  # Base case

    lb = len(compute_lower_bound(G))
    if lb >= best:
        return best  # Prune this branch

    # Choose vertex to branch on
    v = max(G.degree, key=lambda x: x[1])[0]

    # Case 1: Don't remove neighbors, only remove v
    G1 = G.copy()
    G1.remove_node(v)
    res1 = branch_and_reduce(G1, depth + 1, best)

    # Case 2: Cover v and its neighbors in one clique
    G2 = G.copy()
    neighbors = list(G2.neighbors(v))
    G2.remove_nodes_from(neighbors + [v])
    res2 = 1 + branch_and_reduce(G2, depth + 1, min(best, res1))

    return min(res1, res2)
