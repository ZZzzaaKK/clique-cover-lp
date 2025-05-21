import networkx as nx
from reductions import apply_all_reductions
from lower_bound_linear import compute_lower_bound

def branch_and_reduce(G: nx.Graph, depth=0):
    G, trace = apply_all_reductions(G, verbose=False, timing=False)
    
    if G.number_of_nodes() == 0:
        return 0  # base case

    lb = compute_lower_bound(G)
    
    # Choose branching vertex (e.g., highest degree)
    v = max(G.degree, key=lambda x: x[1])[0]

    G1 = G.copy()
    G1.remove_node(v)
    res1 = branch_and_reduce(G1, depth + 1)

    G2 = G.copy()
    neighbors = list(G2.neighbors(v))
    G2.remove_nodes_from(neighbors + [v])
    res2 = 1 + branch_and_reduce(G2, depth + 1)

    return min(res1, res2)
