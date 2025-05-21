import networkx as nx
from typing import Set, List

def compute_lower_bound(G: nx.Graph) -> Set[int]:
    """
    Computes a maximal independent set in G using a linear-time reduction-based algorithm.
    This set can be used as a lower bound for the clique cover problem.
    """
    G = G.copy()
    I = set()
    S = []  # stack

    def delete_vertex(u):
        if u in G:
            G.remove_node(u)

    def degree_one_reduction():
        for u in list(G.nodes):
            if G.degree(u) == 1:
                S.append(u)
                delete_vertex(u)
                break

    def degree_two_path_reduction():
        for u in list(G.nodes):
            if G.degree(u) == 2:
                path = find_maximal_degree_two_path(G, u)
                if is_cycle(G, path):
                    delete_vertex(u)
                else:
                    v1, vl = path[0], path[-1]
                    neighbors_v1 = list(G.neighbors(v1))
                    neighbors_vl = list(G.neighbors(vl))
                    v = neighbors_v1[0] if neighbors_v1 else None
                    w = neighbors_vl[0] if neighbors_vl else None

                    if v == w:
                        delete_vertex(v)
                    else:
                        if len(path) % 2 == 1:
                            if v and w and G.has_edge(v, w):
                                delete_vertex(v)
                                delete_vertex(w)
                            else:
                                for node in path[1:]:
                                    delete_vertex(node)
                                if v and w:
                                    G.add_edge(v1, w)
                                S.extend(reversed(path[1:]))
                        else:
                            for node in path:
                                delete_vertex(node)
                            if v and w and not G.has_edge(v, w):
                                G.add_edge(v, w)
                            S.extend(reversed(path))
                break

    def inexact_reduction():
        u = max(G.degree, key=lambda x: x[1])[0]
        S.append(u)
        delete_vertex(u)

    def find_maximal_degree_two_path(G, start):
        path = [start]
        current = start
        while True:
            neighbors = [n for n in G.neighbors(current) if G.degree(n) == 2 and n not in path]
            if not neighbors:
                break
            next_node = neighbors[0]
            path.append(next_node)
            current = next_node
        return path

    def is_cycle(G, path):
        return len(path) > 2 and G.has_edge(path[0], path[-1])

    while len(G) > 0:
        if any(G.degree(n) == 1 for n in G.nodes):
            degree_one_reduction()
        elif any(G.degree(n) == 2 for n in G.nodes):
            degree_two_path_reduction()
        else:
            inexact_reduction()

    # Build maximal independent set
    visited = set()
    while S:
        u = S.pop()
        if u not in visited:
            I.add(u)
            visited.update(G.neighbors(u))
            visited.add(u)

    return I
