#!/usr/bin/env python3
import networkx as nx
from algorithms.cluster_editing import solve_cluster_editing_ilp

def test_triangle_graph():
    """Test on a triangle - should either keep as one clique or split into 3 singletons"""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (0, 2)])

    print("Triangle graph test:")
    print(f"Original edges: {list(G.edges())}")

    modifications, cost = solve_cluster_editing_ilp(G, None)
    print(f"Modifications: {modifications}")
    print(f"Cost: {cost}")

    # Apply modifications to see result
    result = G.copy()
    for u, v in modifications:
        if result.has_edge(u, v):
            result.remove_edge(u, v)
        else:
            result.add_edge(u, v)

    clusters = list(nx.connected_components(result))
    print(f"Resulting clusters: {clusters}")
    print()

def test_path_graph():
    """Test on a path P4: 0-1-2-3. Should split into 2 cliques or 4 singletons"""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3)])

    print("Path graph P4 test:")
    print(f"Original edges: {list(G.edges())}")

    modifications, cost = solve_cluster_editing_ilp(G, None)
    print(f"Modifications: {modifications}")
    print(f"Cost: {cost}")

    # Apply modifications
    result = G.copy()
    for u, v in modifications:
        if result.has_edge(u, v):
            result.remove_edge(u, v)
        else:
            result.add_edge(u, v)

    clusters = list(nx.connected_components(result))
    print(f"Resulting clusters: {clusters}")
    print()

def test_weighted():
    """Test with custom weights - make it expensive to remove edge (0,1)"""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2)])

    # High cost to remove (0,1), low cost for others
    weights = {(0, 1): 10, (1, 2): 1}

    print("Weighted test:")
    print(f"Original edges: {list(G.edges())}")
    print(f"Weights: {weights}")

    modifications, cost = solve_cluster_editing_ilp(G, weights)
    print(f"Modifications: {modifications}")
    print(f"Cost: {cost}")

    # Apply modifications
    result = G.copy()
    for u, v in modifications:
        if result.has_edge(u, v):
            result.remove_edge(u, v)
        else:
            result.add_edge(u, v)

    clusters = list(nx.connected_components(result))
    print(f"Resulting clusters: {clusters}")
    print()

if __name__ == "__main__":
    test_triangle_graph()
    test_path_graph()
    test_weighted()
