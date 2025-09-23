#!/usr/bin/env python3
import networkx as nx
from algorithms.cluster_editing import solve_cluster_editing_ilp
import ast
import os

def parse_test_graph_file(filepath):
    """
    Parse a test graph file with expected modifications

    Format:
    1: 2 3
    2: 1 4 5
    ...
    # Calculated by hand
    Modifications: {[(1, 2), (4, 5)], [(2, 3), (4, 5)]}

    Returns:
        tuple: (graph, expected_modifications_list)
    """
    graph = nx.Graph()
    expected_modifications = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    parsing_adjacency = True

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            parsing_adjacency = False
            continue

        if parsing_adjacency and ':' in line:
            parts = line.split(':')
            node = int(parts[0])
            if len(parts) > 1 and parts[1].strip():
                neighbors = [int(x) for x in parts[1].split()]
                for neighbor in neighbors:
                    if node < neighbor:  # Add edge only once
                        graph.add_edge(node, neighbor)

        elif line.startswith('Modifications:'):
            modifications_str = line[len('Modifications:'):].strip()
            try:
                modifications_list = ast.literal_eval(modifications_str)
                expected_modifications = [set(tuple(sorted(edge)) for edge in mod) for mod in modifications_list]
            except:
                print(f"Warning: Could not parse modifications: {modifications_str}")

    return graph, expected_modifications

def validate_solution(modifications, expected_modifications_list, cost, expected_cost=None):
    """
    Check if the found solution matches one of the expected optimal solutions

    Args:
        modifications: set of edge modifications from ILP solver
        expected_modifications_list: list of sets of expected optimal modifications
        cost: cost returned by ILP solver
        expected_cost: expected optimal cost (if known)

    Returns:
        tuple: (is_valid, matching_solution_index)
    """
    normalized_modifications = set(tuple(sorted(edge)) for edge in modifications)

    for i, expected in enumerate(expected_modifications_list):
        if normalized_modifications == expected:
            if expected_cost is not None and abs(cost - expected_cost) > 1e-6:
                print(f"Warning: Cost mismatch. Expected {expected_cost}, got {cost}")
            return True, i

    return False, -1

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

def test_curated_graphs():
    """Test on hand-crafted examples with known optimal solutions"""
    test_dir = "test_graphs/curated/cluster_editing/unweighted"

    if not os.path.exists(test_dir):
        print(f"Test directory {test_dir} not found, skipping curated tests")
        return

    for filename in sorted(os.listdir(test_dir)):
        if filename.endswith('.txt'):
            filepath = os.path.join(test_dir, filename)
            print(f"Testing {filename}:")

            try:
                graph, expected_modifications_list = parse_test_graph_file(filepath)
                print(f"Original edges: {list(graph.edges())}")
                print(f"Expected solutions: {expected_modifications_list}")

                modifications, cost = solve_cluster_editing_ilp(graph, None)
                print(f"ILP solution: {modifications}")
                print(f"Cost: {cost}")

                is_valid, solution_idx = validate_solution(modifications, expected_modifications_list, cost)

                if is_valid:
                    print(f"✓ PASS: Matches expected solution {solution_idx}")
                else:
                    print(f"✗ FAIL: Does not match any expected solution")
                    print(f"Expected one of: {expected_modifications_list}")

                # Show resulting clusters
                result = graph.copy()
                for u, v in modifications:
                    if result.has_edge(u, v):
                        result.remove_edge(u, v)
                    else:
                        result.add_edge(u, v)
                clusters = list(nx.connected_components(result))
                print(f"Resulting clusters: {clusters}")
                print()

            except Exception as e:
                print(f"Error testing {filename}: {e}")
                print()

if __name__ == "__main__":
    test_triangle_graph()
    test_path_graph()
    test_weighted()
    test_curated_graphs()
