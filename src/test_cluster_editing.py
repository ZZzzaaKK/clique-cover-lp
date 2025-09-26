#!/usr/bin/env python3
import networkx as nx
from algorithms.cluster_editing import kernelize_edge_cuts, solve_cluster_editing_ilp
import ast
import os


def parse_test_graph_file(filepath):
    """
    Parse a test graph file with expected modifications

    Unweighted Format:
    1: 2 3
    2: 1 4 5
    ...

    Modifications: [{(1, 2), (4, 5)}, {(2, 3), (4, 5)}]

    Weighted Format:
    1: 2 3
    2: 1 4 5
    ...
    Weights:
    (1, 2): 5.0
    (2, 3): -2.0

    # Calculated by hand
    Modifications: [{(1, 2), (4, 5)}, {(2, 3), (4, 5)}]

    Returns:
        tuple: (graph, expected_modifications_list, weights_dict_or_none)
    """
    graph = nx.Graph()
    expected_modifications = []

    with open(filepath, "r") as f:
        lines = f.readlines()

    parsing_adjacency = True

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            parsing_adjacency = False
            continue

        if parsing_adjacency and ":" in line:
            parts = line.split(":")
            node = int(parts[0])
            if len(parts) > 1 and parts[1].strip():
                neighbors = [int(x) for x in parts[1].split()]
                for neighbor in neighbors:
                    if node < neighbor:  # Add edge only once
                        graph.add_edge(node, neighbor)

        elif line.startswith("Modifications:"):
            modifications_str = line[len("Modifications:") :].strip()
            try:
                modifications_list = ast.literal_eval(modifications_str)
                expected_modifications = [
                    set(tuple(sorted(edge)) for edge in mod)
                    for mod in modifications_list
                ]
            except Exception as e:
                print(
                    f"Warning: Could not parse modifications: {modifications_str}, error: {e}"
                )

    weights_dict = None

    # Look for weights section
    for line in lines:
        line = line.strip()
        if line.startswith("# Weights"):
            weights_dict = {}
            continue
        elif weights_dict is not None and ":" in line and line.startswith("("):
            # Parse weight line: (1, 2): 5.0
            try:
                colon_idx = line.index(":")
                edge_str = line[:colon_idx].strip()
                weight_str = line[colon_idx + 1 :].strip()
                edge = ast.literal_eval(edge_str)
                weight = float(weight_str)
                # Normalize edge order
                edge = tuple(sorted(edge))
                weights_dict[edge] = weight
            except Exception as e:
                print(f"Warning: Could not parse weight line: {line} - Error: {e}")

    return graph, expected_modifications, weights_dict


def validate_solution(
    modifications, expected_modifications_list, cost, expected_cost=None
):
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


def convert_similarity_to_weights(similarity_scores, threshold=None, method="linear"):
    """
    Convert similarity scores to cluster editing weights

    For cluster editing:
    - Positive weight = cost to REMOVE an existing edge
    - Negative weight = cost to ADD a missing edge

    Args:
        similarity_scores: dict of {(u,v): score} where higher score = more similar
        threshold: similarity threshold. If None, uses median
        method: 'linear', 'exp', or 'threshold'

    Returns:
        dict: {(u,v): weight} suitable for cluster editing ILP
    """
    if not similarity_scores:
        return {}

    scores = list(similarity_scores.values())
    if threshold is None:
        threshold = sorted(scores)[len(scores) // 2]  # median

    weights = {}

    if method == "threshold":
        # Simple threshold: high similarity = high cost to remove
        for edge, score in similarity_scores.items():
            if score >= threshold:
                weights[edge] = score / max(scores)  # Normalize to [0,1]
            else:
                weights[edge] = -(threshold - score) / max(
                    scores
                )  # Negative for dissimilar

    elif method == "linear":
        # Linear scaling around threshold
        max_score = max(scores)
        min_score = min(scores)
        score_range = max_score - min_score

        for edge, score in similarity_scores.items():
            # Convert to [-1, 1] range around threshold
            normalized = 2 * (score - threshold) / score_range
            weights[edge] = normalized

    elif method == "exp":
        # Exponential scaling emphasizes differences
        import math

        max_score = max(scores)

        for edge, score in similarity_scores.items():
            # Exponential decay from high similarity
            weights[edge] = math.exp((score - threshold) / (max_score - threshold))
            if score < threshold:
                weights[edge] = -weights[edge]

    return weights


def load_similarity_graph(filepath, score_col="score"):
    """
    Load a complete similarity graph from TSV format like RF02246.tsv

    Args:
        filepath: path to TSV file with columns idA, idB, score
        score_col: name of score column

    Returns:
        tuple: (complete_graph, similarity_scores_dict)
    """
    import pandas as pd

    df = pd.read_csv(filepath, sep="\t")

    # Get unique nodes
    nodes = set(df["idA"].unique()) | set(df["idB"].unique())

    # Create complete graph
    graph = nx.Graph()
    graph.add_nodes_from(nodes)

    # Add all edges with scores
    similarity_scores = {}
    for _, row in df.iterrows():
        u, v = row["idA"], row["idB"]
        if u != v:  # Skip self-loops
            edge = tuple(sorted([u, v]))
            graph.add_edge(u, v)
            similarity_scores[edge] = row[score_col]

    return graph, similarity_scores


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
    weights = {(0, 1): 10, (1, 2): 3, (0, 2): -5}

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
    test_dir = "test_graphs/curated/cluster_editing/weighted"

    if not os.path.exists(test_dir):
        print(f"Test directory {test_dir} not found, skipping curated tests")
        return

    # Recursively find all .txt files in subdirectories
    test_files = []
    for root, dirs, files in os.walk(test_dir):
        for filename in files:
            if filename.endswith(".txt"):
                filepath = os.path.join(root, filename)
                # Get relative path from test_dir for cleaner display
                rel_path = os.path.relpath(filepath, test_dir)
                test_files.append((filepath, rel_path))

    # Sort by relative path for consistent ordering
    test_files.sort(key=lambda x: x[1])

    for filepath, rel_path in test_files:
        print(f"Testing {rel_path}:")

        try:
            graph, expected_modifications_list, weights = parse_test_graph_file(
                filepath
            )
            print(f"Original edges: {list(graph.edges())}")
            print(f"Expected solutions: {expected_modifications_list}")

            modifications, cost = solve_cluster_editing_ilp(graph, weights)
            print(f"ILP solution: {modifications}")
            print(f"Cost: {cost}")

            is_valid, solution_idx = validate_solution(
                modifications, expected_modifications_list, cost
            )

            if is_valid:
                print(f"✓ PASS: Matches expected solution {solution_idx}")
            else:
                print("✗ FAIL: Does not match any expected solution")
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
            print(f"Error testing {filepath}: {e}")
            print()


def test_curated_graphs_reduced():
    """Test on hand-crafted examples with known optimal solutions"""
    test_dir = "test_graphs/curated/cluster_editing"

    if not os.path.exists(test_dir):
        print(f"Test directory {test_dir} not found, skipping curated tests")
        return

    # Recursively find all .txt files in subdirectories
    test_files = []
    for root, dirs, files in os.walk(test_dir):
        for filename in files:
            if filename.endswith(".txt"):
                filepath = os.path.join(root, filename)
                # Get relative path from test_dir for cleaner display
                rel_path = os.path.relpath(filepath, test_dir)
                test_files.append((filepath, rel_path))

    # Sort by relative path for consistent ordering
    test_files.sort(key=lambda x: x[1])

    for filepath, rel_path in test_files:
        print(f"Testing {rel_path}:")

        try:
            graph, expected_modifications_list, weights = parse_test_graph_file(
                filepath
            )
            print(f"Original edges: {list(graph.edges())}")
            print(f"Expected solutions: {expected_modifications_list}")

            (reduced_graph, reduced_weights, remaining_k, applied_modifications) = (
                kernelize_edge_cuts(graph, weights)
            )
            print(f"Remaining k: {remaining_k}")
            print(f"Applied modifications: {applied_modifications}")
            modifications, cost = solve_cluster_editing_ilp(
                reduced_graph, reduced_weights
            )
            print(f"ILP solution: {modifications}")
            print(f"Cost: {cost}")

            is_valid, solution_idx = validate_solution(
                modifications, expected_modifications_list, cost
            )

            if is_valid:
                print(f"✓ PASS: Matches expected solution {solution_idx}")
            else:
                print("✗ FAIL: Does not match any expected solution")
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
            print(f"Error testing {filepath}: {e}")
            print()


def test_similarity_data():
    """Test cluster editing on similarity data like RF02246.tsv"""
    similarity_file = "test_graphs/rfam/RF02246.tsv"

    if not os.path.exists(similarity_file):
        print(f"Similarity file {similarity_file} not found, skipping test")
        return

    print("Testing RF02246.tsv similarity data:")

    # Load the complete similarity graph
    try:
        graph, similarity_scores = load_similarity_graph(similarity_file)
        print(
            f"Loaded graph with {graph.number_of_nodes} nodes and {graph.number_of_edges} edges"
        )
        print(
            f"Score range: {min(similarity_scores.values()):.0f} - {max(similarity_scores.values()):.0f}"
        )

        # Convert to cluster editing weights using median threshold
        scores = sorted(similarity_scores.values())
        median_score = scores[len(scores) // 2]
        print(f"Using median threshold: {median_score:.0f}")

        weights = convert_similarity_to_weights(
            similarity_scores, threshold=median_score, method="linear"
        )

        # Show some example weights
        print(
            "\nExample edge weights (positive = costly to remove, negative = costly to add):"
        )
        edge_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]
        for edge, weight in edge_weights:
            orig_score = similarity_scores[edge]
            print(f"  {edge}: weight={weight:.3f} (original_score={orig_score:.0f})")

        print("\nRunning cluster editing ILP...")
        modifications, cost = solve_cluster_editing_ilp(graph, weights)

        print(f"Found {len(modifications)} edge modifications with cost {cost:.3f}")

        # Apply modifications to see clusters
        result_graph = graph.copy()
        for u, v in modifications:
            if result_graph.has_edge(u, v):
                result_graph.remove_edge(u, v)
            else:
                result_graph.add_edge(u, v)

        clusters = list(nx.connected_components(result_graph))
        print(f"Resulting clusters: {len(clusters)} clusters")
        for i, cluster in enumerate(sorted(clusters, key=len, reverse=True)):
            print(f"  Cluster {i + 1}: {len(cluster)} nodes")

    except Exception as e:
        print(f"Error testing similarity data: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # test_triangle_graph()
    # test_path_graph()
    # test_weighted()
    test_curated_graphs_reduced()
    # test_curated_graphs()
    # test_similarity_data()
