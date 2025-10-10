import argparse
import networkx as nx
from pathlib import Path


def convert_dissimilarity_to_weights(
    dissimilarity_scores, threshold=None, method="linear"
):
    """
    Convert dissimilarity scores to cluster editing weights

    For cluster editing:
    - Positive weight = cost to REMOVE an existing edge
    - Negative weight = cost to ADD a missing edge

    Args:
        dissimilarity_scores: dict of {(u,v): score} where LOWER score = more similar
        threshold: dissimilarity threshold. If None, uses median
        method: 'linear', 'exp', or 'threshold'

    Returns:
        dict: {(u,v): weight} suitable for cluster editing ILP
    """
    if not dissimilarity_scores:
        return {}

    scores = list(dissimilarity_scores.values())

    weights = {}

    if threshold is None:
        threshold = sorted(scores)[len(scores) // 2]  # median

    if method == "threshold":
        # Simple threshold: low dissimilarity = high cost to remove
        for edge, score in dissimilarity_scores.items():
            if score <= threshold:  # INVERTED: low dissimilarity = similar
                weights[edge] = (threshold - score) / max(scores)  # Normalize to [0,1]
            else:
                weights[edge] = -(score - threshold) / max(
                    scores
                )  # Negative for dissimilar

    elif method == "linear":
        # Linear scaling around threshold
        max_score = max(scores)
        min_score = min(scores)
        score_range = max_score - min_score

        for edge, score in dissimilarity_scores.items():
            # Convert to [-1, 1] range around threshold
            # INVERTED: low dissimilarity (similar) → positive weight
            normalized = 2 * (threshold - score) / score_range
            weights[edge] = normalized

    elif method == "exp":
        # Exponential scaling emphasizes differences
        import math

        max_score = max(scores)
        min_score = min(scores)

        for edge, score in dissimilarity_scores.items():
            # Exponential emphasis for low dissimilarity (similar pairs)
            # INVERTED: low dissimilarity → positive weight
            normalized = (
                (threshold - score) / (threshold - min_score)
                if score <= threshold
                else (threshold - score) / (max_score - threshold)
            )
            weights[edge] = (
                math.exp(normalized) if score <= threshold else -math.exp(-normalized)
            )

    return weights


def load_similarity_graph(filepath, score_col="score", shift_col="shifts"):
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
            similarity_scores[edge] = (
                row[score_col] if int(row[shift_col]) > 0 else -row[score_col]
            )

    return graph, similarity_scores


def save_graph_with_weights(graph, weights, filepath):
    """
    Save graph and weights to file in the specified format.
    Maps node IDs to sequential numbers starting from 1.
    Only includes edges with positive weights in the adjacency list.
    """
    # Create mapping from original IDs to sequential numbers
    sorted_nodes = sorted(graph.nodes())
    node_to_num = {node: i + 1 for i, node in enumerate(sorted_nodes)}

    # Build adjacency lists with only positive-weight edges
    adjacency = {node_to_num[node]: [] for node in sorted_nodes}
    for (u, v), weight in weights.items():
        if weight > 0:
            u_num = node_to_num[u]
            v_num = node_to_num[v]
            adjacency[u_num].append(v_num)
            adjacency[v_num].append(u_num)

    with open(filepath, "w") as f:
        for node_num in sorted(adjacency.keys()):
            neighbors = " ".join(map(str, sorted(adjacency[node_num])))
            f.write(f"{node_num}: {neighbors}\n")

        f.write("\n# Weights\n")
        for (u, v), weight in weights.items():
            u_num = node_to_num[u]
            v_num = node_to_num[v]
            # Ensure consistent ordering (lower number first)
            if u_num > v_num:
                u_num, v_num = v_num, u_num
            f.write(f"({u_num}, {v_num}): {weight}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert specified rfam file to adjacency format"
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="test_graphs/rfam/RF02246.tsv",
        help="Path to rfam file (default: test_graphs/rfam/RF02246.tsv)",
    )
    parser.add_argument(
        "--method",
        "-m",
        choices=["linear", "threshold", "exp"],
        default="linear",
        help="Weight conversion method (default: linear)",
    )
    args = parser.parse_args()

    graph, scores = load_similarity_graph(args.path)
    weights = convert_dissimilarity_to_weights(scores, method=args.method)

    output_path = Path(args.path).with_suffix(".txt")
    save_graph_with_weights(graph, weights, output_path)
    print(f"Graph saved to {output_path}")


if __name__ == "__main__":
    main()
