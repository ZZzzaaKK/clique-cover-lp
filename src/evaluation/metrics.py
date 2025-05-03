"""
Metrics for evaluating clique covering algorithms.
"""
import networkx as nx
from typing import List, Set, Dict

def count_used_colors(solution: List[Set[int]]) -> int:
    """
    Count the number of colors (cliques) used in the solution.

    Args:
        solution: List of cliques (each represented as a set of nodes)

    Returns:
        Number of colors used
    """
    return len(solution)

def verify_solution(G: nx.Graph, solution: List[Set[int]]) -> bool:
    """
    Verify that a solution is valid.

    A valid solution:
    1. Every node is in at least one clique
    2. Every clique in the solution is a clique in G

    Args:
        G: NetworkX graph
        solution: List of cliques (each represented as a set of nodes)

    Returns:
        True if the solution is valid, False otherwise
    """
    # Check that every node is covered
    covered_nodes = set()
    for clique in solution:
        covered_nodes.update(clique)

    if covered_nodes != set(G.nodes()):
        return False

    # Check that every set in the solution is a clique in G
    for clique in solution:
        subgraph = G.subgraph(clique)
        if subgraph.number_of_edges() != (len(clique) * (len(clique) - 1)) // 2:
            return False

    return True

def solution_quality(ground_truth: Dict[int, int], solution: List[Set[int]]) -> Dict[str, float]:
    """
    Calculate various quality metrics for a solution compared to ground truth.

    Args:
        ground_truth: Dictionary mapping node to its original clique id
        solution: List of cliques (each represented as a set of nodes)

    Returns:
        Dictionary of quality metrics
    """
    # Convert ground truth to set format
    gt_communities = {}
    for node, comm_id in ground_truth.items():
        if comm_id not in gt_communities:
            gt_communities[comm_id] = set()
        gt_communities[comm_id].add(node)

    ground_truth_cliques = list(gt_communities.values())

    # Calculate metrics
    metrics = {
        "num_ground_truth_cliques": len(ground_truth_cliques),
        "num_solution_cliques": len(solution),
        "clique_difference": len(solution) - len(ground_truth_cliques),
        "recovery_rate": 0.0,  # How many ground truth cliques are recovered
        "purity": 0.0,  # Average purity of solution cliques
        "fragmentation": 0.0,  # Average number of solution cliques per ground truth clique
    }

    # Calculate recovery rate
    recovered = 0
    for gt_clique in ground_truth_cliques:
        for sol_clique in solution:
            if gt_clique.issubset(sol_clique):
                recovered += 1
                break

    if ground_truth_cliques:
        metrics["recovery_rate"] = recovered / len(ground_truth_cliques)

    # Calculate purity
    total_purity = 0
    for sol_clique in solution:
        # Find the ground truth clique with maximum overlap
        max_overlap = 0
        for gt_clique in ground_truth_cliques:
            overlap = len(sol_clique.intersection(gt_clique))
            max_overlap = max(max_overlap, overlap)

        if len(sol_clique) > 0:
            total_purity += max_overlap / len(sol_clique)

    if solution:
        metrics["purity"] = total_purity / len(solution)

    # Calculate fragmentation
    fragmentation_counts = [0] * len(ground_truth_cliques)
    for i, gt_clique in enumerate(ground_truth_cliques):
        for sol_clique in solution:
            if not gt_clique.isdisjoint(sol_clique):
                fragmentation_counts[i] += 1

    if ground_truth_cliques:
        metrics["fragmentation"] = sum(fragmentation_counts) / len(ground_truth_cliques)

    return metrics

def compare_solutions(G: nx.Graph, solution1: List[Set[int]], solution2: List[Set[int]]) -> Dict[str, float]:
    """
    Compare two solutions to each other.

    Args:
        G: NetworkX graph
        solution1: First solution (list of cliques)
        solution2: Second solution (list of cliques)

    Returns:
        Dictionary of comparison metrics
    """
    # Check validity
    valid1 = verify_solution(G, solution1)
    valid2 = verify_solution(G, solution2)

    comparison = {
        "solution1_valid": valid1,
        "solution2_valid": valid2,
        "solution1_cliques": len(solution1),
        "solution2_cliques": len(solution2),
        "clique_difference": len(solution1) - len(solution2),
        "jaccard_similarity": 0.0,  # Similarity between the two solutions
    }

    # Calculate Jaccard similarity between the two solutions
    # Convert each solution to a set of frozensets for comparison
    sol1_sets = {frozenset(clique) for clique in solution1}
    sol2_sets = {frozenset(clique) for clique in solution2}

    # Jaccard similarity: |A ∩ B| / |A ∪ B|
    intersection = len(sol1_sets.intersection(sol2_sets))
    union = len(sol1_sets.union(sol2_sets))

    if union > 0:
        comparison["jaccard_similarity"] = intersection / union

    return comparison
