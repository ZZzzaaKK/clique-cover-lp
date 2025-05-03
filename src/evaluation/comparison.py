"""
Utilities for comparing different clique covering algorithms.
"""
import time
import networkx as nx
import pandas as pd
from typing import List, Set, Dict, Callable, Any, Tuple

from src.algorithms.chalupa import ChalupaHeuristic
from src.algorithms.ilp_solver import ILPCliqueCover
from src.evaluation.metrics import verify_solution, solution_quality, compare_solutions

def run_algorithm_with_timing(algorithm_func: Callable, G: nx.Graph, **kwargs) -> Tuple[List[Set[int]], float]:
    """
    Run an algorithm and time its execution.

    Args:
        algorithm_func: Function that implements the algorithm
        G: NetworkX graph
        **kwargs: Additional arguments to pass to the algorithm

    Returns:
        Tuple of (solution, runtime in seconds)
    """
    start_time = time.time()
    solution = algorithm_func(G, **kwargs)
    end_time = time.time()

    return solution, end_time - start_time

def compare_algorithms(G: nx.Graph, ground_truth: Dict[int, int],
                      algorithms: Dict[str, Callable],
                      max_time: int = 300) -> pd.DataFrame:
    """
    Compare multiple algorithms on a single graph.

    Args:
        G: NetworkX graph
        ground_truth: Dictionary mapping node to its original clique id
        algorithms: Dictionary mapping algorithm name to function
        max_time: Maximum time (in seconds) to allow for each algorithm

    Returns:
        DataFrame with comparison results
    """
    results = []

    for algorithm_name, algorithm_func in algorithms.items():
        try:
            # Run algorithm with time limit
            solution, runtime = run_algorithm_with_timing(algorithm_func, G, time_limit=max_time)

            # Check if solution is valid
            valid = verify_solution(G, solution)

            # Calculate solution quality
            if valid:
                quality = solution_quality(ground_truth, solution)
            else:
                quality = {
                    "num_ground_truth_cliques": 0,
                    "num_solution_cliques": 0,
                    "clique_difference": float('inf'),
                    "recovery_rate": 0.0,
                    "purity": 0.0,
                    "fragmentation": 0.0
                }

            # Add result
            results.append({
                "algorithm": algorithm_name,
                "runtime": runtime,
                "valid": valid,
                "num_cliques": len(solution),
                "recovery_rate": quality["recovery_rate"],
                "purity": quality["purity"],
                "fragmentation": quality["fragmentation"]
            })
        except Exception as e:
            # Algorithm failed
            print(f"Algorithm {algorithm_name} failed: {e}")
            results.append({
                "algorithm": algorithm_name,
                "runtime": max_time,
                "valid": False,
                "num_cliques": 0,
                "recovery_rate": 0.0,
                "purity": 0.0,
                "fragmentation": 0.0
            })

    return pd.DataFrame(results)

def batch_comparison(graphs: List[nx.Graph], ground_truths: List[Dict[int, int]],
                    graph_names: List[str],
                    algorithms: Dict[str, Callable],
                    max_time: int = 300) -> pd.DataFrame:
    """
    Compare multiple algorithms on multiple graphs.

    Args:
        graphs: List of NetworkX graphs
        ground_truths: List of dictionaries mapping node to its original clique id
        graph_names: Names of the graphs
        algorithms: Dictionary mapping algorithm name to function
        max_time: Maximum time (in seconds) to allow for each algorithm

    Returns:
        DataFrame with comparison results
    """
    all_results = []

    for i, (G, truth, name) in enumerate(zip(graphs, ground_truths, graph_names)):
        print(f"Processing graph {i+1}/{len(graphs)}: {name}")

        # Compare algorithms on this graph
        results = compare_algorithms(G, truth, algorithms, max_time)

        # Add graph information
        results["graph"] = name
        results["nodes"] = G.number_of_nodes()
        results["edges"] = G.number_of_edges()

        all_results.append(results)

    # Combine all results
    return pd.concat(all_results, ignore_index=True)

def run_chalupa(G: nx.Graph, **kwargs) -> List[Set[int]]:
    """Run Chalupa's heuristic algorithm."""
    solver = ChalupaHeuristic(G, **kwargs)
    return solver.solve()

def run_ilp(G: nx.Graph, **kwargs) -> List[Set[int]]:
    """Run ILP solver."""
    solver = ILPCliqueCover(G, **kwargs)
    return solver.solve()
