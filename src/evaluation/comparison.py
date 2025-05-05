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
                    max_time: int = 300,
                    visualize: bool = False) -> pd.DataFrame:
    """
    Compare multiple algorithms on multiple graphs.

    Args:
        graphs: List of NetworkX graphs
        ground_truths: List of dictionaries mapping node to its original clique id
        graph_names: Names of the graphs
        algorithms: Dictionary mapping algorithm name to function
        max_time: Maximum time (in seconds) to allow for each algorithm
        visualize: Whether to visualize the solutions

    Returns:
        DataFrame with comparison results
    """
    all_results = []
    all_solutions = {}

    for i, (G, truth, name) in enumerate(zip(graphs, ground_truths, graph_names)):
        print(f"Processing graph {i+1}/{len(graphs)}: {name}")

        # Compare algorithms on this graph
        results = compare_algorithms(G, truth, algorithms, max_time)

        # Store solutions for visualization
        if visualize:
            # Create solution dictionary for this graph
            solutions_for_graph = {}
            
            # Gather solutions from each algorithm
            for algorithm_name, algorithm_func in algorithms.items():
                try:
                    solution, _ = run_algorithm_with_timing(algorithm_func, G, time_limit=max_time)
                    solutions_for_graph[algorithm_name] = solution
                except Exception as e:
                    print(f"Algorithm {algorithm_name} failed during visualization: {e}")
            
            all_solutions[name] = solutions_for_graph
            
            # Convert ground truth to cliques format for visualization
            if i == 0:  # Only visualize the first graph to avoid too many plots
                try:
                    from src.visualization.plot import visualize_graph, visualize_solution_comparison
                    import matplotlib.pyplot as plt
                    
                    # Create ground truth cliques
                    gt_communities = {}
                    for node, comm_id in truth.items():
                        if comm_id not in gt_communities:
                            gt_communities[comm_id] = set()
                        gt_communities[comm_id].add(node)
                    ground_truth_cliques = list(gt_communities.values())
                    
                    # Visualize graph
                    visualize_graph(G, truth, f"Graph: {name}")
                    
                    # Visualize each algorithm's solution
                    for algorithm_name, solution in solutions_for_graph.items():
                        visualize_solution_comparison(G, ground_truth_cliques, solution,
                                                 f"{algorithm_name} vs Ground Truth")
                    
                    # Visualize all solutions side by side if there are multiple algorithms
                    if len(solutions_for_graph) > 1:
                        fig, axes = plt.subplots(1, len(solutions_for_graph), figsize=(5*len(solutions_for_graph), 5))
                        fig.suptitle(f"Algorithm Comparison on {name}")
                        
                        for j, (algorithm_name, solution) in enumerate(solutions_for_graph.items()):
                            # Create community mapping for this solution
                            algo_communities = {}
                            for k, clique in enumerate(solution):
                                for node in clique:
                                    algo_communities[node] = k
                            
                            # Use same layout for all algorithms
                            pos = nx.spring_layout(G, seed=42)
                            
                            # Get unique communities and assign colors
                            unique_communities = set(algo_communities.values())
                            import numpy as np
                            color_map = plt.cm.rainbow(np.linspace(0, 1, len(unique_communities)))
                            community_colors = {comm: color_map[i] for i, comm in enumerate(unique_communities)}
                            node_colors = [community_colors.get(algo_communities.get(node, -1), 'gray') for node in G.nodes()]
                            
                            # Plot on the appropriate subplot
                            ax = axes[j] if len(solutions_for_graph) > 1 else axes
                            nx.draw_networkx_nodes(G, pos, node_size=100, node_color=node_colors, alpha=0.8, ax=ax)
                            nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5, ax=ax)
                            ax.set_title(f"{algorithm_name} ({len(solution)} cliques)")
                            ax.axis('off')
                        
                        plt.tight_layout()
                        plt.show()
                except ImportError:
                    print("Visualization modules not available. Skipping visualization.")
                except Exception as e:
                    print(f"Error during visualization: {e}")

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
