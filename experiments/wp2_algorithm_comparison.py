"""Script for comparing and visualizing different clique covering algorithms."""

import sys
import os
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add the src directory to the path so we can import our modules
src_path = Path(__file__).parent.parent
sys.path.append(str(src_path))

from src.data.data_loader import load_test_case, list_available_test_cases
from src.algorithms.chalupa import ChalupaHeuristic
from src.algorithms.ilp_solver import ILPCliqueCover
from src.evaluation.comparison import batch_comparison, run_chalupa, run_ilp
from src.visualization.plot import visualize_graph, visualize_solution_comparison

def main():
    """Run the algorithm comparison experiment."""
    # List available test cases
    test_cases = list_available_test_cases()
    if not test_cases:
        print("No test cases found. Please run wp0_generate_test_cases.py first.")
        return

    print(f"Found {len(test_cases)} test cases.")
    
    # Select a subset of test cases to avoid long runtime
    selected_cases = test_cases[:3]  # Use first 3 test cases
    print(f"Selected test cases: {selected_cases}")
    
    # Load graphs, ground truths, and prepare for batch comparison
    graphs = []
    ground_truths = []
    graph_names = []
    
    for case_name in selected_cases:
        G_original, G_perturbed, communities, stats_original, stats_perturbed = load_test_case(case_name)
        graphs.append(G_perturbed)
        ground_truths.append(communities)
        graph_names.append(case_name)
    
    # Define algorithms to compare
    algorithms = {
        "Chalupa": lambda G, **kwargs: run_chalupa(G, population_size=50, iterations=100),
        "ILP": lambda G, **kwargs: run_ilp(G, time_limit=60, verbose=False)
    }
    
    # Run comparison with visualization enabled
    results_df = batch_comparison(
        graphs=graphs,
        ground_truths=ground_truths,
        graph_names=graph_names,
        algorithms=algorithms,
        max_time=120,  # 2 minute timeout per algorithm
        visualize=True  # Enable visualization
    )
    
    # Save results
    os.makedirs("results", exist_ok=True)
    results_df.to_csv("results/algorithm_comparison.csv", index=False)
    
    # Print summary
    print("\nSummary of results:")
    summary = results_df.groupby("algorithm")[["runtime", "num_cliques", "recovery_rate", "purity"]].mean()
    print(summary)
    
    # Create comparison visualizations
    plt.figure(figsize=(12, 10))
    
    # Plot runtime comparison
    plt.subplot(2, 2, 1)
    results_df.pivot(index="graph", columns="algorithm", values="runtime").plot(kind="bar")
    plt.title("Runtime Comparison")
    plt.ylabel("Runtime (seconds)")
    plt.xticks(rotation=45)
    
    # Plot number of cliques comparison
    plt.subplot(2, 2, 2)
    results_df.pivot(index="graph", columns="algorithm", values="num_cliques").plot(kind="bar")
    plt.title("Number of Cliques")
    plt.ylabel("Number of Cliques")
    plt.xticks(rotation=45)
    
    # Plot recovery rate comparison
    plt.subplot(2, 2, 3)
    results_df.pivot(index="graph", columns="algorithm", values="recovery_rate").plot(kind="bar")
    plt.title("Recovery Rate Comparison")
    plt.ylabel("Recovery Rate")
    plt.xticks(rotation=45)
    
    # Plot purity comparison
    plt.subplot(2, 2, 4)
    results_df.pivot(index="graph", columns="algorithm", values="purity").plot(kind="bar")
    plt.title("Purity Comparison")
    plt.ylabel("Purity")
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig("results/algorithm_comparison.png")
    plt.show()
    
if __name__ == "__main__":
    main()