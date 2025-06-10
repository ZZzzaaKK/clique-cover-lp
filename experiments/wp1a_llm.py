"""
Experiment to evaluate Chalupa's heuristic algorithm (WP1a).
"""
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
from src.evaluation.metrics import verify_solution, solution_quality
from src.visualization.plot import visualize_graph, visualize_solution_comparison

def main():
    """Run the Chalupa heuristic experiment."""
    # List available test cases
    test_cases = list_available_test_cases()
    if not test_cases:
        print("No test cases found. Please run wp0_generate_test_cases.py first.")
        return

    print(f"Found {len(test_cases)} test cases.")

    results = []

    # Process each test case
    for case_name in test_cases:
        print(f"\nProcessing test case: {case_name}")
        G_original, G_perturbed, communities, stats_original, stats_perturbed = load_test_case(case_name)

        # Run Chalupa's heuristic with different parameters
        for iterations in [50, 100, 200]:
            for population_size in [30, 50]:
                print(f"  Running with iterations={iterations}, population_size={population_size}")

                # Measure runtime
                import time
                start_time = time.time()

                # Create and run solver
                solver = ChalupaHeuristic(G_perturbed, population_size=population_size,
                                         iterations=iterations)
                solution = solver.solve()

                runtime = time.time() - start_time

                # Verify solution
                is_valid = verify_solution(G_perturbed, solution)

                # Calculate solution quality
                if is_valid:
                    quality = solution_quality(communities, solution)
                else:
                    quality = {
                        "recovery_rate": 0.0,
                        "purity": 0.0,
                        "fragmentation": 0.0
                    }

                # Add to results
                results.append({
                    "case_name": case_name,
                    "iterations": iterations,
                    "population_size": population_size,
                    "runtime": runtime,
                    "valid": is_valid,
                    "num_cliques": len(solution),
                    "recovery_rate": quality["recovery_rate"],
                    "purity": quality["purity"],
                    "fragmentation": quality["fragmentation"],
                    "nodes": G_perturbed.number_of_nodes(),
                    "edges": G_perturbed.number_of_edges()
                })

                # Visualize one example solution
                if case_name == test_cases[0] and iterations == 100 and population_size == 50:
                    # Convert ground truth to clique format
                    gt_communities = {}
                    for node, comm_id in communities.items():
                        if comm_id not in gt_communities:
                            gt_communities[comm_id] = set()
                        gt_communities[comm_id].add(node)
                    ground_truth_cliques = list(gt_communities.values())

                    visualize_solution_comparison(G_perturbed, ground_truth_cliques, solution,
                                               "Chalupa Heuristic vs Ground Truth")

    # Convert results to DataFrame and save
    results_df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    results_df.to_csv("results/chalupa_results.csv", index=False)

    # Print summary
    print("\nSummary of results:")
    summary = results_df.groupby(["iterations", "population_size"])[["valid", "num_cliques", "runtime", "recovery_rate"]].mean()
    print(summary)

    # Visualize results
    plt.figure(figsize=(12, 8))

    # Plot runtime vs parameters
    plt.subplot(2, 2, 1)
    pivot = results_df.pivot_table(index="iterations", columns="population_size", values="runtime")
    pivot.plot(marker='o')
    plt.title("Runtime by Parameters")
    plt.ylabel("Runtime (s)")

    # Plot number of cliques vs parameters
    plt.subplot(2, 2, 2)
    pivot = results_df.pivot_table(index="iterations", columns="population_size", values="num_cliques")
    pivot.plot(marker='o')
    plt.title("Number of Cliques by Parameters")
    plt.ylabel("Number of Cliques")

    # Plot recovery rate vs parameters
    plt.subplot(2, 2, 3)
    pivot = results_df.pivot_table(index="iterations", columns="population_size", values="recovery_rate")
    pivot.plot(marker='o')
    plt.title("Recovery Rate by Parameters")
    plt.ylabel("Recovery Rate")

    # Plot purity vs parameters
    plt.subplot(2, 2, 4)
    pivot = results_df.pivot_table(index="iterations", columns="population_size", values="purity")
    pivot.plot(marker='o')
    plt.title("Purity by Parameters")
    plt.ylabel("Purity")

    plt.tight_layout()
    plt.savefig("results/chalupa_parameter_study.png")

    # Plot runtime vs. graph size
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(results_df["nodes"], results_df["runtime"])
    plt.title("Runtime vs. Number of Nodes")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Runtime (s)")

    plt.subplot(1, 2, 2)
    plt.scatter(results_df["edges"], results_df["runtime"])
    plt.title("Runtime vs. Number of Edges")
    plt.xlabel("Number of Edges")
    plt.ylabel("Runtime (s)")

    plt.tight_layout()
    plt.savefig("results/chalupa_scaling_study.png")

if __name__ == "__main__":
    main()
