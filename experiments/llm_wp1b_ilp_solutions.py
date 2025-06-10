"""
Experiment to evaluate the ILP solver for clique coloring (WP1b).
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
from src.algorithms.ilp_solver import ILPCliqueCover
from src.evaluation.metrics import verify_solution, solution_quality
from src.visualization.plot import visualize_graph, visualize_solution_comparison

def main():
    """Run the ILP solver experiment."""
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

        # Skip large graphs that would be too slow for ILP
        if G_perturbed.number_of_nodes() > 100:
            print(f"  Skipping large graph with {G_perturbed.number_of_nodes()} nodes")
            continue

        # Run ILP solver with different time limits
        for time_limit in [30, 60, 300]:
            print(f"  Running with time_limit={time_limit}s")

            # Measure runtime
            import time
            start_time = time.time()

            # Create and run solver
            solver = ILPCliqueCover(G_perturbed, time_limit=time_limit, verbose=False)
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
                "time_limit": time_limit,
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
            if case_name == test_cases[0] and time_limit == 300:
                # Convert ground truth to clique format
                gt_communities = {}
                for node, comm_id in communities.items():
                    if comm_id not in gt_communities:
                        gt_communities[comm_id] = set()
                    gt_communities[comm_id].add(node)
                ground_truth_cliques = list(gt_communities.values())

                visualize_solution_comparison(G_perturbed, ground_truth_cliques, solution,
                                           "ILP Solution vs Ground Truth")

    # Convert results to DataFrame and save
    results_df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    results_df.to_csv("results/ilp_results.csv", index=False)

    # Print summary
    print("\nSummary of results:")
    summary = results_df.groupby("time_limit")[["valid", "num_cliques", "runtime", "recovery_rate"]].mean()
    print(summary)

    # Visualize results
    plt.figure(figsize=(12, 6))

    # Plot runtime vs time limit
    plt.subplot(1, 3, 1)
    results_df.groupby("time_limit")["runtime"].mean().plot(kind="bar")
    plt.title("Average Runtime by Time Limit")
    plt.ylabel("Runtime (s)")

    # Plot number of cliques vs time limit
    plt.subplot(1, 3, 2)
    results_df.groupby("time_limit")["num_cliques"].mean().plot(kind="bar")
    plt.title("Average Number of Cliques")
    plt.ylabel("Number of Cliques")

    # Plot recovery rate vs time limit
    plt.subplot(1, 3, 3)
    results_df.groupby("time_limit")["recovery_rate"].mean().plot(kind="bar")
    plt.title("Average Recovery Rate")
    plt.ylabel("Recovery Rate")

    plt.tight_layout()
    plt.savefig("results/ilp_time_limit_study.png")

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
    plt.savefig("results/ilp_scaling_study.png")

if __name__ == "__main__":
    main()
