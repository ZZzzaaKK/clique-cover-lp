"""Script for visualizing test cases and algorithm results."""

import sys
import os
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# Add the src directory to the path so we can import our modules
src_path = Path(__file__).parent.parent
sys.path.append(str(src_path))

from src.data.data_loader import load_test_case, list_available_test_cases
from src.algorithms.chalupa import ChalupaHeuristic
from src.algorithms.ilp_solver import ILPCliqueCover
from src.visualization.plot import visualize_graph, visualize_solution_comparison
from src.evaluation.metrics import verify_solution, solution_quality

def main():
    """Visualize test cases and algorithm results."""
    parser = argparse.ArgumentParser(description="Visualize test cases and algorithm results")
    parser.add_argument("--case", help="Name of the test case to visualize", default=None)
    parser.add_argument("--all", help="Visualize all test cases", action="store_true")
    parser.add_argument("--original", help="Visualize original graph", action="store_true")
    parser.add_argument("--perturbed", help="Visualize perturbed graph", action="store_true")
    parser.add_argument("--chalupa", help="Run and visualize Chalupa solution", action="store_true")
    parser.add_argument("--ilp", help="Run and visualize ILP solution", action="store_true")
    args = parser.parse_args()

    # List available test cases
    test_cases = list_available_test_cases()
    if not test_cases:
        print("No test cases found. Please run wp0_generate_test_cases.py first.")
        return

    print(f"Found {len(test_cases)} test cases.")

    # Determine which test cases to process
    cases_to_process = []
    if args.case:
        if args.case in test_cases:
            cases_to_process = [args.case]
        else:
            print(f"Test case '{args.case}' not found.")
            return
    elif args.all:
        cases_to_process = test_cases
    else:
        # Default: use the first test case
        cases_to_process = [test_cases[0]]
        print(f"No specific test case selected. Using default: {cases_to_process[0]}")

    # Process each test case
    for case_name in cases_to_process:
        print(f"\nProcessing test case: {case_name}")
        G_original, G_perturbed, communities, stats_original, stats_perturbed = load_test_case(case_name)

        # Convert ground truth to clique format
        gt_communities = {}
        for node, comm_id in communities.items():
            if comm_id not in gt_communities:
                gt_communities[comm_id] = set()
            gt_communities[comm_id].add(node)
        ground_truth_cliques = list(gt_communities.values())

        # Visualize original graph if requested
        if args.original:
            visualize_graph(G_original, communities, f"Original Graph: {case_name}")

        # Visualize perturbed graph if requested
        if args.perturbed:
            visualize_graph(G_perturbed, communities, f"Perturbed Graph: {case_name}")

        # Run and visualize Chalupa solution if requested
        if args.chalupa:
            print("  Running Chalupa's heuristic...")
            solver = ChalupaHeuristic(G_perturbed, population_size=50, iterations=100)
            chalupa_solution = solver.solve()
            is_valid = verify_solution(G_perturbed, chalupa_solution)
            quality = solution_quality(communities, chalupa_solution) if is_valid else {}

            print(f"  Chalupa solution found {len(chalupa_solution)} cliques.")
            print(f"  Valid solution: {is_valid}")
            if is_valid:
                print(f"  Recovery rate: {quality.get('recovery_rate', 0):.2f}")
                print(f"  Purity: {quality.get('purity', 0):.2f}")

            visualize_solution_comparison(G_perturbed, ground_truth_cliques, chalupa_solution,
                                      "Chalupa Heuristic vs Ground Truth")

        # Run and visualize ILP solution if requested
        if args.ilp:
            # Skip large graphs that would be too slow for ILP
            if G_perturbed.number_of_nodes() > 100:
                print(f"  Skipping ILP for large graph with {G_perturbed.number_of_nodes()} nodes")
            else:
                print("  Running ILP solver...")
                solver = ILPCliqueCover(G_perturbed, time_limit=60, verbose=False)
                ilp_solution = solver.solve()
                is_valid = verify_solution(G_perturbed, ilp_solution)
                quality = solution_quality(communities, ilp_solution) if is_valid else {}

                print(f"  ILP solution found {len(ilp_solution)} cliques.")
                print(f"  Valid solution: {is_valid}")
                if is_valid:
                    print(f"  Recovery rate: {quality.get('recovery_rate', 0):.2f}")
                    print(f"  Purity: {quality.get('purity', 0):.2f}")

                visualize_solution_comparison(G_perturbed, ground_truth_cliques, ilp_solution,
                                          "ILP Solution vs Ground Truth")

        # If both algorithms were run, compare them directly
        if args.chalupa and args.ilp and G_perturbed.number_of_nodes() <= 100:
            solver1 = ChalupaHeuristic(G_perturbed, population_size=50, iterations=100)
            chalupa_solution = solver1.solve()

            solver2 = ILPCliqueCover(G_perturbed, time_limit=60, verbose=False)
            ilp_solution = solver2.solve()

            plt.figure(figsize=(15, 7))

            # Show Chalupa solution
            plt.subplot(1, 2, 1)
            algo_communities = {}
            for i, clique in enumerate(chalupa_solution):
                for node in clique:
                    algo_communities[node] = i

            pos = nx.spring_layout(G_perturbed, seed=42)
            unique_communities = set(algo_communities.values())
            color_map = plt.cm.rainbow(np.linspace(0, 1, len(unique_communities)))
            community_colors = {comm: color_map[i] for i, comm in enumerate(unique_communities)}
            node_colors = [community_colors[algo_communities[node]] for node in G_perturbed.nodes()]

            nx.draw_networkx_nodes(G_perturbed, pos, node_size=100, node_color=node_colors, alpha=0.8)
            nx.draw_networkx_edges(G_perturbed, pos, width=0.5, alpha=0.5)
            plt.title(f"Chalupa Solution ({len(chalupa_solution)} cliques)")
            plt.axis('off')

            # Show ILP solution
            plt.subplot(1, 2, 2)
            algo_communities = {}
            for i, clique in enumerate(ilp_solution):
                for node in clique:
                    algo_communities[node] = i

            unique_communities = set(algo_communities.values())
            color_map = plt.cm.rainbow(np.linspace(0, 1, len(unique_communities)))
            community_colors = {comm: color_map[i] for i, comm in enumerate(unique_communities)}
            node_colors = [community_colors[algo_communities[node]] for node in G_perturbed.nodes()]

            nx.draw_networkx_nodes(G_perturbed, pos, node_size=100, node_color=node_colors, alpha=0.8)
            nx.draw_networkx_edges(G_perturbed, pos, width=0.5, alpha=0.5)
            plt.title(f"ILP Solution ({len(ilp_solution)} cliques)")
            plt.axis('off')

            plt.suptitle("Algorithm Comparison")
            plt.tight_layout()
            plt.show()
            plt.savefig("comparison.png")
            print("Plot saved to comparison.png")

if __name__ == "__main__":
    main()
