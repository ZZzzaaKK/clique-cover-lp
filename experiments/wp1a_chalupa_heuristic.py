"""
Test script for Chalupa's heuristic algorithm on clique covering problem.

This script evaluates the performance of the Chalupa heuristic on test cases
and compares results against ground truth data.
"""

import sys
import os
import time
import json
import pickle
import numpy as np
import networkx as nx

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.algorithms.chalupa import ChalupaHeuristic
from src.evaluation.metrics import verify_solution, solution_quality

def load_graph_from_pickle(path: str) -> nx.Graph:
    """Load a NetworkX graph from pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)

def run_chalupa_test(data_path: str, num_runs: int = 5):
    """
    Test the Chalupa heuristic on a given dataset.

    Args:
        data_path: Path to the pickle file containing the graph
        num_runs: Number of times to run the algorithm (for statistical analysis)

    Returns:
        Dictionary containing test results and statistics
    """
    print(f"\n{'='*60}")
    print(f"Testing Chalupa Heuristic on: {data_path}")
    print(f"{'='*60}")

    # Load the graph
    G = load_graph_from_pickle(data_path)
    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Try to load ground truth if available
    ground_truth = None
    metadata_path = data_path.replace('.pkl', '_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            ground_truth = {int(k): v for k, v in metadata.get('communities', {}).items()}
            print(f"Ground truth loaded: {len(set(ground_truth.values()))} communities")

    # Run the algorithm multiple times
    results = []
    total_time = 0

    for run in range(num_runs):
        print(f"\n--- Run {run + 1}/{num_runs} ---")

        # Initialize and run the heuristic
        chalupa = ChalupaHeuristic(data_path)

        start_time = time.time()
        result = chalupa.run()
        end_time = time.time()

        runtime = end_time - start_time
        total_time += runtime

        # Validate the solution
        clique_covering = result['clique_covering']
        is_valid = verify_solution(G, clique_covering) if clique_covering else False

        print(f"  Lower bound: {result['lower_bound']}")
        print(f"  Upper bound: {result['upper_bound']}")
        print(f"  Bounds interval: {result['bounds_interval']}")
        print(f"  Runtime: {runtime:.3f} seconds")
        print(f"  Solution valid: {is_valid}")

        # Store results
        run_result = {
            'run_number': run + 1,
            'lower_bound': result['lower_bound'],
            'upper_bound': result['upper_bound'],
            'clique_covering': clique_covering,
            'independent_set': result['independent_set'],
            'runtime': runtime,
            'is_valid': is_valid,
            'gap': result['upper_bound'] - result['lower_bound'] if result['upper_bound'] != float('inf') else float('inf')
        }

        # Compare with ground truth if available
        if ground_truth and clique_covering and is_valid:
            quality_metrics = solution_quality(ground_truth, clique_covering)
            run_result['quality_metrics'] = quality_metrics

            print(f"  Ground truth cliques: {quality_metrics['num_ground_truth_cliques']}")
            print(f"  Solution cliques: {quality_metrics['num_solution_cliques']}")
            print(f"  Recovery rate: {quality_metrics['recovery_rate']:.3f}")
            print(f"  Purity: {quality_metrics['purity']:.3f}")

        results.append(run_result)

    # Compute statistics across runs
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")

    if results:
        lower_bounds = [r['lower_bound'] for r in results]
        upper_bounds = [r['upper_bound'] for r in results if r['upper_bound'] != float('inf')]
        gaps = [r['gap'] for r in results if r['gap'] != float('inf')]
        runtimes = [r['runtime'] for r in results]
        valid_runs = [r for r in results if r['is_valid']]

        print(f"Valid runs: {len(valid_runs)}/{len(results)}")
        print(f"Average runtime: {np.mean(runtimes):.3f} ± {np.std(runtimes):.3f} seconds")

        if lower_bounds:
            print(f"Lower bounds: min={min(lower_bounds)}, max={max(lower_bounds)}, avg={np.mean(lower_bounds):.2f}")

        if upper_bounds:
            print(f"Upper bounds: min={min(upper_bounds)}, max={max(upper_bounds)}, avg={np.mean(upper_bounds):.2f}")

        if gaps:
            print(f"Gaps: min={min(gaps)}, max={max(gaps)}, avg={np.mean(gaps):.2f}")

        # Best solution found
        best_run = min(valid_runs, key=lambda x: x['upper_bound']) if valid_runs else None
        if best_run:
            print(f"\nBest solution found:")
            print(f"  Run {best_run['run_number']}: {best_run['upper_bound']} cliques")
            print(f"  Bounds: [{best_run['lower_bound']}, {best_run['upper_bound']}]")

            if 'quality_metrics' in best_run:
                qm = best_run['quality_metrics']
                print(f"  vs Ground truth: {qm['num_ground_truth_cliques']} cliques")
                print(f"  Difference: {qm['clique_difference']:+d}")
                print(f"  Recovery rate: {qm['recovery_rate']:.3f}")
                print(f"  Purity: {qm['purity']:.3f}")

    return {
        'data_path': data_path,
        'graph_stats': {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'density': nx.density(G)
        },
        'ground_truth_available': ground_truth is not None,
        'runs': results,
        'num_runs': num_runs,
        'total_time': total_time
    }

def main():
    """Main test function."""
    print("Chalupa Heuristic Testing Suite")
    print("=" * 60)

    # Test on the specified dataset
    data_path = "data/skewed_n10_min6_max22_r30_perturbed.pkl"

    if not os.path.exists(data_path):
        print(f"Error: Data file not found: {data_path}")
        print("Available files in data directory:")
        if os.path.exists("data"):
            for f in os.listdir("data"):
                if f.endswith('.pkl'):
                    print(f"  {f}")
        return

    # Run comprehensive test
    results = run_chalupa_test(data_path, num_runs=5)

    # Save results to file
    output_file = "results/chalupa_heuristic_results.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Convert sets to lists for JSON serialization
    json_results = results.copy()
    for run in json_results['runs']:
        if 'clique_covering' in run and run['clique_covering']:
            run['clique_covering'] = [list(clique) for clique in run['clique_covering']]
        if 'independent_set' in run and run['independent_set']:
            run['independent_set'] = list(run['independent_set'])

    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
