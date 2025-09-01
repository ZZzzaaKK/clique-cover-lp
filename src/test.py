#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path
from wrappers import chalupa_wrapper, ilp_wrapper, reduced_ilp_wrapper, interactive_reduced_ilp_wrapper
from utils import get_value

class TestRunner:
    def __init__(self, test_data_dir="data"):
        self.test_data_dir = Path(test_data_dir)

    def run_tests(self, algorithm_func, attribute_name="Vertex Clique Cover Number", timeout=60):
        """Run algorithm on all files and compare with ground truth"""
        results = []
        problem_type = "vertex_clique_cover" if attribute_name == "Vertex Clique Cover Number" else "chromatic_number"
        for txt_file in self.test_data_dir.glob("**/*.txt"):
            ground_truth = get_value(txt_file, attribute_name)
            if ground_truth is None:
                continue
            else:
                try:
                    ground_truth = int(ground_truth)
                except ValueError as e:
                    print(f"Error parsing ground truth for {txt_file}: {e}")
            start_time = time.time()
            if algorithm_func in [ilp_wrapper, reduced_ilp_wrapper, interactive_reduced_ilp_wrapper]:
                result = algorithm_func(str(txt_file), problem_type, time_limit=timeout)
            else:
                result = algorithm_func(str(txt_file), problem_type)
            predicted = result[0] if result else None
            is_optimal = result[1] if result else False
            end_time = time.time()
            if predicted is not None:
                results.append({
                    'file': txt_file.name,
                    'predicted': predicted,
                    'actual': ground_truth,
                    'deviation': predicted - ground_truth,
                    'correct': predicted == ground_truth,
                    'time_taken': end_time - start_time,
                    'is_optimal': is_optimal
                })
            else:
                results.append({
                    'file': txt_file.name,
                    'predicted': "Calculation timed out",
                    'actual': ground_truth,
                    'correct': False,
                    'time_taken': end_time - start_time,
                    'is_optimal': False
                })

        return results

def save_summary(results, name):
    correct = sum(1 for r in results if r['correct'])
    total = len(results)
    successful_results = [r for r in results if r.get('deviation') is not None]
    timed_out_results = len(results) - len(successful_results)

    output_file = f"results/raw/{name}.txt"
    with open(output_file, 'w') as f:
        f.write(f"Test Results for {name}\n")
        f.write("=" * 50 + "\n\n")
        for result in results:
            f.write(f"File: {result['file']}\n")
            f.write(f"Optimal solution found: {result.get('optimal', False)}\n")
            f.write(f"Predicted: {result['predicted']}\n")
            f.write(f"Actual: {result['actual']}\n")
            f.write(f"Deviation: {result.get('deviation', 'N/A')}\n")
            f.write(f"Correct: {result['correct']}\n")
            f.write(f"Time taken: {result['time_taken']}\n")
            f.write("-" * 30 + "\n")

        f.write("\nSummary:\n")
        f.write(f"Total tests: {total}\n")
        f.write(f"Successful: {len(successful_results)}\n")
        f.write(f"Timed out: {timed_out_results}\n")
        f.write(f"Correct predictions (overall): {correct}/{total} ({correct/total*100:.1f}%)\n")

        if len(successful_results) > 0:
            correct_successful = sum(1 for r in successful_results if r['correct'])
            f.write(f"Accuracy (on successful): {correct_successful/len(successful_results)*100:.1f}%")
            deviation_sum = sum(r['deviation'] for r in successful_results)
            f.write(f"\nAverage Deviation (on successful): {deviation_sum / len(successful_results):.2f}\n")


def main():
    parser = argparse.ArgumentParser(description='Run graph coloring algorithm tests')
    parser.add_argument('--chalupa', action='store_true', help='Run Chalupa heuristic')
    parser.add_argument('--ilp', action='store_true', help='Run ILP solver')
    parser.add_argument('--reduced-ilp', action='store_true', help='Run reduced ILP solver')
    parser.add_argument('--timeout', type=int, default=60, help='Timeout for ILP solvers in seconds')
    parser.add_argument('--interactive-reduced-ilp', action='store_true', help='Run interactive reduced ILP solver')
    parser.add_argument('--all', action='store_true', help='Run all algorithms')
    parser.add_argument('--chromatic-number', action='store_true', help='Test against Chromatic Number instead of Vertex Clique Cover Number')
    parser.add_argument('path', nargs='?', default='test_graphs/generated/perturbed',
                       help='Path to test data directory (default: test_graphs/generated/perturbed)')

    args = parser.parse_args()

    # If no algorithm specified, show help
    if not any([args.chalupa, args.ilp, args.reduced_ilp, args.interactive_reduced_ilp, args.all]):
        parser.print_help()
        sys.exit(1)

    runner = TestRunner(args.path)

    algorithms = []
    if args.all:
        algorithms = [
            ('chalupa', chalupa_wrapper),
            ('ilp', ilp_wrapper),
            ('reduced_ilp', reduced_ilp_wrapper),
            ('interactive_reduced_ilp', interactive_reduced_ilp_wrapper)
        ]
    else:
        if args.chalupa:
            algorithms.append(('chalupa', chalupa_wrapper))
        if args.ilp:
            algorithms.append(('ilp', ilp_wrapper))
        if args.reduced_ilp:
            algorithms.append(('reduced_ilp', reduced_ilp_wrapper))
        if args.interactive_reduced_ilp:
            algorithms.append(('interactive_reduced_ilp', interactive_reduced_ilp_wrapper))

    attribute_name = "Vertex Clique Cover Number"
    if args.chromatic_number:
        attribute_name = "Chromatic Number"

    for name, wrapper in algorithms:
        print(f"\nTesting {name.replace('_', ' ').title()} Algorithm against {attribute_name}:")
        results = runner.run_tests(wrapper, attribute_name, timeout=args.timeout)
        output_name = f"{Path(args.path).name}_{name}"
        save_summary(results, output_name)
        print(f"Results saved to results/raw/{output_name}.txt")

if __name__ == "__main__":
    main()
