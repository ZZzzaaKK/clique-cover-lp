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

    def run_tests(self, algorithm_func, attribute_name="Chromatic Number"):
        """Run algorithm on all files and compare with ground truth"""
        results = []
        for txt_file in self.test_data_dir.glob("**/*.txt"):
            ground_truth = get_value(txt_file, attribute_name)
            if ground_truth is None:
                continue
            else:
                ground_truth = int(ground_truth)
            start_time = time.time()
            predicted = algorithm_func(str(txt_file))
            end_time = time.time()
            results.append({
                'file': txt_file.name,
                'predicted': predicted,
                'actual': ground_truth,
                'deviation': predicted - ground_truth,
                'correct': predicted == ground_truth,
                'time_taken': end_time - start_time
            })

        return results

def save_summary(results, name):
    correct = sum(1 for r in results if r['correct'])
    total = len(results)
    output_file = f"results/{name}.txt"
    with open(output_file, 'w') as f:
        f.write(f"Test Results for {name}\n")
        f.write("=" * 50 + "\n\n")
        for result in results:
            f.write(f"File: {result['file']}\n")
            f.write(f"Predicted: {result['predicted']}\n")
            f.write(f"Actual: {result['actual']}\n")
            f.write(f"Deviation: {result['deviation']}\n")
            f.write(f"Correct: {result['correct']}\n")
            f.write(f"Time taken: {result['time_taken']}\n")
            f.write("-" * 30 + "\n")

        f.write("\nSummary:\n")
        f.write(f"Correct: {correct}/{total}")

        if total > 0:
            f.write(f" ({correct/total*100:.1f}%)\n")
            deviation_sum = sum(r['deviation'] for r in results)
            f.write(f"Average Deviation: {deviation_sum / total:.2f}\n")

def main():
    parser = argparse.ArgumentParser(description='Run graph coloring algorithm tests')
    parser.add_argument('--chalupa', action='store_true', help='Run Chalupa heuristic')
    parser.add_argument('--ilp', action='store_true', help='Run ILP solver')
    parser.add_argument('--reduced-ilp', action='store_true', help='Run reduced ILP solver')
    parser.add_argument('--interactive-reduced-ilp', action='store_true', help='Run interactive reduced ILP solver')
    parser.add_argument('--all', action='store_true', help='Run all algorithms')
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

    for name, wrapper in algorithms:
        print(f"\nTesting {name.replace('_', ' ').title()} Algorithm:")
        results = runner.run_tests(wrapper, "Chromatic Number")
        output_name = f"{Path(args.path).name}_{name}"
        save_summary(results, output_name)
        print(f"Results saved to results/{output_name}.txt")

if __name__ == "__main__":
    main()
