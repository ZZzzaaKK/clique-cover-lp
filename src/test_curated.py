from pathlib import Path
from utils import get_ground_truth
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wrappers import chalupa_wrapper, ilp_wrapper

class TestRunner:
    def __init__(self, test_data_dir="data"):
        self.test_data_dir = Path(test_data_dir)

    def run_tests(self, algorithm_func, attribute_name="Chromatic Number"):
        """Run algorithm on all files and compare with ground truth"""
        results = []

        for txt_file in self.test_data_dir.glob("**/*.txt"):
            ground_truth = get_ground_truth(txt_file, attribute_name)
            if ground_truth is None:
                continue

            predicted = algorithm_func(str(txt_file))
            if predicted is None:
                continue

            results.append({
                'file': txt_file.name,
                'predicted': predicted,
                'actual': ground_truth,
                'deviation': predicted - ground_truth,
                'correct': predicted == ground_truth
            })

        # Print summary
        correct = sum(1 for r in results if r['correct'])
        total = len(results)
        if total > 0:
            print(f"Correct: {correct}/{total} ({correct/total*100:.1f}%)")
            deviation_sum = sum(r['deviation'] for r in results)
            print(f"Average Deviation: {deviation_sum / total:.2f}")

        return results

# Usage:
if __name__ == "__main__":
    runner = TestRunner("test_cases/curated")

    print("Testing Chalupa Algorithm:")
    chalupa_results = runner.run_tests(chalupa_wrapper, "Chromatic Number")

    print("\nTesting ILP Solver:")
    ilp_results = runner.run_tests(ilp_wrapper, "Chromatic Number")
