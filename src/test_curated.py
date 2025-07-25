from pathlib import Path
from utils import get_value
import time
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

        f.write(f"\nSummary:\n")
        f.write(f"Correct: {correct}/{total}")
        if total > 0:
            f.write(f" ({correct/total*100:.1f}%)\n")
            deviation_sum = sum(r['deviation'] for r in results)
            f.write(f"Average Deviation: {deviation_sum / total:.2f}\n")

if __name__ == "__main__":
    path = "test_cases/generated/perturbed"
    runner = TestRunner(path)

    print("Testing Chalupa Algorithm:")
    chalupa_results = runner.run_tests(chalupa_wrapper, "Chromatic Number")

    print("\nTesting ILP Solver:")
    ilp_results = runner.run_tests(ilp_wrapper, "Chromatic Number")

    save_summary(chalupa_results, f"{path.replace('/', '_')}_chalupa")
    save_summary(ilp_results, f"{path.replace('/', '_')}_ilp")
