from pathlib import Path
import networkx as nx
from algorithms.chalupa import ChalupaHeuristic
from algorithms.ilp_solver import solve_ilp_clique_cover
from loader import txt_to_networkx, get_ground_truth

class SimpleTestRunner:
    def __init__(self, test_data_dir="data"):
        self.test_data_dir = Path(test_data_dir)

    def chalupa_wrapper(self, txt_filepath):
        """Wrapper for Chalupa algorithm"""
        try:
            G = txt_to_networkx(txt_filepath)
            chalupa = ChalupaHeuristic(nx.complement(G))
            result = chalupa.run()
            return result['upper_bound']
        except Exception as e:
            print(f"Chalupa failed on {txt_filepath}: {e}")
            return None

    def ilp_wrapper(self, txt_filepath):
        """Wrapper for ILP solver"""
        try:
            G = txt_to_networkx(txt_filepath)
            result = solve_ilp_clique_cover(G)
            if 'error' in result:
                print(f"ILP failed on {txt_filepath}: {result['error']}")
                return None
            return result['chromatic_number']
        except Exception as e:
            print(f"ILP failed on {txt_filepath}: {e}")
            return None

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
    runner = SimpleTestRunner("test_cases/curated")

    print("Testing Chalupa Algorithm:")
    chalupa_results = runner.run_tests(runner.chalupa_wrapper, "Chromatic Number")

    print("\nTesting ILP Solver:")
    ilp_results = runner.run_tests(runner.ilp_wrapper, "Chromatic Number")
