from pathlib import Path
import networkx as nx
from src.algorithms.chalupa import ChalupaHeuristic
from src.algorithms.ilp_solver import solve_ilp_clique_cover

class SimpleTestRunner:
    def __init__(self, test_data_dir="data"):
        self.test_data_dir = Path(test_data_dir)

    def get_ground_truth(self, filepath, attribute_name="Chromatic Number"):
        """Extract just the ground truth value from a txt file"""
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith(f"{attribute_name}:"):
                    value_str = line.split(':', 1)[1].strip()
                    return int(value_str)
        return None

    def txt_to_networkx(self, txt_filepath):
        """Convert txt adjacency list format to NetworkX graph"""
        G = nx.Graph()

        with open(txt_filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if ':' in line and line.split(':')[0].strip().isdigit():
                    parts = line.split(':')
                    vertex = int(parts[0].strip())
                    G.add_node(vertex)

                    if len(parts) > 1 and parts[1].strip():
                        neighbors = [int(x) for x in parts[1].strip().split()]
                        for neighbor in neighbors:
                            G.add_edge(vertex, neighbor)
                elif line and not line.split(':')[0].strip().isdigit():
                    # Hit the attributes section, stop parsing
                    break

        # Convert to 0-indexed for Chalupa algorithm
        mapping = {old: new for new, old in enumerate(sorted(G.nodes()))}
        G = nx.relabel_nodes(G, mapping)

        return G


    def chalupa_wrapper(self, txt_filepath):
        """Wrapper for Chalupa algorithm"""
        try:
            G = self.txt_to_networkx(txt_filepath)
            chalupa = ChalupaHeuristic(nx.complement(G))
            result = chalupa.run()
            return result['upper_bound']
        except Exception as e:
            print(f"Chalupa failed on {txt_filepath}: {e}")
            return None

    def ilp_wrapper(self, txt_filepath):
        """Wrapper for ILP solver"""
        try:
            G = self.txt_to_networkx(txt_filepath)
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
            ground_truth = self.get_ground_truth(txt_file, attribute_name)
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
    runner = SimpleTestRunner("data/curated")

    print("Testing Chalupa Algorithm:")
    chalupa_results = runner.run_tests(runner.chalupa_wrapper, "Chromatic Number")

    print("\nTesting ILP Solver:")
    ilp_results = runner.run_tests(runner.ilp_wrapper, "Chromatic Number")
