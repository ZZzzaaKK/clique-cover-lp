# src/wp3_evaluation.py
"""
WP3 Evaluation using existing test graphs from test_graphs/generated/perturbed
(kein Greedy-Fallback; ILP mit Gurobi)
"""
import sys
from pathlib import Path

# sys.path früh erweitern, falls direkt gestartetet
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import json
from typing import Dict, List, Any, Optional, Tuple, Set
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from src.utils import txt_to_networkx
from src.algorithms.cluster_editing_kernelization import (
    AdvancedClusterEditingInstance,
    AdvancedKernelization,
)
from src.algorithms.cluster_editing_solver import ClusterEditingSolver

# ----------------------------------------------------------------------
# Hilfsfunktionen für Cluster-Vergleich
# ----------------------------------------------------------------------
def _pair_set_from_clusters(clusters: List[Set[int]]) -> Set[Tuple[int, int]]:
    """Erzeugt die Menge aller ungeordneten Knotenpaare, die im selben Cluster liegen."""
    pairs: Set[Tuple[int, int]] = set()
    for C in clusters:
        cl = sorted(C)
        for i, u in enumerate(cl):
            for v in cl[i + 1:]:
                a, b = (u, v) if u < v else (v, u)
                pairs.add((a, b))
    return pairs


def _jaccard_pairs(clA: List[Set[int]], clB: List[Set[int]]) -> float:
    """Jaccard-Index der Co-Clustering-Paare von zwei Partitionen."""
    A = _pair_set_from_clusters(clA)
    B = _pair_set_from_clusters(clB)
    if not A and not B:
        return 1.0
    return len(A & B) / float(len(A | B))


# ----------------------------------------------------------------------
class WP3TestGraphEvaluator:
    def __init__(self,
                 test_graphs_dir: str = "test_graphs/generated/perturbed",
                 output_dir: str = "results/wp3",
                 ilp_time_limit: Optional[float] = None,
                 ilp_gap: Optional[float] = None,
                 ilp_threads: Optional[int] = None):
        self.test_graphs_dir = Path(test_graphs_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.ilp_time_limit = ilp_time_limit
        self.ilp_gap = ilp_gap
        self.ilp_threads = ilp_threads

        if not self.test_graphs_dir.exists():
            raise ValueError(f"Test graphs directory not found: {self.test_graphs_dir}")

        self.test_graphs = self._load_test_graphs()
        print(f"Loaded {len(self.test_graphs)} test graphs from {self.test_graphs_dir}")

    def _load_test_graphs(self) -> Dict[str, nx.Graph]:
        graphs: Dict[str, nx.Graph] = {}
        for file_path in self.test_graphs_dir.glob("*"):
            if file_path.suffix in ['.txt', '.edges', '.edgelist']:
                try:
                    if file_path.suffix == '.txt':
                        graph = self._read_txt_graph(file_path)
                    else:
                        graph = nx.read_edgelist(file_path, nodetype=int)
                    graphs[file_path.stem] = graph
                except Exception as e:
                    print(f"Warning: Could not load {file_path.name}: {e}")
            elif file_path.suffix == '.graphml':
                try:
                    graph = nx.read_graphml(file_path)
                    graphs[file_path.stem] = graph
                except Exception as e:
                    print(f"Warning: Could not load {file_path.name}: {e}")
        return graphs

    def _read_txt_graph(self, file_path: Path) -> nx.Graph:
        return txt_to_networkx(str(file_path))

    # ---------------- WP3.a ----------------
    def evaluate_kernelization_on_test_graphs(self,
                                              configurations: Optional[List[Dict]] = None) -> pd.DataFrame:
        print("\n" + "=" * 80)
        print("WP3.a: Testing Kernelization on Existing Test Graphs")
        print("=" * 80)

        if configurations is None:
            configurations = [
                {'name': 'Basic', 'preprocessing': False, 'smart_ordering': False},
                {'name': 'With Preprocessing', 'preprocessing': True, 'smart_ordering': False},
                {'name': 'Smart Ordering', 'preprocessing': False, 'smart_ordering': True},
                {'name': 'Full Optimization', 'preprocessing': True, 'smart_ordering': True}
            ]

        rows: List[Dict[str, Any]] = []

        for graph_name, graph in self.test_graphs.items():
            print(f"\nProcessing: {graph_name}  "
                  f"({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)")
            weights = self._create_weights_for_graph(graph)

            for config in configurations:
                instance = AdvancedClusterEditingInstance(graph.copy(), weights.copy())
                kernelizer = AdvancedKernelization(
                    instance,
                    use_preprocessing=config.get('preprocessing', False),
                    use_smart_ordering=config.get('smart_ordering', False),
                    use_parallel=config.get('parallel', True)
                )
                t0 = time.time()
                _ = kernelizer.kernelize(max_iterations=100)
                t = time.time() - t0
                stats = kernelizer.get_comprehensive_stats()

                rows.append({
                    'graph_name': graph_name,
                    'config': config['name'],
                    'original_nodes': graph.number_of_nodes(),
                    'original_edges': graph.number_of_edges(),
                    'kernel_nodes': stats['final_graph']['nodes'],
                    'kernel_edges': stats['final_graph']['edges'],
                    'reduction_ratio': stats['reduction_ratio'],
                    'iterations': stats['iterations'],
                    'time_seconds': t,
                    'vertices_merged': stats.get('vertices_merged', 0),
                    'forbidden_edges': stats.get('forbidden_edges', 0)
                })

        df = pd.DataFrame(rows)
        df.to_csv(self.output_dir / "test_graphs_kernelization.csv", index=False)
        self._print_effectiveness_summary(df)
        return df

    # ---------------- WP3.b ----------------
    def compare_with_without_kernelization(self) -> pd.DataFrame:
        print("\n" + "=" * 80)
        print("WP3.b: Comparing Performance With and Without Kernelization")
        print("=" * 80)

        results: List[Dict[str, Any]] = []
        jsonl_path = self.output_dir / "wp3b_runs.jsonl"
        open(jsonl_path, "w").close()  # truncate

        for graph_name, graph in self.test_graphs.items():
            if graph.number_of_nodes() > 500:
                print(f"Skipping {graph_name} (too large for comparison)")
                continue

            print(f"\nProcessing: {graph_name}")
            # --- WITHOUT K ---
            print("  Without kernelization...", end="")
            solver_no_kernel = ClusterEditingSolver(graph.copy())
            t0 = time.time()
            result_no_kernel = solver_no_kernel.solve(
                use_kernelization=False,
                clustering_algorithm='ilp',
                time_limit=self.ilp_time_limit,
                mip_gap=self.ilp_gap,
                threads=self.ilp_threads,
            )
            time_no_kernel = time.time() - t0
            print(f" done ({time_no_kernel:.3f}s)")

            # --- WITH K ---
            print("  With kernelization...", end="")
            solver_with_kernel = ClusterEditingSolver(graph.copy())
            t0 = time.time()
            result_with_kernel = solver_with_kernel.solve(
                use_kernelization=True,
                kernelization_type='optimized',
                clustering_algorithm='ilp',
                time_limit=self.ilp_time_limit,
                mip_gap=self.ilp_gap,
                threads=self.ilp_threads,
            )
            time_with_kernel = time.time() - t0
            print(f" done ({time_with_kernel:.3f}s)")

            kstats = result_with_kernel.get('kernel_stats') or {}
            kernel_nodes = (kstats.get('final_graph', {}) or {}).get('nodes', graph.number_of_nodes())

            # Zusatzmetriken
            kernel_reduction = 1.0 - (kernel_nodes / float(graph.number_of_nodes())) if graph.number_of_nodes() > 0 else 0.0
            cl_nok = result_no_kernel['clusters']
            cl_k = result_with_kernel['clusters']
            solution_change = _jaccard_pairs(cl_nok, cl_k)
            obj_gap_abs = abs(result_no_kernel['editing_cost'] - result_with_kernel['editing_cost'])

            row = {
                'graph_name': graph_name,
                'nodes': graph.number_of_nodes(),
                'edges': graph.number_of_edges(),
                'time_no_kernel': time_no_kernel,
                'time_with_kernel': time_with_kernel,
                'speedup': (time_no_kernel / time_with_kernel) if time_with_kernel > 0 else 1.0,
                'cost_no_kernel': result_no_kernel['editing_cost'],
                'cost_with_kernel': result_with_kernel['editing_cost'],
                'cost_ratio': (result_with_kernel['editing_cost'] / result_no_kernel['editing_cost']
                               if result_no_kernel['editing_cost'] > 0 else 1.0),
                'clusters_no_kernel': result_no_kernel['num_clusters'],
                'clusters_with_kernel': result_with_kernel['num_clusters'],
                'kernel_size': kernel_nodes,
                'kernel_reduction': kernel_reduction,
                'solution_change_jaccard': solution_change,
                'obj_gap_abs': obj_gap_abs,
                'solver': 'ilp'
            }
            results.append(row)
            with open(jsonl_path, "a") as jf:
                jf.write(json.dumps(row) + "\n")

            print(f"  Speedup: {row['speedup']:.2f}x")

        df = pd.DataFrame(results)
        df.to_csv(self.output_dir / "kernelization_improvements.csv", index=False)
        self._print_improvement_summary(df)
        self._create_improvement_plots(df)
        self._write_summary_report(jsonl_path, df)
        return df

    # ---------------- WP3.c ----------------
    def create_comparison_data_for_vcc(self) -> Dict[str, Any]:
        print("\n" + "=" * 80)
        print("Creating Comparison Data for VCC")
        print("=" * 80)

        comparison_data = {'graphs': {}, 'kernelization_stats': {}, 'solution_quality': {}}

        for graph_name, graph in self.test_graphs.items():
            print(f"Processing {graph_name}...")
            comparison_data['graphs'][graph_name] = {
                'nodes': graph.number_of_nodes(),
                'edges': graph.number_of_edges(),
                'density': nx.density(graph),
                'clustering_coefficient': nx.average_clustering(graph) if graph.number_of_nodes() > 0 else 0.0
            }

            weights = self._create_weights_for_graph(graph)
            instance = AdvancedClusterEditingInstance(graph.copy(), weights.copy())
            kernelizer = AdvancedKernelization(instance, use_preprocessing=True, use_smart_ordering=True)
            _ = kernelizer.kernelize()
            stats = kernelizer.get_comprehensive_stats()

            comparison_data['kernelization_stats'][graph_name] = {
                'reduction_ratio': stats['reduction_ratio'],
                'kernel_nodes': stats['final_graph']['nodes'],
                'kernel_edges': stats['final_graph']['edges'],
                'iterations': stats['iterations']
            }

            solver = ClusterEditingSolver(graph.copy())
            result = solver.solve(
                use_kernelization=True,
                kernelization_type='optimized',
                clustering_algorithm='ilp',
                time_limit=self.ilp_time_limit,
                mip_gap=self.ilp_gap,
                threads=self.ilp_threads,
            )

            comparison_data['solution_quality'][graph_name] = {
                'num_clusters': result['num_clusters'],
                'editing_cost': result['editing_cost'],
                'time_seconds': result['time_seconds']
            }

        with open(self.output_dir / "cluster_editing_results.json", 'w') as f:
            json.dump(comparison_data, f, indent=2)

        print(f"Comparison data saved to {self.output_dir / 'cluster_editing_results.json'}")
        return comparison_data

    # -------------- helpers --------------
    def _create_weights_for_graph(self, graph: nx.Graph) -> Dict[Tuple[int, int], float]:
        weights: Dict[Tuple[int, int], float] = {}
        for u, v, data in graph.edges(data=True):
            weights[(min(u, v), max(u, v))] = float(data.get('weight', 1.0))
        nodes = list(graph.nodes())
        for i, u in enumerate(nodes):
            for v in nodes[i + 1:]:
                e = (min(u, v), max(u, v))
                if e not in weights and not graph.has_edge(u, v):
                    weights[e] = -1.0
        return weights

    def _print_effectiveness_summary(self, df: pd.DataFrame):
        print("\n" + "=" * 60)
        print("KERNELIZATION EFFECTIVENESS SUMMARY")
        print("=" * 60)
        for config in df['config'].unique():
            config_df = df[df['config'] == config]
            print(f"\n{config}:")
            print(f"  Average reduction: {config_df['reduction_ratio'].mean():.1%}")
            print(f"  Best reduction: {config_df['reduction_ratio'].max():.1%}")
            print(f"  Average time: {config_df['time_seconds'].mean():.3f}s")

        print("\nBest reduced graphs:")
        best = df.groupby('graph_name')['reduction_ratio'].mean().sort_values(ascending=False).head(5)
        for gname, red in best.items():
            print(f"  {gname}: {red:.1%}")

    def _print_improvement_summary(self, df: pd.DataFrame):
        print("\n" + "=" * 60)
        print("IMPROVEMENT SUMMARY")
        print("=" * 60)
        print(f"Average speedup: {df['speedup'].mean():.2f}x")
        print(f"Max speedup: {df['speedup'].max():.2f}x")
        print(f"Average cost ratio: {df['cost_ratio'].mean():.3f}")
        print(f"Graphs with >2x speedup: {(df['speedup'] > 2).sum()}/{len(df)}")

    def _create_improvement_plots(self, df: pd.DataFrame):
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        ax = axes[0, 0]
        ax.scatter(df['nodes'], df['speedup'], alpha=0.7)
        ax.set_xlabel('Graph Size (nodes)')
        ax.set_ylabel('Speedup')
        ax.set_title('Speedup vs Graph Size')
        ax.axhline(y=1, color='r', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.scatter(df['nodes'], df['kernel_size'], alpha=0.7)
        mx = df['nodes'].max() if len(df) else 1
        ax.plot([0, mx], [0, mx], 'r--', alpha=0.5)
        ax.set_xlabel('Original Size (nodes)')
        ax.set_ylabel('Kernel Size (nodes)')
        ax.set_title('Kernel Reduction')
        ax.grid(True, alpha=0.3)

        ax = axes[0, 2]
        x = np.arange(len(df))
        width = 0.35
        ax.bar(x - width/2, df['time_no_kernel'], width, label='No Kernel', alpha=0.85)
        ax.bar(x + width/2, df['time_with_kernel'], width, label='With Kernel', alpha=0.85)
        ax.set_xlabel('Graph')
        ax.set_ylabel('Time (s)')
        ax.set_title('Runtime Comparison (ILP)')
        ax.set_xticks(x)
        ax.set_xticklabels(df['graph_name'], rotation=60, ha='right', fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.scatter(df['cost_no_kernel'], df['cost_with_kernel'], alpha=0.7)
        maxc = max(df['cost_no_kernel'].max(), df['cost_with_kernel'].max()) if len(df) else 1
        ax.plot([0, maxc], [0, maxc], 'r--', alpha=0.5)
        ax.set_xlabel('Cost without Kernelization')
        ax.set_ylabel('Cost with Kernelization')
        ax.set_title('Solution Quality Preservation (ILP)')
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        ax.hist(df['speedup'], bins=20, alpha=0.85)
        ax.set_xlabel('Speedup (NoK / K)')
        ax.set_ylabel('#Graphs')
        ax.set_title('Speedup Distribution')

        ax = axes[1, 2]
        ratio = df['kernel_size'] / df['nodes']
        ax.hist(ratio, bins=20, alpha=0.85)
        ax.set_xlabel('Kernel / Original')
        ax.set_ylabel('#Graphs')
        ax.set_title('Kernel Size Ratio')

        plt.tight_layout()
        plt.savefig(self.output_dir / "improvement_plots_extended.png", dpi=150)
        plt.close()
        print(f"Plots saved to {self.output_dir / 'improvement_plots_extended.png'}")

    def _write_summary_report(self, jsonl_path: Path, df: pd.DataFrame):
        summary = {
            "n_graphs": int(len(df)),
            "avg_speedup": float(df['speedup'].mean()) if len(df) else None,
            "median_speedup": float(df['speedup'].median()) if len(df) else None,
            "avg_cost_ratio": float(df['cost_ratio'].mean()) if len(df) else None,
            "solver": "ilp",
            "ilp_time_limit": self.ilp_time_limit,
            "ilp_gap": self.ilp_gap,
            "ilp_threads": self.ilp_threads,
        }
        with open(self.output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        lines = [
            "# WP3.b Summary Report",
            "",
            f"- Graphs: **{summary['n_graphs']}**",
            f"- Solver: **{summary['solver']}**",
            f"- Avg speedup: **{summary['avg_speedup']:.2f}x**" if summary["avg_speedup"] is not None else "-",
            f"- Median speedup: **{summary['median_speedup']:.2f}x**" if summary["median_speedup"] is not None else "-",
            f"- Avg cost ratio (K/NoK): **{summary['avg_cost_ratio']:.3f}**" if summary["avg_cost_ratio"] is not None else "-",
            "",
            "### Parameter",
            f"- time_limit: {self.ilp_time_limit}",
            f"- mip_gap: {self.ilp_gap}",
            f"- threads: {self.ilp_threads}",
            "",
            "### Datenquelle",
            f"- JSONL: `{jsonl_path}`",
            f"- CSV: `{self.output_dir / 'kernelization_improvements.csv'}`",
        ]
        (self.output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
        print(f"Summary saved to {self.output_dir / 'summary.md'}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="WP3 Evaluation using existing test graphs")
    parser.add_argument('--test-dir', type=str, default='test_graphs/generated/perturbed')
    parser.add_argument('--output-dir', type=str, default='results/wp3')
    parser.add_argument('--task', choices=['effectiveness', 'improvements', 'comparison', 'all'], default='all')
    parser.add_argument('--ilp-time-limit', type=float, default=None)
    parser.add_argument('--ilp-gap', type=float, default=None)
    parser.add_argument('--ilp-threads', type=int, default=None)
    args = parser.parse_args()

    try:
        evaluator = WP3TestGraphEvaluator(
            test_graphs_dir=args.test_dir,
            output_dir=args.output_dir,
            ilp_time_limit=args.ilp_time_limit,
            ilp_gap=args.ilp_gap,
            ilp_threads=args.ilp_threads,
        )
    except ValueError as e:
        print(f"Error: {e}")
        print(f"Please ensure test graphs are in {args.test_dir}")
        return

    if args.task in ['effectiveness', 'all']:
        evaluator.evaluate_kernelization_on_test_graphs()
    if args.task in ['improvements', 'all']:
        evaluator.compare_with_without_kernelization()
    if args.task in ['comparison', 'all']:
        evaluator.create_comparison_data_for_vcc()

    print("\n" + "=" * 60)
    print("WP3 EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
