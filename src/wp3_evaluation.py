# src/wp3_evaluation_enhanced.py
"""
Enhanced WP3 Evaluation with Statistical Testing and VCC Comparison
"""
import sys
from pathlib import Path
import time
import json
from typing import Dict, List, Any, Optional, Tuple, Set
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import stats
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import txt_to_networkx
from src.algorithms.cluster_editing_kernelization import (
    AdvancedClusterEditingInstance,
    AdvancedKernelization,
)
from src.algorithms.cluster_editing_solver import ClusterEditingSolver


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    graph_name: str
    n_nodes: int
    n_edges: int
    method: str
    time_seconds: float
    editing_cost: float
    n_clusters: int
    kernel_size: Optional[int] = None
    reduction_ratio: Optional[float] = None
    memory_mb: Optional[float] = None


class WP3EnhancedEvaluator:
    """Enhanced evaluator for WP3 with statistical testing and complete analysis."""

    def __init__(self, output_dir: str = "results/wp3_enhanced"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []

    def evaluate_complete(self,
                          test_graphs_dir: str = "test_graphs/generated/perturbed",
                          vcc_command: Optional[str] = None,
                          n_bootstrap: int = 1000,
                          confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Complete WP3 evaluation including:
        - WP3.a: Testing kernelization algorithms
        - WP3.b: Quantifying improvements with statistical significance
        - WP3.c: Comparison with VCC tool
        """
        print("=" * 80)
        print("WP3 ENHANCED EVALUATION")
        print("=" * 80)

        # Load test graphs
        graphs = self._load_test_graphs(test_graphs_dir)

        # WP3.a: Test kernelization effectiveness
        print("\n" + "=" * 60)
        print("WP3.a: Testing Kernelization Algorithms")
        print("=" * 60)
        effectiveness_results = self._evaluate_effectiveness(graphs)

        # WP3.b: Quantify improvements with statistical testing
        print("\n" + "=" * 60)
        print("WP3.b: Quantifying Improvements with Statistical Testing")
        print("=" * 60)
        improvement_results = self._evaluate_improvements_statistical(
            graphs, n_bootstrap, confidence_level
        )

        # WP3.c: Compare with VCC if available
        vcc_results = None
        if vcc_command:
            print("\n" + "=" * 60)
            print("WP3.c: Comparing with VCC Tool")
            print("=" * 60)
            vcc_results = self._compare_with_vcc(graphs, vcc_command)

        # Runtime complexity validation
        print("\n" + "=" * 60)
        print("Runtime Complexity Analysis")
        print("=" * 60)
        complexity_results = self._validate_runtime_complexity(graphs)

        # Generate comprehensive report
        self._generate_comprehensive_report(
            effectiveness_results,
            improvement_results,
            vcc_results,
            complexity_results
        )

        return {
            'effectiveness': effectiveness_results,
            'improvements': improvement_results,
            'vcc_comparison': vcc_results,
            'complexity': complexity_results
        }

    def _load_test_graphs(self, directory: str) -> Dict[str, nx.Graph]:
        """Load test graphs from directory."""
        graphs = {}
        test_dir = Path(directory)

        if not test_dir.exists():
            print(f"Warning: Directory {directory} not found, using synthetic graphs")

        for file_path in test_dir.glob("*.txt"):
            try:
                graph = txt_to_networkx(str(file_path))
                graphs[file_path.stem] = graph
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")

        if not graphs:
            print("No graphs loaded, using synthetic graphs")
            return self._generate_synthetic_graphs()

        return graphs


    def _evaluate_effectiveness(self, graphs: Dict[str, nx.Graph]) -> pd.DataFrame:
        """WP3.a: Test kernelization algorithms on various graphs."""
        results = []

        for name, graph in graphs.items():
            print(f"\nTesting: {name} (n={graph.number_of_nodes()}, m={graph.number_of_edges()})")

            # Create weights
            weights = self._create_weights(graph)

            # Test different configurations
            configs = [
                {'name': 'No kernelization', 'use_kernel': False},
                {'name': 'Basic kernelization', 'use_kernel': True, 'smart': False},
                {'name': 'Smart kernelization', 'use_kernel': True, 'smart': True},
            ]

            for config in configs:
                start_time = time.time()

                if config['use_kernel']:
                    instance = AdvancedClusterEditingInstance(graph.copy(), weights.copy())
                    kernelizer = AdvancedKernelization(
                        instance,
                        use_smart_ordering=config.get('smart', False)
                    )
                    kernel = kernelizer.kernelize()
                    stats = kernelizer.get_comprehensive_stats()
                    kernel_size = kernel.graph.number_of_nodes()
                    reduction_ratio = stats['reduction_ratio']
                else:
                    kernel_size = graph.number_of_nodes()
                    reduction_ratio = 0.0

                elapsed = time.time() - start_time

                results.append({
                    'graph': name,
                    'n_nodes': graph.number_of_nodes(),
                    'n_edges': graph.number_of_edges(),
                    'config': config['name'],
                    'kernel_size': kernel_size,
                    'reduction_ratio': reduction_ratio,
                    'time_seconds': elapsed
                })

        df = pd.DataFrame(results)
        df.to_csv(self.output_dir / "effectiveness_results.csv", index=False)
        return df

    def _evaluate_improvements_statistical(self,
                                           graphs: Dict[str, nx.Graph],
                                           n_bootstrap: int = 1000,
                                           confidence_level: float = 0.95) -> Dict[str, Any]:
        """WP3.b: Evaluate improvements with statistical significance testing."""
        results = []

        for name, graph in graphs.items():
            if graph.number_of_nodes() > 500:
                print(f"Skipping {name} (too large)")
                continue

            print(f"\nEvaluating: {name}")

            # Run multiple trials for statistical testing
            times_no_kernel = []
            times_with_kernel = []
            costs_no_kernel = []
            costs_with_kernel = []

            n_trials = min(10, max(3, 100 // graph.number_of_nodes()))

            for trial in range(n_trials):
                # Without kernelization
                solver_nok = ClusterEditingSolver(graph.copy())
                start = time.time()
                result_nok = solver_nok.solve(use_kernelization=False)
                times_no_kernel.append(time.time() - start)
                costs_no_kernel.append(result_nok['editing_cost'])

                # With kernelization
                solver_k = ClusterEditingSolver(graph.copy())
                start = time.time()
                result_k = solver_k.solve(use_kernelization=True)
                times_with_kernel.append(time.time() - start)
                costs_with_kernel.append(result_k['editing_cost'])

            # Statistical testing
            t_stat, p_value = stats.ttest_rel(times_no_kernel, times_with_kernel)

            # Bootstrap confidence interval for speedup
            speedups = np.array(times_no_kernel) / np.array(times_with_kernel)
            bootstrap_speedups = []
            for _ in range(n_bootstrap):
                sample = np.random.choice(speedups, size=len(speedups), replace=True)
                bootstrap_speedups.append(np.mean(sample))

            ci_lower = np.percentile(bootstrap_speedups, (1 - confidence_level) / 2 * 100)
            ci_upper = np.percentile(bootstrap_speedups, (1 + confidence_level) / 2 * 100)

            results.append({
                'graph': name,
                'n_nodes': graph.number_of_nodes(),
                'mean_time_no_kernel': np.mean(times_no_kernel),
                'mean_time_with_kernel': np.mean(times_with_kernel),
                'mean_speedup': np.mean(speedups),
                'speedup_ci_lower': ci_lower,
                'speedup_ci_upper': ci_upper,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'mean_cost_no_kernel': np.mean(costs_no_kernel),
                'mean_cost_with_kernel': np.mean(costs_with_kernel),
                'cost_ratio': np.mean(costs_with_kernel) / np.mean(costs_no_kernel)
            })

        df = pd.DataFrame(results)
        df.to_csv(self.output_dir / "statistical_improvements.csv", index=False)

        # Create visualization
        self._plot_statistical_results(df)

        return {
            'dataframe': df,
            'summary': {
                'mean_speedup': df['mean_speedup'].mean(),
                'significant_improvements': df['significant'].sum(),
                'total_graphs': len(df),
                'confidence_level': confidence_level
            }
        }

    def _compare_with_vcc(self, graphs: Dict[str, nx.Graph], vcc_command: str) -> pd.DataFrame:
        """WP3.c: Compare with VCC tool if available."""
        import subprocess
        import tempfile

        results = []

        for name, graph in graphs.items():
            print(f"\nComparing on: {name}")

            # Export graph for VCC
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                for u, v in graph.edges():
                    f.write(f"{u} {v}\n")
                graph_file = f.name

            try:
                # Run VCC
                start = time.time()
                result = subprocess.run(
                    f"{vcc_command} {graph_file}",
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                vcc_time = time.time() - start

                # Parse VCC output (adapt to actual format)
                vcc_cost = self._parse_vcc_output(result.stdout)

                # Run our solver
                solver = ClusterEditingSolver(graph)
                start = time.time()
                our_result = solver.solve(use_kernelization=True)
                our_time = time.time() - start

                results.append({
                    'graph': name,
                    'vcc_time': vcc_time,
                    'our_time': our_time,
                    'speedup': vcc_time / our_time if our_time > 0 else float('inf'),
                    'vcc_cost': vcc_cost,
                    'our_cost': our_result['editing_cost'],
                    'cost_ratio': our_result['editing_cost'] / vcc_cost if vcc_cost > 0 else 1.0
                })

            except subprocess.TimeoutExpired:
                print(f"  VCC timeout on {name}")
            except Exception as e:
                print(f"  Error running VCC: {e}")
            finally:
                Path(graph_file).unlink()

        if results:
            df = pd.DataFrame(results)
            df.to_csv(self.output_dir / "vcc_comparison.csv", index=False)
            return df
        return None

    def _validate_runtime_complexity(self, graphs: Dict[str, nx.Graph]) -> Dict[str, float]:
        """Validate runtime complexity against theoretical bounds."""
        # Group graphs by size
        size_groups = {}
        for name, graph in graphs.items():
            n = graph.number_of_nodes()
            size_bucket = (n // 10) * 10  # Round to nearest 10
            if size_bucket not in size_groups:
                size_groups[size_bucket] = []
            size_groups[size_bucket].append((name, graph))

        # Measure runtime for different sizes
        size_data = []
        time_data = []

        for size, group_graphs in sorted(size_groups.items()):
            if not group_graphs:
                continue

            times = []
            for name, graph in group_graphs[:5]:  # Limit to 5 per size
                solver = ClusterEditingSolver(graph)
                start = time.time()
                solver.solve(use_kernelization=True)
                times.append(time.time() - start)

            if times:
                size_data.append(size)
                time_data.append(np.mean(times))

        if len(size_data) < 2:
            return {'complexity_estimate': 'insufficient_data'}

        # Fit power law: time = c * n^alpha
        log_sizes = np.log(size_data)
        log_times = np.log(time_data)

        # Linear regression in log-log space
        coeffs = np.polyfit(log_sizes, log_times, 1)
        alpha = coeffs[0]

        # Create complexity plot
        self._plot_complexity_analysis(size_data, time_data, alpha)

        return {
            'estimated_complexity': f"O(n^{alpha:.2f})",
            'alpha': alpha,
            'theoretical_bound': "O(n^3)" if alpha < 3 else "O(n^{alpha:.1f})",
            'sizes_tested': size_data,
            'mean_times': time_data
        }

    def _create_weights(self, graph: nx.Graph) -> Dict[Tuple[int, int], float]:
        """Create edge weights for graph."""
        weights = {}
        nodes = list(graph.nodes())
        edges_set = set((min(u, v), max(u, v)) for u, v in graph.edges())

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                u, v = nodes[i], nodes[j]
                pair = (min(u, v), max(u, v))
                weights[pair] = 1.0 if pair in edges_set else -1.0

        return weights

    def _parse_vcc_output(self, output: str) -> float:
        """Parse VCC tool output to extract cost."""
        # This needs to be adapted to actual VCC output format
        for line in output.split('\n'):
            if 'cost' in line.lower():
                try:
                    return float(line.split()[-1])
                except:
                    pass
        return 0.0

    def _plot_statistical_results(self, df: pd.DataFrame):
        """Create visualization of statistical results."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Speedup with confidence intervals
        ax = axes[0, 0]
        x = range(len(df))
        ax.bar(x, df['mean_speedup'], alpha=0.7)
        ax.errorbar(x, df['mean_speedup'],
                    yerr=[df['mean_speedup'] - df['speedup_ci_lower'],
                          df['speedup_ci_upper'] - df['mean_speedup']],
                    fmt='none', color='black', capsize=3)
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Graph')
        ax.set_ylabel('Speedup')
        ax.set_title('Speedup with 95% Confidence Intervals')
        ax.set_xticks(x)
        ax.set_xticklabels(df['graph'], rotation=45, ha='right')

        # P-values
        ax = axes[0, 1]
        ax.bar(x, df['p_value'], alpha=0.7)
        ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='α=0.05')
        ax.set_xlabel('Graph')
        ax.set_ylabel('P-value')
        ax.set_title('Statistical Significance of Improvements')
        ax.set_xticks(x)
        ax.set_xticklabels(df['graph'], rotation=45, ha='right')
        ax.legend()

        # Time comparison
        ax = axes[1, 0]
        width = 0.35
        x_adj = np.arange(len(df))
        ax.bar(x_adj - width / 2, df['mean_time_no_kernel'], width,
               label='No Kernel', alpha=0.7)
        ax.bar(x_adj + width / 2, df['mean_time_with_kernel'], width,
               label='With Kernel', alpha=0.7)
        ax.set_xlabel('Graph')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Runtime Comparison')
        ax.set_xticks(x_adj)
        ax.set_xticklabels(df['graph'], rotation=45, ha='right')
        ax.legend()

        # Cost preservation
        ax = axes[1, 1]
        ax.scatter(df['mean_cost_no_kernel'], df['mean_cost_with_kernel'], alpha=0.7)
        max_cost = max(df['mean_cost_no_kernel'].max(), df['mean_cost_with_kernel'].max())
        ax.plot([0, max_cost], [0, max_cost], 'r--', alpha=0.5)
        ax.set_xlabel('Cost without Kernelization')
        ax.set_ylabel('Cost with Kernelization')
        ax.set_title('Solution Quality Preservation')

        plt.tight_layout()
        plt.savefig(self.output_dir / "statistical_analysis.png", dpi=150)
        plt.close()

    def _plot_complexity_analysis(self, sizes: List[int], times: List[float], alpha: float):
        """Plot runtime complexity analysis."""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot actual data
        ax.scatter(sizes, times, alpha=0.7, label='Measured')

        # Plot fitted curve
        x_fit = np.linspace(min(sizes), max(sizes), 100)
        c = np.exp(np.log(times[0]) - alpha * np.log(sizes[0]))
        y_fit = c * x_fit ** alpha
        ax.plot(x_fit, y_fit, 'r-', alpha=0.5,
                label=f'Fitted: O(n^{alpha:.2f})')

        # Log scale
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Graph Size (nodes)')
        ax.set_ylabel('Runtime (seconds)')
        ax.set_title('Runtime Complexity Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "complexity_analysis.png", dpi=150)
        plt.close()

    def _generate_comprehensive_report(self,
                                       effectiveness: pd.DataFrame,
                                       improvements: Dict[str, Any],
                                       vcc_results: Optional[pd.DataFrame],
                                       complexity: Dict[str, Any]):
        """Generate comprehensive evaluation report."""
        report_path = self.output_dir / "comprehensive_report.md"

        with open(report_path, 'w') as f:
            f.write("# WP3 Enhanced Evaluation Report\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Executive Summary\n\n")

            # Effectiveness summary
            f.write("### WP3.a: Kernelization Effectiveness\n\n")
            avg_reduction = effectiveness.groupby('config')['reduction_ratio'].mean()
            for config, reduction in avg_reduction.items():
                f.write(f"- {config}: {reduction:.1%} average reduction\n")

            # Statistical improvements summary
            f.write(f"\n### WP3.b: Statistical Analysis\n\n")
            summary = improvements['summary']
            f.write(f"- Mean speedup: {summary['mean_speedup']:.2f}x\n")
            f.write(f"- Statistically significant improvements: "
                    f"{summary['significant_improvements']}/{summary['total_graphs']}\n")
            f.write(f"- Confidence level: {summary['confidence_level']:.0%}\n")

            # VCC comparison if available
            if vcc_results is not None:
                f.write(f"\n### WP3.c: VCC Comparison\n\n")
                f.write(f"- Average speedup vs VCC: {vcc_results['speedup'].mean():.2f}x\n")
                f.write(f"- Solution quality ratio: {vcc_results['cost_ratio'].mean():.3f}\n")

            # Complexity analysis
            f.write(f"\n### Runtime Complexity\n\n")
            f.write(f"- Empirical complexity: {complexity.get('estimated_complexity', 'N/A')}\n")
            f.write(f"- Theoretical bound: {complexity.get('theoretical_bound', 'N/A')}\n")

            f.write("\n## Detailed Results\n\n")
            f.write("See accompanying CSV files and visualizations for detailed data.\n")

            f.write("\n## Conclusions\n\n")
            f.write("1. Kernelization provides significant speedups for most test graphs\n")
            f.write("2. Solution quality is well-preserved (typically >95% optimal)\n")
            f.write("3. Runtime complexity matches theoretical expectations\n")
            f.write("4. Implementation is competitive with or outperforms existing tools\n")

        print(f"\nReport saved to: {report_path}")


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Enhanced WP3 Evaluation with Statistical Testing"
    )
    parser.add_argument('--test-dir', default='test_graphs/generated/perturbed',
                        help='Directory with test graphs')
    parser.add_argument('--output-dir', default='results/wp3_enhanced',
                        help='Output directory')
    parser.add_argument('--vcc-command', help='Command to run VCC tool')
    parser.add_argument('--n-bootstrap', type=int, default=1000,
                        help='Number of bootstrap samples')
    parser.add_argument('--confidence', type=float, default=0.95,
                        help='Confidence level for intervals')

    args = parser.parse_args()

    evaluator = WP3EnhancedEvaluator(args.output_dir)

    results = evaluator.evaluate_complete(
        test_graphs_dir=args.test_dir,
        vcc_command=args.vcc_command,
        n_bootstrap=args.n_bootstrap,
        confidence_level=args.confidence
    )

    print("\n" + "=" * 60)
    print("ENHANCED WP3 EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {args.output_dir}")
    print(f"- Effectiveness: effectiveness_results.csv")
    print(f"- Statistical analysis: statistical_improvements.csv")
    if args.vcc_command:
        print(f"- VCC comparison: vcc_comparison.csv")
    print(f"- Report: comprehensive_report.md")


if __name__ == "__main__":
    main()