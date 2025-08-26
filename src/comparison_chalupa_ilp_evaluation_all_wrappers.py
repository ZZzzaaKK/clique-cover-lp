"""
WP1.c Evaluation: Comprehensive comparison of all solver variants
Extends the original comparison to include reduced and interactive reduced ILP methods
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional
import networkx as nx
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.append('src')

# Import project modules
from src.wrapperV2 import (
    ilp_wrapper,
    reduced_ilp_wrapper,
    chalupa_wrapper,
    interactive_reduced_ilp_wrapper,
    _chalupa_warmstart
)
from src.utils import txt_to_networkx, get_value
from src.simulator import GraphGenerator, GraphConfig
from src.utils_metrics import (set_global_seeds, safe_ratio, rel_change,
                               clean_for_plot, nanmean, safe_idxmax, should_kernelize, estimate_loglog_slope)

set_global_seeds(33)

# Configure plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class WP1cEvaluator:
    """Main evaluation class for WP1.c comparisons"""

    def __init__(self, output_dir="evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def evaluate_single_instance(self, filepath: str, timeout: int = 300) -> Dict:
        """
        Evaluate a single graph instance with both Chalupa and ILP.

        Returns dict with:
        - filename, n_nodes, n_edges, density
        - chalupa_theta, chalupa_time
        - ilp_theta, ilp_time, ilp_status
        - quality_ratio (chalupa/ilp)
        """
        result = {'filepath': filepath}

        # Load graph and get basic properties
        try:
            G = txt_to_networkx(filepath)
            result['n_nodes'] = G.number_of_nodes()
            result['n_edges'] = G.number_of_edges()
            result['density'] = nx.density(G)

            # Extract perturbation level from filename if available
            import re
            match = re.search(r'r(\d+)', Path(filepath).stem)
            if match:
                result['perturbation'] = int(match.group(1)) / 100
            else:
                result['perturbation'] = None

        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None

        # Run Chalupa heuristic
        try:
            start = time.time()
            chalupa_result = chalupa_wrapper(filepath)
            chalupa_time = time.time() - start

            result['chalupa_theta'] = chalupa_result if chalupa_result else None
            result['chalupa_time'] = chalupa_time
        except Exception as e:
            print(f"Chalupa failed on {filepath}: {e}")
            result['chalupa_theta'] = None
            result['chalupa_time'] = None

        # Run ILP solver (with timeout)
        try:
            ilp_res = ilp_wrapper(
                filepath,
                use_warmstart=False,  # Fair comparison
                time_limit=timeout,
                mip_gap=0.01,
                verbose=False,
                return_assignment=False
            )

            if isinstance(ilp_res, dict):
                result['ilp_theta'] = ilp_res.get('theta')
                result['ilp_time'] = ilp_res.get('time', None)
                result['ilp_status'] = ilp_res.get('status', 'unknown')
                result['ilp_gap'] = ilp_res.get('gap', None)
            else:
                result['ilp_theta'] = ilp_res
                result['ilp_time'] = None
                result['ilp_status'] = 'solved'
                result['ilp_gap'] = 0.0

        except Exception as e:
            print(f"ILP failed on {filepath}: {e}")
            result['ilp_theta'] = None
            result['ilp_time'] = None
            result['ilp_status'] = 'failed'

        # Calculate quality metrics
        if result['chalupa_theta'] and result['ilp_theta']:
            result['quality_ratio'] = result['chalupa_theta'] / result['ilp_theta']
            result['absolute_gap'] = result['chalupa_theta'] - result['ilp_theta']
        else:
            result['quality_ratio'] = None
            result['absolute_gap'] = None

        return result

    def run_evaluation_suite(self, test_dir: str = "test_graphs/generated"):
        """Run evaluation on all test instances"""
        print(f"Starting evaluation on {test_dir}")

        # Find all test files
        test_files = list(Path(test_dir).glob("**/*.txt"))
        print(f"Found {len(test_files)} test instances")

        # Evaluate each instance
        for i, filepath in enumerate(test_files):
            print(f"[{i + 1}/{len(test_files)}] Evaluating {filepath.name}...")
            result = self.evaluate_single_instance(str(filepath))
            if result:
                self.results.append(result)

        # Convert to DataFrame for analysis
        self.df = pd.DataFrame(self.results)

        # Save raw results
        self.save_results()

        return self.df

    def generate_test_cases_with_varying_perturbation(self):
        """Generate test cases with systematic perturbation variation"""
        print("Generating test cases with varying perturbation levels...")

        output_dir = Path("test_graphs/perturbation_study")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Fixed parameters
        num_cliques = 5
        clique_size = 8

        # Vary perturbation from 0% to 90%
        perturbation_levels = np.arange(0, 1.0, 0.1)

        for pert in perturbation_levels:
            config = GraphConfig(
                num_cliques=num_cliques,
                distribution_type="uniform",
                uniform_size=clique_size,
                edge_removal_prob=pert,
                edge_addition_prob=pert / 4  # Keep ratio consistent
            )

            result = GraphGenerator.generate_test_case(config)
            G_original, G_perturbed, communities, _, _ = result

            # Save perturbed graph
            filename = f"uniform_n{num_cliques}_s{clique_size}_p{int(pert * 100):03d}.txt"
            filepath = output_dir / filename

            with open(filepath, 'w') as f:
                for node in G_perturbed.nodes():
                    neighbors = list(G_perturbed.neighbors(node))
                    neighbors_str = ' '.join(map(str, neighbors))
                    f.write(f"{node}: {neighbors_str}\n")
                f.write("\n")
                f.write(f"Perturbation: {pert:.2f}\n")
                f.write(f"Number of Vertices: {G_perturbed.number_of_nodes()}\n")
                f.write(f"Number of Edges: {G_perturbed.number_of_edges()}\n")

        print(f"Generated {len(perturbation_levels)} test cases in {output_dir}")

    def create_runtime_plots(self):
        """Create runtime comparison plots"""
        if self.df.empty:
            print("No data available for plotting")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Runtime vs Problem Size (nodes)
        ax = axes[0, 0]
        valid_data = self.df.dropna(subset=['chalupa_time', 'ilp_time', 'n_nodes'])

        ax.scatter(valid_data['n_nodes'], valid_data['chalupa_time'],
                   alpha=0.6, label='Chalupa', s=50)
        ax.scatter(valid_data['n_nodes'], valid_data['ilp_time'],
                   alpha=0.6, label='ILP', s=50)
        ax.set_xlabel('Number of Nodes')
        ax.set_ylabel('Runtime (seconds)')
        ax.set_title('Runtime vs Problem Size')
        ax.legend()
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        # 2. Runtime vs Density
        ax = axes[0, 1]
        ax.scatter(valid_data['density'], valid_data['chalupa_time'],
                   alpha=0.6, label='Chalupa', s=50)
        ax.scatter(valid_data['density'], valid_data['ilp_time'],
                   alpha=0.6, label='ILP', s=50)
        ax.set_xlabel('Graph Density')
        ax.set_ylabel('Runtime (seconds)')
        ax.set_title('Runtime vs Graph Density')
        ax.legend()
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        # 3. Runtime Speedup
        ax = axes[1, 0]
        speedup = valid_data['ilp_time'] / valid_data['chalupa_time']
        ax.hist(speedup, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(x=1, color='red', linestyle='--', label='No speedup')
        ax.set_xlabel('Speedup Factor (ILP time / Chalupa time)')
        ax.set_ylabel('Frequency')
        ax.set_title('Chalupa Speedup Distribution')
        ax.legend()
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)

        # 4. Box plot comparison
        ax = axes[1, 1]
        runtime_data = pd.DataFrame({
            'Chalupa': valid_data['chalupa_time'],
            'ILP': valid_data['ilp_time']
        })
        runtime_data.plot(kind='box', ax=ax, grid=True)
        ax.set_ylabel('Runtime (seconds)')
        ax.set_title('Runtime Distribution Comparison')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        plt.suptitle('Runtime Analysis: Chalupa vs ILP', fontsize=16, y=1.02)
        plt.tight_layout()

        output_path = self.output_dir / f"runtime_analysis_{self.timestamp}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved runtime plots to {output_path}")
        plt.show()

    def create_quality_plots(self):
        """Create solution quality comparison plots"""
        if self.df.empty:
            print("No data available for plotting")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Filter for valid comparisons
        valid_data = self.df.dropna(subset=['quality_ratio', 'n_nodes'])

        # 1. Quality Ratio Distribution
        ax = axes[0, 0]
        ax.hist(valid_data['quality_ratio'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        ax.axvline(x=1.0, color='green', linestyle='--', linewidth=2, label='Optimal (ratio=1)')
        ax.set_xlabel('Quality Ratio (Chalupa θ / ILP θ)')
        ax.set_ylabel('Frequency')
        ax.set_title('Solution Quality Distribution')
        mean_ratio = valid_data['quality_ratio'].mean()
        ax.axvline(x=mean_ratio, color='red', linestyle=':', linewidth=2,
                   label=f'Mean = {mean_ratio:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Quality vs Problem Size
        ax = axes[0, 1]
        scatter = ax.scatter(valid_data['n_nodes'], valid_data['quality_ratio'],
                             c=valid_data['density'], cmap='viridis', s=50, alpha=0.6)
        ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5)
        ax.set_xlabel('Number of Nodes')
        ax.set_ylabel('Quality Ratio')
        ax.set_title('Solution Quality vs Problem Size')
        plt.colorbar(scatter, ax=ax, label='Density')
        ax.grid(True, alpha=0.3)

        # 3. Absolute Gap Distribution
        ax = axes[1, 0]
        absolute_gaps = valid_data['absolute_gap']
        ax.hist(absolute_gaps, bins=30, edgecolor='black', alpha=0.7, color='coral')
        ax.set_xlabel('Absolute Gap (Chalupa θ - ILP θ)')
        ax.set_ylabel('Frequency')
        ax.set_title('Absolute Gap Distribution')
        ax.axvline(x=0, color='green', linestyle='--', linewidth=2, label='Optimal (gap=0)')
        mean_gap = absolute_gaps.mean()
        ax.axvline(x=mean_gap, color='red', linestyle=':', linewidth=2,
                   label=f'Mean = {mean_gap:.2f}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Success Rate Analysis
        ax = axes[1, 1]

        # Calculate success rates for different thresholds
        thresholds = [1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3]
        success_rates = []

        for threshold in thresholds:
            success_rate = (valid_data['quality_ratio'] <= threshold).mean() * 100
            success_rates.append(success_rate)

        ax.plot(thresholds, success_rates, marker='o', linewidth=2, markersize=8)
        ax.set_xlabel('Quality Threshold')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Chalupa Success Rate at Different Quality Thresholds')
        ax.grid(True, alpha=0.3)

        # Add annotations
        for i, (t, s) in enumerate(zip(thresholds, success_rates)):
            if i % 2 == 0:  # Annotate every other point to avoid clutter
                ax.annotate(f'{s:.1f}%', xy=(t, s), xytext=(5, 5),
                            textcoords='offset points', fontsize=9)

        plt.suptitle('Solution Quality Analysis: Chalupa vs ILP', fontsize=16, y=1.02)
        plt.tight_layout()

        output_path = self.output_dir / f"quality_analysis_{self.timestamp}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved quality plots to {output_path}")
        plt.show()

    def create_perturbation_analysis(self):
        """Analyze effect of perturbation strength on algorithm performance"""
        # First generate test cases if needed
        perturbation_dir = Path("test_graphs/perturbation_study")
        if not perturbation_dir.exists() or len(list(perturbation_dir.glob("*.txt"))) < 5:
            self.generate_test_cases_with_varying_perturbation()

        # Evaluate perturbation study instances
        print("Evaluating perturbation study instances...")
        pert_results = []

        for filepath in sorted(perturbation_dir.glob("*.txt")):
            result = self.evaluate_single_instance(str(filepath), timeout=60)
            if result:
                # Extract perturbation level from filename
                import re
                match = re.search(r'p(\d{3})', filepath.stem)
                if match:
                    result['perturbation'] = int(match.group(1)) / 100
                pert_results.append(result)

        if not pert_results:
            print("No perturbation results available")
            return

        pert_df = pd.DataFrame(pert_results)

        # Create perturbation analysis plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Quality vs Perturbation
        ax = axes[0, 0]
        valid = pert_df.dropna(subset=['perturbation', 'quality_ratio'])

        if not valid.empty:
            ax.plot(valid['perturbation'] * 100, valid['quality_ratio'],
                    marker='o', linewidth=2, markersize=8, color='darkblue')
            ax.fill_between(valid['perturbation'] * 100, 1.0, valid['quality_ratio'],
                            alpha=0.3, color='lightblue')
            ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Optimal')
            ax.set_xlabel('Perturbation Level (%)')
            ax.set_ylabel('Quality Ratio (Chalupa/ILP)')
            ax.set_title('Solution Quality vs Perturbation Strength')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 2. Runtime vs Perturbation
        ax = axes[0, 1]
        valid = pert_df.dropna(subset=['perturbation', 'chalupa_time', 'ilp_time'])

        if not valid.empty:
            ax.plot(valid['perturbation'] * 100, valid['chalupa_time'],
                    marker='o', label='Chalupa', linewidth=2, markersize=8)
            ax.plot(valid['perturbation'] * 100, valid['ilp_time'],
                    marker='s', label='ILP', linewidth=2, markersize=8)
            ax.set_xlabel('Perturbation Level (%)')
            ax.set_ylabel('Runtime (seconds)')
            ax.set_title('Runtime vs Perturbation Strength')
            ax.legend()
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)

        # 3. Absolute θ values vs Perturbation
        ax = axes[1, 0]
        valid = pert_df.dropna(subset=['perturbation', 'chalupa_theta', 'ilp_theta'])

        if not valid.empty:
            ax.plot(valid['perturbation'] * 100, valid['chalupa_theta'],
                    marker='o', label='Chalupa θ', linewidth=2, markersize=8)
            ax.plot(valid['perturbation'] * 100, valid['ilp_theta'],
                    marker='s', label='ILP θ (optimal)', linewidth=2, markersize=8)
            ax.fill_between(valid['perturbation'] * 100,
                            valid['ilp_theta'], valid['chalupa_theta'],
                            alpha=0.3, color='red', label='Gap')
            ax.set_xlabel('Perturbation Level (%)')
            ax.set_ylabel('Clique Cover Number θ(G)')
            ax.set_title('Clique Cover Number vs Perturbation')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 4. ILP Feasibility Analysis
        ax = axes[1, 1]

        # Group by perturbation level and calculate statistics
        if 'perturbation' in pert_df.columns:
            pert_groups = pert_df.groupby('perturbation')

            pert_levels = []
            ilp_success_rates = []
            avg_gaps = []

            for pert, group in pert_groups:
                pert_levels.append(pert * 100)
                # Success rate (ILP solved optimally)
                success_rate = (group['ilp_status'] == 'optimal').mean() * 100 if 'ilp_status' in group else 0
                ilp_success_rates.append(success_rate)
                # Average MIP gap
                avg_gap = group['ilp_gap'].mean() if 'ilp_gap' in group else None
                avg_gaps.append(avg_gap * 100 if avg_gap else 0)

            ax2 = ax.twinx()

            line1 = ax.plot(pert_levels, ilp_success_rates,
                            marker='o', color='green', linewidth=2,
                            markersize=8, label='ILP Success Rate')
            line2 = ax2.plot(pert_levels, avg_gaps,
                             marker='s', color='orange', linewidth=2,
                             markersize=8, label='Avg MIP Gap')

            ax.set_xlabel('Perturbation Level (%)')
            ax.set_ylabel('ILP Success Rate (%)', color='green')
            ax2.set_ylabel('Average MIP Gap (%)', color='orange')
            ax.set_title('ILP Solver Performance vs Perturbation')
            ax.tick_params(axis='y', labelcolor='green')
            ax2.tick_params(axis='y', labelcolor='orange')

            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='best')
            ax.grid(True, alpha=0.3)

        plt.suptitle('Perturbation Strength Analysis', fontsize=16, y=1.02)
        plt.tight_layout()

        output_path = self.output_dir / f"perturbation_analysis_{self.timestamp}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved perturbation plots to {output_path}")
        plt.show()

    def save_results(self):
        """Save evaluation results to files"""
        if not self.results:
            print("No results to save")
            return

        # Save as JSON
        json_path = self.output_dir / f"evaluation_results_{self.timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"Saved results to {json_path}")

        # Save as CSV
        if hasattr(self, 'df') and not self.df.empty:
            csv_path = self.output_dir / f"evaluation_results_{self.timestamp}.csv"
            self.df.to_csv(csv_path, index=False)
            print(f"Saved CSV to {csv_path}")

    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        if not hasattr(self, 'df') or self.df.empty:
            print("No data available for summary")
            return

        report_path = self.output_dir / f"summary_report_{self.timestamp}.txt"

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("WP1.c EVALUATION SUMMARY REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            # Overall statistics
            f.write("OVERALL STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total instances evaluated: {len(self.df)}\n")

            valid_comparisons = self.df.dropna(subset=['quality_ratio'])
            f.write(f"Valid comparisons: {len(valid_comparisons)}\n\n")

            # Runtime statistics
            f.write("RUNTIME PERFORMANCE\n")
            f.write("-" * 40 + "\n")

            chalupa_times = self.df['chalupa_time'].dropna()
            ilp_times = self.df['ilp_time'].dropna()

            f.write(f"Chalupa Runtime:\n")
            f.write(f"  Mean: {chalupa_times.mean():.3f}s\n")
            f.write(f"  Median: {chalupa_times.median():.3f}s\n")
            f.write(f"  Min: {chalupa_times.min():.3f}s\n")
            f.write(f"  Max: {chalupa_times.max():.3f}s\n\n")

            f.write(f"ILP Runtime:\n")
            f.write(f"  Mean: {ilp_times.mean():.3f}s\n")
            f.write(f"  Median: {ilp_times.median():.3f}s\n")
            f.write(f"  Min: {ilp_times.min():.3f}s\n")
            f.write(f"  Max: {ilp_times.max():.3f}s\n\n")

            # Calculate speedup
            speedup_data = self.df.dropna(subset=['chalupa_time', 'ilp_time'])
            if not speedup_data.empty:
                speedups = speedup_data['ilp_time'] / speedup_data['chalupa_time']
                f.write(f"Average Speedup (ILP/Chalupa): {speedups.mean():.1f}x\n")
                f.write(f"Median Speedup: {speedups.median():.1f}x\n\n")

            # Solution quality statistics
            f.write("SOLUTION QUALITY\n")
            f.write("-" * 40 + "\n")

            if not valid_comparisons.empty:
                quality_ratios = valid_comparisons['quality_ratio']
                f.write(f"Quality Ratio (Chalupa/ILP):\n")
                f.write(f"  Mean: {quality_ratios.mean():.4f}\n")
                f.write(f"  Median: {quality_ratios.median():.4f}\n")
                f.write(f"  Min: {quality_ratios.min():.4f}\n")
                f.write(f"  Max: {quality_ratios.max():.4f}\n")
                f.write(f"  Std Dev: {quality_ratios.std():.4f}\n\n")

                # Success rates
                f.write("Success Rates (Chalupa finds optimal):\n")
                optimal = (quality_ratios == 1.0).mean() * 100
                within_5 = (quality_ratios <= 1.05).mean() * 100
                within_10 = (quality_ratios <= 1.10).mean() * 100
                within_20 = (quality_ratios <= 1.20).mean() * 100

                f.write(f"  Optimal (ratio = 1.0): {optimal:.1f}%\n")
                f.write(f"  Within 5% of optimal: {within_5:.1f}%\n")
                f.write(f"  Within 10% of optimal: {within_10:.1f}%\n")
                f.write(f"  Within 20% of optimal: {within_20:.1f}%\n\n")

                # Absolute gaps
                abs_gaps = valid_comparisons['absolute_gap'].dropna()
                if not abs_gaps.empty:
                    f.write(f"Absolute Gap (Chalupa - ILP):\n")
                    f.write(f"  Mean: {abs_gaps.mean():.2f}\n")
                    f.write(f"  Median: {abs_gaps.median():.2f}\n")
                    f.write(f"  Max: {abs_gaps.max():.0f}\n\n")

            # Problem size analysis
            f.write("PROBLEM SIZE ANALYSIS\n")
            f.write("-" * 40 + "\n")

            size_bins = pd.cut(self.df['n_nodes'], bins=[0, 20, 50, 100, 200, float('inf')],
                               labels=['<20', '20-50', '50-100', '100-200', '>200'])

            for size_range in size_bins.unique():
                if pd.notna(size_range):
                    subset = valid_comparisons[size_bins == size_range]
                    if not subset.empty:
                        f.write(f"\nNodes {size_range}:\n")
                        f.write(f"  Instances: {len(subset)}\n")
                        f.write(f"  Avg Quality Ratio: {subset['quality_ratio'].mean():.3f}\n")
                        f.write(f"  Optimal Solutions: {(subset['quality_ratio'] == 1.0).mean() * 100:.1f}%\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")

        print(f"Saved summary report to {report_path}")


class ExtendedWP1cEvaluator(WP1cEvaluator):
    """Extended evaluation class for WP1.c comparisons with all solver variants"""

    def __init__(self, output_dir="evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def evaluate_single_instance(self, filepath: str, timeout: int = 300) -> Dict:
        """
        Evaluate a single graph instance with all solver variants:
        - Chalupa heuristic
        - Standard ILP
        - ILP with warmstart
        - Reduced ILP (with graph reductions)
        - Interactive reduced ILP (iterative reduction + heuristic)

        Returns dict with comprehensive metrics for all methods
        """
        result = {'filepath': filepath}

        # Load graph and get basic properties
        try:
            G = txt_to_networkx(filepath)
            result['n_nodes'] = G.number_of_nodes()
            result['n_edges'] = G.number_of_edges()
            result['density'] = nx.density(G)

            # Extract perturbation level from filename if available
            import re
            match = re.search(r'r(\d+)', Path(filepath).stem)
            if match:
                result['perturbation'] = int(match.group(1)) / 100
            else:
                result['perturbation'] = None

        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None

        # 1. Run Chalupa heuristic (provides upper bound for θ(G))
        # Chalupa is a greedy heuristic that iteratively finds cliques to cover the graph
        try:
            start = time.time()
            chalupa_result = chalupa_wrapper(filepath)
            chalupa_time = time.time() - start

            result['chalupa_theta'] = chalupa_result if chalupa_result else None
            result['chalupa_time'] = chalupa_time
        except Exception as e:
            print(f"Chalupa failed on {filepath}: {e}")
            result['chalupa_theta'] = None
            result['chalupa_time'] = None

        # 2. Run standard ILP solver (exact solution without any enhancements)
        # This solves the clique cover problem by coloring the complement graph
        try:
            start = time.time()
            ilp_res = ilp_wrapper(
                filepath,
                use_warmstart=False,  # No warmstart for fair baseline
                time_limit=timeout,
                mip_gap=0.01,
                verbose=False,
                return_assignment=False
            )
            ilp_time = time.time() - start

            if isinstance(ilp_res, dict):
                result['ilp_theta'] = ilp_res.get('theta')
                result['ilp_time'] = ilp_time
                result['ilp_status'] = ilp_res.get('status', 'unknown')
                result['ilp_gap'] = ilp_res.get('gap', None)
            else:
                result['ilp_theta'] = ilp_res
                result['ilp_time'] = ilp_time
                result['ilp_status'] = 'solved'
                result['ilp_gap'] = 0.0

        except Exception as e:
            print(f"ILP failed on {filepath}: {e}")
            result['ilp_theta'] = None
            result['ilp_time'] = None
            result['ilp_status'] = 'failed'

        # 3. Run ILP with Chalupa warmstart (uses heuristic solution to initialize ILP)
        # Warmstart provides an initial feasible solution to speed up ILP convergence
        try:
            start = time.time()
            ilp_warm_res = ilp_wrapper(
                filepath,
                use_warmstart=True,  # Enable Chalupa-based warmstart
                time_limit=timeout,
                mip_gap=0.01,
                verbose=False,
                return_assignment=False
            )
            ilp_warm_time = time.time() - start

            if isinstance(ilp_warm_res, dict):
                result['ilp_warmstart_theta'] = ilp_warm_res.get('theta')
                result['ilp_warmstart_time'] = ilp_warm_time
                result['ilp_warmstart_status'] = ilp_warm_res.get('status', 'unknown')
                result['ilp_warmstart_gap'] = ilp_warm_res.get('gap', None)
            else:
                result['ilp_warmstart_theta'] = ilp_warm_res
                result['ilp_warmstart_time'] = ilp_warm_time
                result['ilp_warmstart_status'] = 'solved'
                result['ilp_warmstart_gap'] = 0.0

        except Exception as e:
            print(f"ILP with warmstart failed on {filepath}: {e}")
            result['ilp_warmstart_theta'] = None
            result['ilp_warmstart_time'] = None
            result['ilp_warmstart_status'] = 'failed'

        # 4. Run Reduced ILP (applies graph reductions before solving)
        # Graph reductions simplify the problem by removing/merging nodes while preserving θ(G)
        try:
            start = time.time()
            reduced_res = reduced_ilp_wrapper(
                filepath,
                use_warmstart=False,
                time_limit=timeout,
                mip_gap=0.01,
                verbose=False,
                return_assignment=False
            )
            reduced_time = time.time() - start

            if isinstance(reduced_res, dict):
                result['reduced_ilp_theta'] = reduced_res.get('theta')
                result['reduced_ilp_time'] = reduced_time
                result['reduced_ilp_status'] = reduced_res.get('status', 'unknown')
                result['reduced_ilp_gap'] = reduced_res.get('gap', None)
            else:
                result['reduced_ilp_theta'] = reduced_res
                result['reduced_ilp_time'] = reduced_time
                result['reduced_ilp_status'] = 'solved'
                result['reduced_ilp_gap'] = 0.0

        except Exception as e:
            print(f"Reduced ILP failed on {filepath}: {e}")
            result['reduced_ilp_theta'] = None
            result['reduced_ilp_time'] = None
            result['reduced_ilp_status'] = 'failed'

        # 5. Run Interactive Reduced ILP (iterative reduction guided by heuristic upper bounds)
        # This method alternates between computing heuristic bounds and applying reductions
        # until no further improvement is possible, then solves the final reduced problem
        try:
            start = time.time()
            interactive_res = interactive_reduced_ilp_wrapper(
                filepath,
                use_warmstart=False,
                max_rounds=10,  # Maximum iterations of reduction-heuristic cycle
                time_limit=timeout,
                mip_gap=0.01,
                verbose=False,
                return_assignment=False
            )
            interactive_time = time.time() - start

            if isinstance(interactive_res, dict):
                result['interactive_ilp_theta'] = interactive_res.get('theta')
                result['interactive_ilp_time'] = interactive_time
                result['interactive_ilp_status'] = interactive_res.get('status', 'unknown')
                result['interactive_ilp_gap'] = interactive_res.get('gap', None)
            else:
                result['interactive_ilp_theta'] = interactive_res
                result['interactive_ilp_time'] = interactive_time
                result['interactive_ilp_status'] = 'solved'
                result['interactive_ilp_gap'] = 0.0

        except Exception as e:
            print(f"Interactive reduced ILP failed on {filepath}: {e}")
            result['interactive_ilp_theta'] = None
            result['interactive_ilp_time'] = None
            result['interactive_ilp_status'] = 'failed'

        # Calculate quality metrics (using standard ILP as baseline for exact solution)
        if result['ilp_theta']:
            baseline = result['ilp_theta']

            # Quality ratios for all methods
            for method in ['chalupa', 'ilp_warmstart', 'reduced_ilp', 'interactive_ilp']:
                theta_key = f'{method}_theta'
                if result.get(theta_key):
                    result[f'{method}_quality_ratio'] = result[theta_key] / baseline
                    result[f'{method}_absolute_gap'] = result[theta_key] - baseline
                else:
                    result[f'{method}_quality_ratio'] = None
                    result[f'{method}_absolute_gap'] = None

        return result

    def analyze_warmstart_effectiveness(self):
        """Detailed analysis of _chalupa_warmstart effectiveness"""
        if self.df.empty:
            print("No data available for warmstart analysis")
            return

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        # 1. Warmstart impact on different problem sizes
        ax = axes[0, 0]
        if 'n_nodes' in self.df.columns:
            size_bins = pd.cut(self.df['n_nodes'], bins=5)
            warmstart_improvements = []
            bin_centers = []

            for bin_val in size_bins.cat.categories:
                mask = size_bins == bin_val
                if 'ilp_time' in self.df.columns and 'ilp_warmstart_time' in self.df.columns:
                    valid = mask & self.df[['ilp_time', 'ilp_warmstart_time']].notna().all(axis=1)
                    if valid.any():
                        improvement = ((self.df.loc[valid, 'ilp_time'] -
                                        self.df.loc[valid, 'ilp_warmstart_time']) /
                                       self.df.loc[valid, 'ilp_time'] * 100).mean()
                        warmstart_improvements.append(improvement)
                        bin_centers.append(bin_val.mid)

            if warmstart_improvements:
                bars = ax.bar(range(len(warmstart_improvements)), warmstart_improvements,
                              color='green', alpha=0.7, edgecolor='black')
                ax.set_xticks(range(len(warmstart_improvements)))
                ax.set_xticklabels([f'{int(c)}' for c in bin_centers], rotation=45)
                ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
                ax.set_xlabel('Number of Nodes')
                ax.set_ylabel('Runtime Improvement (%)')
                ax.set_title('Warmstart Effectiveness by Problem Size')

                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height,
                            f'{height:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
        ax.grid(True, alpha=0.3)

        # 2. Warmstart quality analysis
        ax = axes[0, 1]
        # Compare how close Chalupa initial solution is to final ILP solution
        if 'chalupa_theta' in self.df.columns and 'ilp_theta' in self.df.columns:
            valid = self.df[['chalupa_theta', 'ilp_theta']].notna().all(axis=1)
            if valid.any():
                # Initial solution quality (Chalupa)
                initial_quality = self.df.loc[valid, 'chalupa_theta']
                optimal = self.df.loc[valid, 'ilp_theta']
                gap = ((initial_quality - optimal) / optimal * 100)

                ax.hist(gap, bins=20, color='orange', alpha=0.7, edgecolor='black')
                ax.axvline(x=0, color='green', linestyle='--', linewidth=2,
                           label='Perfect initial solution')
                ax.set_xlabel('Initial Solution Gap (%)')
                ax.set_ylabel('Frequency')
                ax.set_title('Quality of Chalupa Warmstart Solution')
                mean_gap = gap.mean()
                ax.axvline(x=mean_gap, color='red', linestyle=':', linewidth=2,
                           label=f'Mean gap: {mean_gap:.1f}%')
                ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Correlation between initial quality and speedup
        ax = axes[0, 2]
        if all(col in self.df.columns for col in ['chalupa_theta', 'ilp_theta',
                                                  'ilp_time', 'ilp_warmstart_time']):
            valid = self.df[['chalupa_theta', 'ilp_theta',
                             'ilp_time', 'ilp_warmstart_time']].notna().all(axis=1)
            if valid.any():
                initial_gap = ((self.df.loc[valid, 'chalupa_theta'] -
                                self.df.loc[valid, 'ilp_theta']) /
                               self.df.loc[valid, 'ilp_theta'] * 100)
                speedup = self.df.loc[valid, 'ilp_time'] / self.df.loc[valid, 'ilp_warmstart_time']

                scatter = ax.scatter(initial_gap, speedup, alpha=0.6, s=50,
                                     c=self.df.loc[valid, 'n_nodes'], cmap='viridis')
                ax.set_xlabel('Initial Solution Gap (%)')
                ax.set_ylabel('Speedup Factor')
                ax.set_title('Initial Quality vs Speedup')
                ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='No speedup')
                ax.axvline(x=0, color='green', linestyle='--', alpha=0.5, label='Perfect initial')
                plt.colorbar(scatter, ax=ax, label='Number of Nodes')
                ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Warmstart vs problem density
        ax = axes[1, 0]
        if 'density' in self.df.columns:
            density_bins = pd.cut(self.df['density'], bins=5)
            density_speedups = []
            density_centers = []

            for bin_val in density_bins.cat.categories:
                mask = density_bins == bin_val
                if 'ilp_time' in self.df.columns and 'ilp_warmstart_time' in self.df.columns:
                    valid = mask & self.df[['ilp_time', 'ilp_warmstart_time']].notna().all(axis=1)
                    if valid.any():
                        speedup = (self.df.loc[valid, 'ilp_time'] /
                                   self.df.loc[valid, 'ilp_warmstart_time']).mean()
                        density_speedups.append(speedup)
                        density_centers.append(bin_val.mid)

            if density_speedups:
                ax.plot(density_centers, density_speedups, marker='o',
                        linewidth=2, markersize=10, color='purple')
                ax.fill_between(density_centers, 1, density_speedups,
                                alpha=0.3, color='purple')
                ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='No speedup')
                ax.set_xlabel('Graph Density')
                ax.set_ylabel('Average Speedup Factor')
                ax.set_title('Warmstart Effectiveness vs Graph Density')
                ax.legend()
        ax.grid(True, alpha=0.3)

        # 5. Success rate comparison
        ax = axes[1, 1]
        success_data = {}

        if 'ilp_status' in self.df.columns:
            success_data['Standard ILP'] = (self.df['ilp_status'] == 'optimal').mean() * 100
        if 'ilp_warmstart_status' in self.df.columns:
            success_data['ILP + Warmstart'] = (self.df['ilp_warmstart_status'] == 'optimal').mean() * 100

        if success_data:
            bars = ax.bar(range(len(success_data)), list(success_data.values()),
                          color=['red', 'green'], alpha=0.7, edgecolor='black')
            ax.set_xticks(range(len(success_data)))
            ax.set_xticklabels(list(success_data.keys()))
            ax.set_ylabel('Success Rate (%)')
            ax.set_title('Optimal Solution Success Rate')
            ax.set_ylim(0, 105)

            # Add value labels
            for bar, value in zip(bars, success_data.values()):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                        f'{value:.1f}%', ha='center', va='bottom')
        ax.grid(True, alpha=0.3)

        # 6. Detailed statistics table
        ax = axes[1, 2]
        ax.axis('off')

        # Calculate warmstart statistics
        stats_text = "_chalupa_warmstart Analysis\n" + "=" * 35 + "\n\n"

        if 'ilp_time' in self.df.columns and 'ilp_warmstart_time' in self.df.columns:
            valid = self.df[['ilp_time', 'ilp_warmstart_time']].notna().all(axis=1)
            if valid.any():
                avg_speedup = (self.df.loc[valid, 'ilp_time'] /
                               self.df.loc[valid, 'ilp_warmstart_time']).mean()
                median_speedup = (self.df.loc[valid, 'ilp_time'] /
                                  self.df.loc[valid, 'ilp_warmstart_time']).median()
                improvement_cases = (self.df.loc[valid, 'ilp_warmstart_time'] <
                                     self.df.loc[valid, 'ilp_time']).mean() * 100

                stats_text += "Performance Metrics:\n"
                stats_text += f"• Average speedup: {avg_speedup:.2f}x\n"
                stats_text += f"• Median speedup: {median_speedup:.2f}x\n"
                stats_text += f"• Improvement rate: {improvement_cases:.1f}%\n\n"

        if 'chalupa_theta' in self.df.columns and 'ilp_theta' in self.df.columns:
            valid = self.df[['chalupa_theta', 'ilp_theta']].notna().all(axis=1)
            if valid.any():
                perfect_initial = (self.df.loc[valid, 'chalupa_theta'] ==
                                   self.df.loc[valid, 'ilp_theta']).mean() * 100
                avg_gap = ((self.df.loc[valid, 'chalupa_theta'] -
                            self.df.loc[valid, 'ilp_theta']) /
                           self.df.loc[valid, 'ilp_theta']).mean() * 100

                stats_text += "Initial Solution Quality:\n"
                stats_text += f"• Perfect initial: {perfect_initial:.1f}%\n"
                stats_text += f"• Average gap: {avg_gap:.1f}%\n\n"

        stats_text += "How _chalupa_warmstart works:\n"
        stats_text += "1. Runs Chalupa on complement graph\n"
        stats_text += "2. Extracts color assignment\n"
        stats_text += "3. Provides to ILP as initial solution\n"
        stats_text += "4. ILP refines to optimal (if possible)"

        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle('_chalupa_warmstart Effectiveness Analysis', fontsize=16, y=1.02)
        plt.tight_layout()

        output_path = self.output_dir / f"warmstart_analysis_{self.timestamp}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved warmstart analysis to {output_path}")
        plt.show()

    def create_reduction_effectiveness_plot(self):
        """Analyze the effectiveness of graph reductions"""
        if self.df.empty:
            print("No data available for plotting")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Compare reduced vs interactive reduced
        ax = axes[0, 0]
        if 'reduced_ilp_time' in self.df.columns and 'interactive_ilp_time' in self.df.columns:
            valid_idx = self.df[['reduced_ilp_time', 'interactive_ilp_time']].notna().all(axis=1)
            if valid_idx.any():
                ax.scatter(self.df.loc[valid_idx, 'reduced_ilp_time'],
                           self.df.loc[valid_idx, 'interactive_ilp_time'],
                           alpha=0.6, s=50, c=self.df.loc[valid_idx, 'n_nodes'],
                           cmap='viridis')
                max_time = max(self.df.loc[valid_idx, 'reduced_ilp_time'].max(),
                               self.df.loc[valid_idx, 'interactive_ilp_time'].max())
                ax.plot([0, max_time], [0, max_time], 'r--', alpha=0.5,
                        label='Equal runtime')
                ax.fill_between([0, max_time], [0, max_time], 0,
                                alpha=0.1, color='green',
                                label='Interactive faster')
                plt.colorbar(ax.collections[0], ax=ax, label='Number of Nodes')
        ax.set_xlabel('Reduced ILP Time (s)')
        ax.set_ylabel('Interactive Reduced ILP Time (s)')
        ax.set_title('Single vs Interactive Reduction')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Reduction effectiveness by graph density
        ax = axes[0, 1]
        if 'density' in self.df.columns:
            density_bins = pd.cut(self.df['density'], bins=5)
            reduction_speedup = []
            bin_centers = []

            for bin_val in density_bins.cat.categories:
                mask = density_bins == bin_val
                if 'ilp_time' in self.df.columns and 'reduced_ilp_time' in self.df.columns:
                    valid = mask & self.df[['ilp_time', 'reduced_ilp_time']].notna().all(axis=1)
                    if valid.any():
                        speedup = (self.df.loc[valid, 'ilp_time'] /
                                   self.df.loc[valid, 'reduced_ilp_time']).mean()
                        reduction_speedup.append(speedup)
                        bin_centers.append(bin_val.mid)

            if reduction_speedup:
                ax.bar(range(len(reduction_speedup)), reduction_speedup,
                       color='orange', alpha=0.7, edgecolor='black')
                ax.set_xticks(range(len(reduction_speedup)))
                ax.set_xticklabels([f'{c:.2f}' for c in bin_centers], rotation=45)
                ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5,
                           label='No speedup')
                ax.set_xlabel('Graph Density')
                ax.set_ylabel('Average Speedup Factor')
                ax.set_title('Reduction Effectiveness by Density')
                ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Cumulative runtime distribution
        ax = axes[1, 0]
        methods_to_plot = ['ilp', 'ilp_warmstart', 'reduced_ilp', 'interactive_ilp']
        colors = ['red', 'green', 'orange', 'purple']

        for method, color in zip(methods_to_plot, colors):
            time_col = f'{method}_time'
            if time_col in self.df.columns:
                times = self.df[time_col].dropna().sort_values()
                if not times.empty:
                    cumulative = np.arange(1, len(times) + 1) / len(times) * 100
                    label = method.replace('_', ' ').title()
                    ax.plot(times, cumulative, label=label, color=color, linewidth=2)

        ax.set_xlabel('Runtime (seconds)')
        ax.set_ylabel('Cumulative Percentage (%)')
        ax.set_title('Cumulative Runtime Distribution')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Performance gain summary
        ax = axes[1, 1]
        ax.axis('off')

        # Calculate performance metrics
        summary_text = "Performance Gains Summary\n" + "=" * 30 + "\n\n"

        # Warmstart improvement
        if 'ilp_time' in self.df.columns and 'ilp_warmstart_time' in self.df.columns:
            valid = self.df[['ilp_time', 'ilp_warmstart_time']].notna().all(axis=1)
            if valid.any():
                warmstart_speedup = (self.df.loc[valid, 'ilp_time'] /
                                     self.df.loc[valid, 'ilp_warmstart_time']).mean()
                summary_text += f"Warmstart Speedup: {warmstart_speedup:.2f}x\n"

        # Reduction improvement
        if 'ilp_time' in self.df.columns and 'reduced_ilp_time' in self.df.columns:
            valid = self.df[['ilp_time', 'reduced_ilp_time']].notna().all(axis=1)
            if valid.any():
                reduction_speedup = (self.df.loc[valid, 'ilp_time'] /
                                     self.df.loc[valid, 'reduced_ilp_time']).mean()
                summary_text += f"Reduction Speedup: {reduction_speedup:.2f}x\n"

        # Interactive reduction improvement
        if 'ilp_time' in self.df.columns and 'interactive_ilp_time' in self.df.columns:
            valid = self.df[['ilp_time', 'interactive_ilp_time']].notna().all(axis=1)
            if valid.any():
                interactive_speedup = (self.df.loc[valid, 'ilp_time'] /
                                       self.df.loc[valid, 'interactive_ilp_time']).mean()
                summary_text += f"Interactive Speedup: {interactive_speedup:.2f}x\n"

        # Chalupa speedup
        if 'ilp_time' in self.df.columns and 'chalupa_time' in self.df.columns:
            valid = self.df[['ilp_time', 'chalupa_time']].notna().all(axis=1)
            if valid.any():
                chalupa_speedup = (self.df.loc[valid, 'ilp_time'] /
                                   self.df.loc[valid, 'chalupa_time']).mean()
                summary_text += f"\nChalupa vs ILP: {chalupa_speedup:.1f}x faster\n"

        # Quality comparison
        summary_text += "\n" + "-" * 30 + "\nSolution Quality\n" + "-" * 30 + "\n"

        for method in ['chalupa', 'ilp_warmstart', 'reduced_ilp', 'interactive_ilp']:
            quality_col = f'{method}_quality_ratio'
            if quality_col in self.df.columns:
                avg_quality = self.df[quality_col].dropna().mean()
                method_name = method.replace('_', ' ').title()
                summary_text += f"{method_name}: {avg_quality:.3f}\n"

        ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='center',
                fontfamily='monospace')

        plt.suptitle('Reduction Effectiveness Analysis', fontsize=16, y=1.02)
        plt.tight_layout()

        output_path = self.output_dir / f"reduction_effectiveness_{self.timestamp}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved reduction effectiveness plots to {output_path}")
        plt.show()

    def generate_extended_summary_report(self):
        """Generate a comprehensive summary report including all solver variants"""
        if not hasattr(self, 'df') or self.df.empty:
            print("No data available for summary")
            return

        report_path = self.output_dir / f"extended_summary_report_{self.timestamp}.txt"

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("EXTENDED WP1.c EVALUATION SUMMARY REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            # Overall statistics
            f.write("OVERALL STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total instances evaluated: {len(self.df)}\n\n")

            # Runtime statistics for all methods
            f.write("RUNTIME PERFORMANCE (ALL METHODS)\n")
            f.write("-" * 40 + "\n")

            methods = ['chalupa', 'ilp', 'ilp_warmstart', 'reduced_ilp', 'interactive_ilp']
            method_names = {
                'chalupa': 'Chalupa Heuristic',
                'ilp': 'Standard ILP',
                'ilp_warmstart': 'ILP with Warmstart',
                'reduced_ilp': 'Reduced ILP',
                'interactive_ilp': 'Interactive Reduced ILP'
            }

            for method in methods:
                time_col = f'{method}_time'
                if time_col in self.df.columns:
                    times = self.df[time_col].dropna()
                    if not times.empty:
                        f.write(f"\n{method_names[method]}:\n")
                        f.write(f"  Mean: {times.mean():.3f}s\n")
                        f.write(f"  Median: {times.median():.3f}s\n")
                        f.write(f"  Min: {times.min():.3f}s\n")
                        f.write(f"  Max: {times.max():.3f}s\n")
                        f.write(f"  Std Dev: {times.std():.3f}s\n")

            # Speedup analysis
            f.write("\n\nSPEEDUP ANALYSIS (vs Standard ILP)\n")
            f.write("-" * 40 + "\n")

            for method in ['chalupa', 'ilp_warmstart', 'reduced_ilp', 'interactive_ilp']:
                time_col = f'{method}_time'
                if time_col in self.df.columns and 'ilp_time' in self.df.columns:
                    valid = self.df[[time_col, 'ilp_time']].notna().all(axis=1)
                    if valid.any():
                        speedups = self.df.loc[valid, 'ilp_time'] / self.df.loc[valid, time_col]
                        f.write(f"\n{method_names[method]}:\n")
                        f.write(f"  Mean Speedup: {speedups.mean():.2f}x\n")
                        f.write(f"  Median Speedup: {speedups.median():.2f}x\n")
                        f.write(f"  Max Speedup: {speedups.max():.2f}x\n")

            # Solution quality statistics
            f.write("\n\nSOLUTION QUALITY (vs Standard ILP)\n")
            f.write("-" * 40 + "\n")

            for method in ['chalupa', 'ilp_warmstart', 'reduced_ilp', 'interactive_ilp']:
                quality_col = f'{method}_quality_ratio'
                if quality_col in self.df.columns:
                    quality = self.df[quality_col].dropna()
                    if not quality.empty:
                        f.write(f"\n{method_names[method]}:\n")
                        f.write(f"  Mean Ratio: {quality.mean():.4f}\n")
                        f.write(f"  Median Ratio: {quality.median():.4f}\n")
                        f.write(f"  Min Ratio: {quality.min():.4f}\n")
                        f.write(f"  Max Ratio: {quality.max():.4f}\n")
                        f.write(f"  Std Dev: {quality.std():.4f}\n")

                        # Success rates
                        optimal = (quality == 1.0).mean() * 100
                        within_5 = (quality <= 1.05).mean() * 100
                        within_10 = (quality <= 1.10).mean() * 100

                        f.write(f"  Optimal solutions: {optimal:.1f}%\n")
                        f.write(f"  Within 5% of optimal: {within_5:.1f}%\n")
                        f.write(f"  Within 10% of optimal: {within_10:.1f}%\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF EXTENDED REPORT\n")
            f.write("=" * 80 + "\n")

    def create_method_comparison_plots(self):
        """compares runtimes of all variantes in 1 plot"""
        if not hasattr(self, 'df') or self.df.empty:
            print("No data available for plotting")
            return

        methods = ['chalupa', 'ilp', 'ilp_warmstart', 'reduced_ilp', 'interactive_ilp']
        fig, ax = plt.subplots(figsize=(10, 6))
        data = []
        labels = []
        for m in methods:
            col = f"{m}_time"
            if col in self.df.columns:
                vals = self.df[col].dropna()
                if not vals.empty:
                    data.append(vals.values)
                    labels.append(m.replace('_', ' '))
        if not data:
            print("No method runtimes to plot.")
            return

        ax.boxplot(data, showfliers=False)
        ax.set_xticklabels(labels, rotation=15)
        ax.set_yscale('log')
        ax.set_ylabel('Runtime (s)')
        ax.set_title('Method Runtime Comparison (log scale)')
        ax.grid(True, alpha=0.3)

        out = self.output_dir / f"method_comparison_{self.timestamp}.png"
        plt.savefig(out, dpi=300, bbox_inches='tight')
        print(f"Saved method comparison plots to {out}")
        plt.show()

    def analyze_warmstart_effectiveness(self):
        """Detailed analysis of _chalupa_warmstart effectiveness"""
        if self.df.empty:
            print("No data available for warmstart analysis")
            return

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        # 1. Warmstart impact on different problem sizes
        ax = axes[0, 0]
        if 'n_nodes' in self.df.columns:
            size_bins = pd.cut(self.df['n_nodes'], bins=5)
            warmstart_improvements = []
            bin_centers = []

            for bin_val in size_bins.cat.categories:
                mask = size_bins == bin_val
                if 'ilp_time' in self.df.columns and 'ilp_warmstart_time' in self.df.columns:
                    valid = mask & self.df[['ilp_time', 'ilp_warmstart_time']].notna().all(axis=1)
                    if valid.any():
                        improvement = ((self.df.loc[valid, 'ilp_time'] -
                                        self.df.loc[valid, 'ilp_warmstart_time']) /
                                       self.df.loc[valid, 'ilp_time'] * 100).mean()
                        warmstart_improvements.append(improvement)
                        bin_centers.append(bin_val.mid)

            if warmstart_improvements:
                bars = ax.bar(range(len(warmstart_improvements)), warmstart_improvements,
                              color='green', alpha=0.7, edgecolor='black')
                ax.set_xticks(range(len(warmstart_improvements)))
                ax.set_xticklabels([f'{int(c)}' for c in bin_centers], rotation=45)
                ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
                ax.set_xlabel('Number of Nodes')
                ax.set_ylabel('Runtime Improvement (%)')
                ax.set_title('Warmstart Effectiveness by Problem Size')

                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height,
                            f'{height:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
        ax.grid(True, alpha=0.3)

        # 2. Warmstart quality analysis
        ax = axes[0, 1]
        # Compare how close Chalupa initial solution is to final ILP solution
        if 'chalupa_theta' in self.df.columns and 'ilp_theta' in self.df.columns:
            valid = self.df[['chalupa_theta', 'ilp_theta']].notna().all(axis=1)
            if valid.any():
                # Initial solution quality (Chalupa)
                initial_quality = self.df.loc[valid, 'chalupa_theta']
                optimal = self.df.loc[valid, 'ilp_theta']
                gap = ((initial_quality - optimal) / optimal * 100)

                ax.hist(gap, bins=20, color='orange', alpha=0.7, edgecolor='black')
                ax.axvline(x=0, color='green', linestyle='--', linewidth=2,
                           label='Perfect initial solution')
                ax.set_xlabel('Initial Solution Gap (%)')
                ax.set_ylabel('Frequency')
                ax.set_title('Quality of Chalupa Warmstart Solution')
                mean_gap = gap.mean()
                ax.axvline(x=mean_gap, color='red', linestyle=':', linewidth=2,
                           label=f'Mean gap: {mean_gap:.1f}%')
                ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Correlation between initial quality and speedup
        ax = axes[0, 2]
        if all(col in self.df.columns for col in ['chalupa_theta', 'ilp_theta',
                                                  'ilp_time', 'ilp_warmstart_time']):
            valid = self.df[['chalupa_theta', 'ilp_theta',
                             'ilp_time', 'ilp_warmstart_time']].notna().all(axis=1)
            if valid.any():
                initial_gap = ((self.df.loc[valid, 'chalupa_theta'] -
                                self.df.loc[valid, 'ilp_theta']) /
                               self.df.loc[valid, 'ilp_theta'] * 100)
                speedup = self.df.loc[valid, 'ilp_time'] / self.df.loc[valid, 'ilp_warmstart_time']

                scatter = ax.scatter(initial_gap, speedup, alpha=0.6, s=50,
                                     c=self.df.loc[valid, 'n_nodes'], cmap='viridis')
                ax.set_xlabel('Initial Solution Gap (%)')
                ax.set_ylabel('Speedup Factor')
                ax.set_title('Initial Quality vs Speedup')
                ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='No speedup')
                ax.axvline(x=0, color='green', linestyle='--', alpha=0.5, label='Perfect initial')
                plt.colorbar(scatter, ax=ax, label='Number of Nodes')
                ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Warmstart vs problem density
        ax = axes[1, 0]
        if 'density' in self.df.columns:
            density_bins = pd.cut(self.df['density'], bins=5)
            density_speedups = []
            density_centers = []

            for bin_val in density_bins.cat.categories:
                mask = density_bins == bin_val
                if 'ilp_time' in self.df.columns and 'ilp_warmstart_time' in self.df.columns:
                    valid = mask & self.df[['ilp_time', 'ilp_warmstart_time']].notna().all(axis=1)
                    if valid.any():
                        speedup = (self.df.loc[valid, 'ilp_time'] /
                                   self.df.loc[valid, 'ilp_warmstart_time']).mean()
                        density_speedups.append(speedup)
                        density_centers.append(bin_val.mid)

            if density_speedups:
                ax.plot(density_centers, density_speedups, marker='o',
                        linewidth=2, markersize=10, color='purple')
                ax.fill_between(density_centers, 1, density_speedups,
                                alpha=0.3, color='purple')
                ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='No speedup')
                ax.set_xlabel('Graph Density')
                ax.set_ylabel('Average Speedup Factor')
                ax.set_title('Warmstart Effectiveness vs Graph Density')
                ax.legend()
        ax.grid(True, alpha=0.3)

        # 5. Success rate comparison
        ax = axes[1, 1]
        success_data = {}

        if 'ilp_status' in self.df.columns:
            success_data['Standard ILP'] = (self.df['ilp_status'] == 'optimal').mean() * 100
        if 'ilp_warmstart_status' in self.df.columns:
            success_data['ILP + Warmstart'] = (self.df['ilp_warmstart_status'] == 'optimal').mean() * 100

        if success_data:
            bars = ax.bar(range(len(success_data)), list(success_data.values()),
                          color=['red', 'green'], alpha=0.7, edgecolor='black')
            ax.set_xticks(range(len(success_data)))
            ax.set_xticklabels(list(success_data.keys()))
            ax.set_ylabel('Success Rate (%)')
            ax.set_title('Optimal Solution Success Rate')
            ax.set_ylim(0, 105)

            # Add value labels
            for bar, value in zip(bars, success_data.values()):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                        f'{value:.1f}%', ha='center', va='bottom')
        ax.grid(True, alpha=0.3)

        # 6. Detailed statistics table
        ax = axes[1, 2]
        ax.axis('off')

        # Calculate warmstart statistics
        stats_text = "_chalupa_warmstart Analysis\n" + "=" * 35 + "\n\n"

        if 'ilp_time' in self.df.columns and 'ilp_warmstart_time' in self.df.columns:
            valid = self.df[['ilp_time', 'ilp_warmstart_time']].notna().all(axis=1)
            if valid.any():
                avg_speedup = (self.df.loc[valid, 'ilp_time'] /
                               self.df.loc[valid, 'ilp_warmstart_time']).mean()
                median_speedup = (self.df.loc[valid, 'ilp_time'] /
                                  self.df.loc[valid, 'ilp_warmstart_time']).median()
                improvement_cases = (self.df.loc[valid, 'ilp_warmstart_time'] <
                                     self.df.loc[valid, 'ilp_time']).mean() * 100

                stats_text += "Performance Metrics:\n"
                stats_text += f"• Average speedup: {avg_speedup:.2f}x\n"
                stats_text += f"• Median speedup: {median_speedup:.2f}x\n"
                stats_text += f"• Improvement rate: {improvement_cases:.1f}%\n\n"

        if 'chalupa_theta' in self.df.columns and 'ilp_theta' in self.df.columns:
            valid = self.df[['chalupa_theta', 'ilp_theta']].notna().all(axis=1)
            if valid.any():
                perfect_initial = (self.df.loc[valid, 'chalupa_theta'] ==
                                   self.df.loc[valid, 'ilp_theta']).mean() * 100
                avg_gap = ((self.df.loc[valid, 'chalupa_theta'] -
                            self.df.loc[valid, 'ilp_theta']) /
                           self.df.loc[valid, 'ilp_theta']).mean() * 100

                stats_text += "Initial Solution Quality:\n"
                stats_text += f"• Perfect initial: {perfect_initial:.1f}%\n"
                stats_text += f"• Average gap: {avg_gap:.1f}%\n\n"

        stats_text += "How _chalupa_warmstart works:\n"
        stats_text += "1. Runs Chalupa on complement graph\n"
        stats_text += "2. Extracts color assignment\n"
        stats_text += "3. Provides to ILP as initial solution\n"
        stats_text += "4. ILP refines to optimal (if possible)"

        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle('_chalupa_warmstart Effectiveness Analysis', fontsize=16, y=1.02)
        plt.tight_layout()

        output_path = self.output_dir / f"warmstart_analysis_{self.timestamp}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved warmstart analysis to {output_path}")
        plt.show()

    def create_reduction_effectiveness_plot(self):
        """Analyze the effectiveness of graph reductions"""
        if self.df.empty:
            print("No data available for plotting")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Compare reduced vs interactive reduced
        ax = axes[0, 0]
        if 'reduced_ilp_time' in self.df.columns and 'interactive_ilp_time' in self.df.columns:
            valid_idx = self.df[['reduced_ilp_time', 'interactive_ilp_time']].notna().all(axis=1)
            if valid_idx.any():
                ax.scatter(self.df.loc[valid_idx, 'reduced_ilp_time'],
                           self.df.loc[valid_idx, 'interactive_ilp_time'],
                           alpha=0.6, s=50, c=self.df.loc[valid_idx, 'n_nodes'],
                           cmap='viridis')
                max_time = max(self.df.loc[valid_idx, 'reduced_ilp_time'].max(),
                               self.df.loc[valid_idx, 'interactive_ilp_time'].max())
                ax.plot([0, max_time], [0, max_time], 'r--', alpha=0.5,
                        label='Equal runtime')
                ax.fill_between([0, max_time], [0, max_time], 0,
                                alpha=0.1, color='green',
                                label='Interactive faster')
                plt.colorbar(ax.collections[0], ax=ax, label='Number of Nodes')
        ax.set_xlabel('Reduced ILP Time (s)')
        ax.set_ylabel('Interactive Reduced ILP Time (s)')
        ax.set_title('Single vs Interactive Reduction')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Reduction effectiveness by graph density
        ax = axes[0, 1]
        if 'density' in self.df.columns:
            density_bins = pd.cut(self.df['density'], bins=5)
            reduction_speedup = []
            bin_centers = []

            for bin_val in density_bins.cat.categories:
                mask = density_bins == bin_val
                if 'ilp_time' in self.df.columns and 'reduced_ilp_time' in self.df.columns:
                    valid = mask & self.df[['ilp_time', 'reduced_ilp_time']].notna().all(axis=1)
                    if valid.any():
                        speedup = (self.df.loc[valid, 'ilp_time'] /
                                   self.df.loc[valid, 'reduced_ilp_time']).mean()
                        reduction_speedup.append(speedup)
                        bin_centers.append(bin_val.mid)

            if reduction_speedup:
                ax.bar(range(len(reduction_speedup)), reduction_speedup,
                       color='orange', alpha=0.7, edgecolor='black')
                ax.set_xticks(range(len(reduction_speedup)))
                ax.set_xticklabels([f'{c:.2f}' for c in bin_centers], rotation=45)
                ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5,
                           label='No speedup')
                ax.set_xlabel('Graph Density')
                ax.set_ylabel('Average Speedup Factor')
                ax.set_title('Reduction Effectiveness by Density')
                ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Cumulative runtime distribution
        ax = axes[1, 0]
        methods_to_plot = ['ilp', 'ilp_warmstart', 'reduced_ilp', 'interactive_ilp']
        colors = ['red', 'green', 'orange', 'purple']

        for method, color in zip(methods_to_plot, colors):
            time_col = f'{method}_time'
            if time_col in self.df.columns:
                times = self.df[time_col].dropna().sort_values()
                if not times.empty:
                    cumulative = np.arange(1, len(times) + 1) / len(times) * 100
                    label = method.replace('_', ' ').title()
                    ax.plot(times, cumulative, label=label, color=color, linewidth=2)

        ax.set_xlabel('Runtime (seconds)')
        ax.set_ylabel('Cumulative Percentage (%)')
        ax.set_title('Cumulative Runtime Distribution')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Performance gain summary
        ax = axes[1, 1]
        ax.axis('off')

        # Calculate performance metrics
        summary_text = "Performance Gains Summary\n" + "=" * 30 + "\n\n"

        # Warmstart improvement
        if 'ilp_time' in self.df.columns and 'ilp_warmstart_time' in self.df.columns:
            valid = self.df[['ilp_time', 'ilp_warmstart_time']].notna().all(axis=1)
            if valid.any():
                warmstart_speedup = (self.df.loc[valid, 'ilp_time'] /
                                     self.df.loc[valid, 'ilp_warmstart_time']).mean()
                summary_text += f"Warmstart Speedup: {warmstart_speedup:.2f}x\n"

        # Reduction improvement
        if 'ilp_time' in self.df.columns and 'reduced_ilp_time' in self.df.columns:
            valid = self.df[['ilp_time', 'reduced_ilp_time']].notna().all(axis=1)
            if valid.any():
                reduction_speedup = (self.df.loc[valid, 'ilp_time'] /
                                     self.df.loc[valid, 'reduced_ilp_time']).mean()
                summary_text += f"Reduction Speedup: {reduction_speedup:.2f}x\n"

        # Interactive reduction improvement
        if 'ilp_time' in self.df.columns and 'interactive_ilp_time' in self.df.columns:
            valid = self.df[['ilp_time', 'interactive_ilp_time']].notna().all(axis=1)
            if valid.any():
                interactive_speedup = (self.df.loc[valid, 'ilp_time'] /
                                       self.df.loc[valid, 'interactive_ilp_time']).mean()
                summary_text += f"Interactive Speedup: {interactive_speedup:.2f}x\n"

        # Chalupa speedup
        if 'ilp_time' in self.df.columns and 'chalupa_time' in self.df.columns:
            valid = self.df[['ilp_time', 'chalupa_time']].notna().all(axis=1)
            if valid.any():
                chalupa_speedup = (self.df.loc[valid, 'ilp_time'] /
                                   self.df.loc[valid, 'chalupa_time']).mean()
                summary_text += f"\nChalupa vs ILP: {chalupa_speedup:.1f}x faster\n"

        # Quality comparison
        summary_text += "\n" + "-" * 30 + "\nSolution Quality\n" + "-" * 30 + "\n"

        for method in ['chalupa', 'ilp_warmstart', 'reduced_ilp', 'interactive_ilp']:
            quality_col = f'{method}_quality_ratio'
            if quality_col in self.df.columns:
                avg_quality = self.df[quality_col].dropna().mean()
                method_name = method.replace('_', ' ').title()
                summary_text += f"{method_name}: {avg_quality:.3f}\n"

        ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='center',
                fontfamily='monospace')

        plt.suptitle('Reduction Effectiveness Analysis', fontsize=16, y=1.02)
        plt.tight_layout()

        output_path = self.output_dir / f"reduction_effectiveness_{self.timestamp}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved reduction effectiveness plots to {output_path}")
        plt.show()

    def generate_extended_summary_report(self):
        """Generate a comprehensive summary report including all solver variants"""
        if not hasattr(self, 'df') or self.df.empty:
            print("No data available for summary")
            return

        report_path = self.output_dir / f"extended_summary_report_{self.timestamp}.txt"

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("EXTENDED WP1.c EVALUATION SUMMARY REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            # Overall statistics
            f.write("OVERALL STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total instances evaluated: {len(self.df)}\n\n")

            # Runtime statistics for all methods
            f.write("RUNTIME PERFORMANCE (ALL METHODS)\n")
            f.write("-" * 40 + "\n")

            methods = ['chalupa', 'ilp', 'ilp_warmstart', 'reduced_ilp', 'interactive_ilp']
            method_names = {
                'chalupa': 'Chalupa Heuristic',
                'ilp': 'Standard ILP',
                'ilp_warmstart': 'ILP with Warmstart',
                'reduced_ilp': 'Reduced ILP',
                'interactive_ilp': 'Interactive Reduced ILP'
            }

            for method in methods:
                time_col = f'{method}_time'
                if time_col in self.df.columns:
                    times = self.df[time_col].dropna()
                    if not times.empty:
                        f.write(f"\n{method_names[method]}:\n")
                        f.write(f"  Mean: {times.mean():.3f}s\n")
                        f.write(f"  Median: {times.median():.3f}s\n")
                        f.write(f"  Min: {times.min():.3f}s\n")
                        f.write(f"  Max: {times.max():.3f}s\n")
                        f.write(f"  Std Dev: {times.std():.3f}s\n")

            # Speedup analysis
            f.write("\n\nSPEEDUP ANALYSIS (vs Standard ILP)\n")
            f.write("-" * 40 + "\n")

            for method in ['chalupa', 'ilp_warmstart', 'reduced_ilp', 'interactive_ilp']:
                time_col = f'{method}_time'
                if time_col in self.df.columns and 'ilp_time' in self.df.columns:
                    valid = self.df[[time_col, 'ilp_time']].notna().all(axis=1)
                    if valid.any():
                        speedups = self.df.loc[valid, 'ilp_time'] / self.df.loc[valid, time_col]
                        f.write(f"\n{method_names[method]}:\n")
                        f.write(f"  Mean Speedup: {speedups.mean():.2f}x\n")
                        f.write(f"  Median Speedup: {speedups.median():.2f}x\n")
                        f.write(f"  Max Speedup: {speedups.max():.2f}x\n")

            # Solution quality statistics
            f.write("\n\nSOLUTION QUALITY (vs Standard ILP)\n")
            f.write("-" * 40 + "\n")

            for method in ['chalupa', 'ilp_warmstart', 'reduced_ilp', 'interactive_ilp']:
                quality_col = f'{method}_quality_ratio'
                if quality_col in self.df.columns:
                    quality = self.df[quality_col].dropna()
                    if not quality.empty:
                        f.write(f"\n{method_names[method]}:\n")
                        f.write(f"  Mean Ratio: {quality.mean():.4f}\n")
                        f.write(f"  Median Ratio: {quality.median():.4f}\n")
                        f.write(f"  Min Ratio: {quality.min():.4f}\n")
                        f.write(f"  Max Ratio: {quality.max():.4f}\n")
                        f.write(f"  Std Dev: {quality.std():.4f}\n")

                        # Success rates
                        optimal = (quality == 1.0).mean() * 100
                        within_5 = (quality <= 1.05).mean() * 100
                        within_10 = (quality <= 1.10).mean() * 100

                        f.write(f"  Optimal solutions: {optimal:.1f}%\n")
                        f.write(f"  Within 5% of optimal: {within_5:.1f}%\n")
                        f.write(f"  Within 10% of optimal: {within_10:.1f}%\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF EXTENDED REPORT\n")
            f.write("=" * 80 + "\n")


def main():
    """Extended main evaluation pipeline with all solver variants"""
    print("=" * 60)
    print("EXTENDED WP1.c EVALUATION: ALL SOLVER VARIANTS")
    print("=" * 60)

    # Initialize extended evaluator
    evaluator = ExtendedWP1cEvaluator()

    # Check if test graphs exist
    test_dir = "test_graphs/generated/perturbed"
    if not Path(test_dir).exists():
        print(f"Test directory {test_dir} not found.")
        print("Please run generate_test_graphs.py first.")
        return

    # Run extended evaluation
    print("\n1. Running extended evaluation suite...")
    print("   Evaluating with:")
    print("   - Chalupa heuristic")
    print("   - Standard ILP")
    print("   - ILP with warmstart")
    print("   - Reduced ILP")
    print("   - Interactive reduced ILP")

    df = evaluator.run_evaluation_suite(test_dir)

    if df.empty:
        print("No evaluation results. Exiting.")
        return

    print(f"\n Evaluated {len(df)} instances")

    # Print quick comparison
    print("\nQUICK METHOD COMPARISON:")
    print("-" * 40)

    methods = ['chalupa', 'ilp', 'ilp_warmstart', 'reduced_ilp', 'interactive_ilp']
    for method in methods:
        time_col = f'{method}_time'
        if time_col in df.columns:
            avg_time = df[time_col].dropna().mean()
            print(f"{method:20s}: {avg_time:8.3f}s avg runtime")

    # Generate all plots
    print("\n2. Creating extended method comparison plots...")
    evaluator.create_method_comparison_plots()

    print("\n3. Creating reduction effectiveness analysis...")
    evaluator.create_reduction_effectiveness_plot()

    print("\n4. Analyzing _chalupa_warmstart effectiveness...")
    evaluator.analyze_warmstart_effectiveness()

    print("\n5. Creating original runtime analysis plots...")
    # Create instance of original evaluator for backward compatibility
    original_evaluator = WP1cEvaluator(output_dir=evaluator.output_dir)
    original_evaluator.df = df  # Share the dataframe
    original_evaluator.results = evaluator.results
    original_evaluator.timestamp = evaluator.timestamp
    original_evaluator.create_runtime_plots()

    print("\n6. Creating original quality analysis plots...")
    original_evaluator.create_quality_plots()

    print("\n7. Creating perturbation analysis...")
    original_evaluator.create_perturbation_analysis()

    # Generate comprehensive summary report
    print("\n8. Generating extended summary report...")
    evaluator.generate_extended_summary_report()

    # Also generate original report for backward compatibility
    print("\n9. Generating original summary report...")
    original_evaluator.generate_summary_report()

    print("\n" + "=" * 60)
    print("EXTENDED EVALUATION COMPLETE!")
    print(f"Results saved to: {evaluator.output_dir}")
    print("=" * 60)

    # Print extended summary
    print("\nEXTENDED SUMMARY:")
    print("-" * 40)

    # Best method by speed
    avg_times = {}
    for method in methods:
        time_col = f'{method}_time'
        if time_col in df.columns:
            avg_time = df[time_col].dropna().mean()
            if not pd.isna(avg_time):
                avg_times[method] = avg_time

    if avg_times:
        fastest = min(avg_times, key=avg_times.get)
        print(f"Fastest method: {fastest} ({avg_times[fastest]:.3f}s avg)")

    # Best exact method by speed
    exact_methods = ['ilp', 'ilp_warmstart', 'reduced_ilp', 'interactive_ilp']
    exact_times = {m: avg_times[m] for m in exact_methods if m in avg_times}
    if exact_times:
        fastest_exact = min(exact_times, key=exact_times.get)
        print(f"Fastest exact method: {fastest_exact} ({exact_times[fastest_exact]:.3f}s avg)")

    # Quality summary for heuristic
    if 'chalupa_quality_ratio' in df.columns:
        chalupa_quality = df['chalupa_quality_ratio'].dropna()
        if not chalupa_quality.empty:
            print(f"\nChalupa quality:")
            print(f"  Average ratio: {chalupa_quality.mean():.3f}")
            print(f"  Finds optimal: {(chalupa_quality == 1.0).mean() * 100:.1f}%")
            print(f"  Within 10% of optimal: {(chalupa_quality <= 1.10).mean() * 100:.1f}%")

    # Enhancement effectiveness
    print("\nEnhancement effectiveness:")

    if 'ilp_time' in df.columns and 'ilp_warmstart_time' in df.columns:
        valid = df[['ilp_time', 'ilp_warmstart_time']].notna().all(axis=1)
        if valid.any():
            warmstart_speedup = (df.loc[valid, 'ilp_time'] /
                                 df.loc[valid, 'ilp_warmstart_time']).mean()
            print(f"  Warmstart speedup: {warmstart_speedup:.2f}x")

    if 'ilp_time' in df.columns and 'reduced_ilp_time' in df.columns:
        valid = df[['ilp_time', 'reduced_ilp_time']].notna().all(axis=1)
        if valid.any():
            reduction_speedup = (df.loc[valid, 'ilp_time'] /
                                 df.loc[valid, 'reduced_ilp_time']).mean()
            print(f"  Reduction speedup: {reduction_speedup:.2f}x")

    if 'ilp_time' in df.columns and 'interactive_ilp_time' in df.columns:
        valid = df[['ilp_time', 'interactive_ilp_time']].notna().all(axis=1)
        if valid.any():
            interactive_speedup = (df.loc[valid, 'ilp_time'] /
                                   df.loc[valid, 'interactive_ilp_time']).mean()
            print(f"  Interactive speedup: {interactive_speedup:.2f}x")


if __name__ == "__main__":
    main()