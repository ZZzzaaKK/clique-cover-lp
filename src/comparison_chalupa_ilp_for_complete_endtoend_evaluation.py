"""
WP1.c Evaluation: Comprehensive comparison of Chalupa heuristic vs ILP solver
Analyzes runtime, solution quality, and perturbation strength effects.

Author: Evaluation Script for Clique Cover Project
Date: 2024
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

# Import project modules (adjust path as needed)
sys.path.append('src')
from .wrapperV2 import ilp_wrapper, reduced_ilp_wrapper
from .wrappers import chalupa_wrapper
from .utils import txt_to_networkx, get_value
from simulator import GraphGenerator, GraphConfig

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


def main():
    """Main evaluation pipeline"""
    print("=" * 60)
    print("WP1.c EVALUATION: CHALUPA VS ILP COMPARISON")
    print("=" * 60)

    # Initialize evaluator
    evaluator = WP1cEvaluator()

    # Check if test graphs exist
    test_dir = "test_graphs/generated/perturbed"
    if not Path(test_dir).exists():
        print(f"Test directory {test_dir} not found.")
        print("Please run generate_test_graphs.py first.")
        return

    # Run evaluation
    print("\n1. Running evaluation suite...")
    df = evaluator.run_evaluation_suite(test_dir)

    if df.empty:
        print("No evaluation results. Exiting.")
        return

    print(f"\n✓ Evaluated {len(df)} instances")
    print(f"✓ Valid comparisons: {df['quality_ratio'].notna().sum()}")

    # Generate plots
    print("\n2. Creating runtime analysis plots...")
    evaluator.create_runtime_plots()

    print("\n3. Creating quality analysis plots...")
    evaluator.create_quality_plots()

    print("\n4. Creating perturbation analysis...")
    evaluator.create_perturbation_analysis()

    # Generate summary report
    print("\n5. Generating summary report...")
    evaluator.generate_summary_report()

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE!")
    print(f"Results saved to: {evaluator.output_dir}")
    print("=" * 60)

    # Print quick summary
    print("\nQUICK SUMMARY:")
    print("-" * 40)

    valid = df.dropna(subset=['quality_ratio'])
    if not valid.empty:
        print(f"Average Quality Ratio: {valid['quality_ratio'].mean():.3f}")
        print(f"Chalupa finds optimal: {(valid['quality_ratio'] == 1.0).mean() * 100:.1f}%")
        print(f"Within 10% of optimal: {(valid['quality_ratio'] <= 1.10).mean() * 100:.1f}%")

        speedup = df.dropna(subset=['chalupa_time', 'ilp_time'])
        if not speedup.empty:
            avg_speedup = (speedup['ilp_time'] / speedup['chalupa_time']).mean()
            print(f"Average Speedup: {avg_speedup:.1f}x")


if __name__ == "__main__":
    main()