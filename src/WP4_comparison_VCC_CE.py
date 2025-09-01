"""
WP4: Comparison of Vertex Clique Cover and Cluster Editing Solutions

This module implements the comparison framework for WP4, comparing solutions
from the Vertex Clique Cover (VCC) and Cluster Editing (CE) problems.

Main objectives:
- Compare θ(G) from VCC with C(G) from CE
- Analyze solution quality and structural differences
- Implement cross-optimization heuristics (Bonus)
- Generate comprehensive reports and visualizations

Usage:
    python src/wp4_comparison.py [options]

Options:
    --test-dir PATH      Directory with test graphs (default: test_graphs/generated)
    --output-dir PATH    Output directory for results (default: results/wp4)
    --quick              Run quick test with fewer instances
    --verbose            Enable verbose output
    --skip-visualizations   Skip generating plots (faster execution)
"""
# src/wp4_comparison_csv.py
"""
WP4: Comparison of VCC and CE using existing CSV results
Compares θ(G) from VCC with C(G) from CE without re-running solvers
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import argparse
from typing import Dict, List, Tuple, Optional
from datetime import datetime

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class WP4CSVComparison:
    """Compare VCC and CE results from existing CSV files."""

    def __init__(self, output_dir: str = "results/wp4"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def load_and_match_results(self,
                               vcc_csv: str = "results/WP1and2/evaluation_VCC.csv",
                               ce_stats_csv: str = "results/wp3/statistical_improvements_CE.csv",
                               ce_eff_csv: str = "results/wp3/effectiveness_results_CE.csv") -> pd.DataFrame:
        """Load CSV files and match graphs between VCC and CE results."""

        print("Loading CSV files...")
        vcc_df = pd.read_csv(vcc_csv)
        ce_stats_df = pd.read_csv(ce_stats_csv) if Path(ce_stats_csv).exists() else None
        ce_eff_df = pd.read_csv(ce_eff_csv) if Path(ce_eff_csv).exists() else None

        # Extract best theta from VCC (prioritize exact methods)
        vcc_df['best_theta'], vcc_df['vcc_method'] = zip(*vcc_df.apply(self._get_best_theta, axis=1))
        vcc_df['best_vcc_time'] = vcc_df.apply(self._get_corresponding_time, axis=1)

        # Extract graph identifiers
        vcc_df['graph_id'] = vcc_df['filepath'].apply(self._extract_graph_id)

        # Prepare comparison dataframe
        comparison_results = []

        # Match with CE results
        if ce_stats_df is not None and 'C_G' in ce_stats_df.columns:
            # Use statistical results (has C_G)
            for _, ce_row in ce_stats_df.iterrows():
                graph_id = ce_row['graph']
                matching_vcc = vcc_df[vcc_df['graph_id'] == graph_id]

                if not matching_vcc.empty:
                    vcc_row = matching_vcc.iloc[0]
                    comparison_results.append({
                        'graph': graph_id,
                        'n_nodes': ce_row['n_nodes'],
                        'n_edges': ce_row.get('n_edges', vcc_row['n_edges']),
                        'density': vcc_row['density'],
                        'theta': vcc_row['best_theta'],
                        'C_G': ce_row['C_G'],
                        'ratio': ce_row['C_G'] / vcc_row['best_theta'] if vcc_row['best_theta'] > 0 else np.inf,
                        'vcc_method': vcc_row['vcc_method'],
                        'vcc_time': vcc_row['best_vcc_time'],
                        'ce_time': ce_row['mean_time_with_kernel'],
                        'theta_minus_C': vcc_row['best_theta'] - ce_row['C_G']
                    })

        elif ce_eff_df is not None and 'n_clusters' in ce_eff_df.columns:
            # Use effectiveness results (has n_clusters)
            # Group by graph and take best config
            ce_grouped = ce_eff_df.groupby('graph').agg({
                'n_nodes': 'first',
                'n_edges': 'first',
                'n_clusters': 'min',  # Take minimum C(G)
                'time_seconds': 'min'
            }).reset_index()

            for _, ce_row in ce_grouped.iterrows():
                graph_id = ce_row['graph']
                matching_vcc = vcc_df[vcc_df['graph_id'] == graph_id]

                if not matching_vcc.empty:
                    vcc_row = matching_vcc.iloc[0]
                    comparison_results.append({
                        'graph': graph_id,
                        'n_nodes': ce_row['n_nodes'],
                        'n_edges': ce_row['n_edges'],
                        'density': vcc_row['density'],
                        'theta': vcc_row['best_theta'],
                        'C_G': ce_row['n_clusters'],
                        'ratio': ce_row['n_clusters'] / vcc_row['best_theta'] if vcc_row['best_theta'] > 0 else np.inf,
                        'vcc_method': vcc_row['vcc_method'],
                        'vcc_time': vcc_row['best_vcc_time'],
                        'ce_time': ce_row['time_seconds'],
                        'theta_minus_C': vcc_row['best_theta'] - ce_row['n_clusters']
                    })

        comparison_df = pd.DataFrame(comparison_results)

        # Filter out invalid entries
        comparison_df = comparison_df[
            (comparison_df['theta'] > 0) &
            (comparison_df['C_G'] > 0) &
            np.isfinite(comparison_df['ratio'])
            ]

        print(f"Matched {len(comparison_df)} graphs between VCC and CE results")

        # Save comparison data
        comparison_df.to_csv(self.output_dir / f"wp4_comparison_{self.timestamp}.csv", index=False)

        return comparison_df

    def _get_best_theta(self, row) -> Tuple[int, str]:
        """Get best theta value from VCC results, prioritizing exact methods."""
        if pd.notna(row.get('interactive_ilp_theta')):
            return int(row['interactive_ilp_theta']), 'interactive_ilp'
        elif pd.notna(row.get('reduced_ilp_theta')):
            return int(row['reduced_ilp_theta']), 'reduced_ilp'
        elif pd.notna(row.get('ilp_warmstart_theta')):
            return int(row['ilp_warmstart_theta']), 'ilp_warmstart'
        elif pd.notna(row.get('ilp_theta')):
            return int(row['ilp_theta']), 'ilp'
        elif pd.notna(row.get('chalupa_theta')):
            return int(row['chalupa_theta']), 'chalupa'
        return 0, 'none'

    def _get_corresponding_time(self, row) -> float:
        """Get runtime corresponding to best theta method."""
        _, method = self._get_best_theta(row)
        time_col = f"{method}_time"
        return row.get(time_col, 0.0) if time_col in row else 0.0

    def _extract_graph_id(self, filepath: str) -> str:
        """Extract graph identifier from filepath."""
        # Extract filename without extension
        path = Path(filepath)
        return path.stem

    def analyze_results(self, df: pd.DataFrame) -> Dict:
        """Perform statistical analysis on comparison results."""

        analysis = {
            'basic_stats': {
                'n_graphs': len(df),
                'mean_theta': df['theta'].mean(),
                'mean_C_G': df['C_G'].mean(),
                'mean_ratio': df['ratio'].mean(),
                'median_ratio': df['ratio'].median(),
                'std_ratio': df['ratio'].std(),
                'min_ratio': df['ratio'].min(),
                'max_ratio': df['ratio'].max()
            }
        }

        # Check mathematical invariant θ(G) ≤ C(G)
        violations = df[df['theta'] > df['C_G']]
        analysis['invariant'] = {
            'satisfied': len(violations) == 0,
            'violations': len(violations),
            'violation_rate': len(violations) / len(df) * 100
        }

        if len(violations) > 0:
            print(f"WARNING: Found {len(violations)} violations of θ(G) ≤ C(G)!")
            print(violations[['graph', 'theta', 'C_G', 'ratio']])

        # Statistical tests
        if len(df) > 1:
            # Wilcoxon signed-rank test
            try:
                stat, p_value = stats.wilcoxon(df['theta'], df['C_G'])
                analysis['wilcoxon'] = {
                    'statistic': stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
            except:
                analysis['wilcoxon'] = {'error': 'Could not perform test'}

            # Paired t-test
            try:
                t_stat, t_p = stats.ttest_rel(df['theta'], df['C_G'])
                analysis['ttest'] = {
                    't_statistic': t_stat,
                    'p_value': t_p,
                    'significant': t_p < 0.05
                }
            except:
                analysis['ttest'] = {'error': 'Could not perform test'}

            # Correlation analysis
            analysis['correlations'] = {
                'theta_C_pearson': stats.pearsonr(df['theta'], df['C_G'])[0],
                'density_ratio_pearson': stats.pearsonr(df['density'], df['ratio'])[0] if 'density' in df else None,
                'size_ratio_pearson': stats.pearsonr(df['n_nodes'], df['ratio'])[0]
            }

            # Group analysis
            if 'density' in df:
                # By density
                df['density_cat'] = pd.cut(df['density'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                           labels=['very_sparse', 'sparse', 'medium', 'dense', 'very_dense'])
                density_analysis = df.groupby('density_cat', observed=False).agg({
                    'ratio': ['mean', 'std', 'count'],
                    'theta': 'mean',
                    'C_G': 'mean'
                })
                analysis['by_density'] = density_analysis.to_dict()

            # By size
            df['size_cat'] = pd.cut(df['n_nodes'], bins=[0, 20, 50, 100, np.inf],
                                    labels=['tiny', 'small', 'medium', 'large'])
            size_analysis = df.groupby('size_cat', observed=False).agg({
                'ratio': ['mean', 'std', 'count'],
                'theta': 'mean',
                'C_G': 'mean'
            })
            analysis['by_size'] = size_analysis.to_dict()

            return analysis

    def create_visualizations(self, df: pd.DataFrame, analysis: Dict):
        """Create comprehensive visualizations."""

        # 1. Main comparison plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # θ vs C scatter
        ax = axes[0, 0]
        scatter = ax.scatter(df['theta'], df['C_G'],
                             c=df['density'] if 'density' in df else 'blue',
                             cmap='viridis', s=50, alpha=0.7)

        # Add diagonal line
        max_val = max(df['theta'].max(), df['C_G'].max())
        ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='θ = C')

        # Add regression line
        z = np.polyfit(df['theta'], df['C_G'], 1)
        p = np.poly1d(z)
        ax.plot(df['theta'], p(df['theta']), 'g-', alpha=0.5,
                label=f'Fit: C = {z[0]:.2f}θ + {z[1]:.2f}')

        ax.set_xlabel('θ(G) - Vertex Clique Cover', fontsize=11)
        ax.set_ylabel('C(G) - Cluster Editing', fontsize=11)
        ax.set_title('VCC vs CE Comparison', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if 'density' in df:
            plt.colorbar(scatter, ax=ax, label='Density')

        # Ratio distribution
        ax = axes[0, 1]
        ax.hist(df['ratio'], bins=20, edgecolor='black', alpha=0.7, color='skyblue')
        ax.axvline(1.0, color='red', linestyle='--', label='Ideal (C/θ = 1)')
        ax.axvline(df['ratio'].mean(), color='green', linestyle='-',
                   label=f'Mean = {df["ratio"].mean():.2f}')
        ax.set_xlabel('C(G)/θ(G) Ratio', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Ratio Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Difference histogram
        ax = axes[0, 2]
        differences = df['C_G'] - df['theta']
        ax.hist(differences, bins=20, edgecolor='black', alpha=0.7, color='coral')
        ax.axvline(0, color='red', linestyle='--', label='No difference')
        ax.axvline(differences.mean(), color='green', linestyle='-',
                   label=f'Mean = {differences.mean():.1f}')
        ax.set_xlabel('C(G) - θ(G)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Absolute Difference', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Ratio by size
        ax = axes[1, 0]
        df.boxplot(column='ratio', by='size_cat', ax=ax)
        ax.set_xlabel('Graph Size Category', fontsize=11)
        ax.set_ylabel('C(G)/θ(G) Ratio', fontsize=11)
        ax.set_title('Ratio by Graph Size', fontsize=12, fontweight='bold')
        plt.sca(ax)
        plt.xticks(rotation=45)

        # Ratio by density
        ax = axes[1, 1]
        if 'density_cat' in df:
            df.boxplot(column='ratio', by='density_cat', ax=ax)
            ax.set_xlabel('Density Category', fontsize=11)
            ax.set_ylabel('C(G)/θ(G) Ratio', fontsize=11)
            ax.set_title('Ratio by Graph Density', fontsize=12, fontweight='bold')
            plt.sca(ax)
            plt.xticks(rotation=45)

        # Runtime comparison
        ax = axes[1, 2]
        if 'vcc_time' in df and 'ce_time' in df:
            ax.scatter(df['vcc_time'], df['ce_time'], alpha=0.7, s=30)
            max_time = max(df['vcc_time'].max(), df['ce_time'].max())
            ax.plot([0, max_time], [0, max_time], 'r--', alpha=0.5, label='Equal time')
            ax.set_xlabel('VCC Time (s)', fontsize=11)
            ax.set_ylabel('CE Time (s)', fontsize=11)
            ax.set_title('Runtime Comparison', fontsize=12, fontweight='bold')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.suptitle('WP4: VCC vs CE Comprehensive Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / f"wp4_analysis_{self.timestamp}.png", dpi=300, bbox_inches='tight')
        plt.show()

        # 2. Additional analysis plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Correlation heatmap
        ax = axes[0]
        corr_matrix = df[['theta', 'C_G', 'ratio', 'n_nodes', 'density']].corr() if 'density' in df else \
            df[['theta', 'C_G', 'ratio', 'n_nodes']].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, ax=ax, cbar_kws={'label': 'Correlation'})
        ax.set_title('Correlation Matrix', fontsize=12, fontweight='bold')

        # Summary statistics
        ax = axes[1]
        ax.axis('off')

        # Format p-values correctly
        wilcoxon_p = 'N/A'
        if 'wilcoxon' in analysis and 'p_value' in analysis['wilcoxon']:
            wilcoxon_p = f"{analysis['wilcoxon']['p_value']:.4f}"

        ttest_p = 'N/A'
        if 'ttest' in analysis and 'p_value' in analysis['ttest']:
            ttest_p = f"{analysis['ttest']['p_value']:.4f}"

        summary_text = f"""
    SUMMARY STATISTICS
    {'=' * 30}

    Graphs analyzed: {analysis['basic_stats']['n_graphs']}

    θ(G) Statistics:
      Mean: {analysis['basic_stats']['mean_theta']:.2f}

    C(G) Statistics:
      Mean: {analysis['basic_stats']['mean_C_G']:.2f}

    C/θ Ratio:
      Mean: {analysis['basic_stats']['mean_ratio']:.3f}
      Median: {analysis['basic_stats']['median_ratio']:.3f}
      Std Dev: {analysis['basic_stats']['std_ratio']:.3f}
      Range: [{analysis['basic_stats']['min_ratio']:.3f}, {analysis['basic_stats']['max_ratio']:.3f}]

    Invariant θ ≤ C:
      Satisfied: {'✓' if analysis['invariant']['satisfied'] else '✗'}
      Violations: {analysis['invariant']['violations']}

    Statistical Tests:
      Wilcoxon p-value: {wilcoxon_p}
      Paired t-test p: {ttest_p}
    """

        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
                fontsize=10, fontfamily='monospace', verticalalignment='top')

        plt.tight_layout()
        plt.savefig(self.output_dir / f"wp4_summary_{self.timestamp}.png", dpi=300, bbox_inches='tight')
        plt.show()

    def generate_report(self, df: pd.DataFrame, analysis: Dict):
        """Generate markdown report."""

        report_path = self.output_dir / f"wp4_report_{self.timestamp}.md"

        with open(report_path, 'w') as f:
            f.write("# WP4: VCC vs CE Comparison Report\n\n")
            f.write(f"Generated: {self.timestamp}\n\n")

            f.write("## Executive Summary\n\n")
            f.write(f"- **Graphs analyzed**: {analysis['basic_stats']['n_graphs']}\n")
            f.write(f"- **Mean C/θ ratio**: {analysis['basic_stats']['mean_ratio']:.3f}\n")
            f.write(
                f"- **Invariant θ ≤ C satisfied**: {'Yes' if analysis['invariant']['satisfied'] else f'No ({analysis["invariant"]["violations"]} violations)'}\n\n")

            f.write("## Key Findings\n\n")

            # Main finding
            if analysis['basic_stats']['mean_ratio'] > 1:
                f.write(
                    f"1. **Cluster Editing requires more clusters**: On average, C(G) = {analysis['basic_stats']['mean_ratio']:.2f} × θ(G)\n")
            else:
                f.write(
                    f"1. **Methods produce similar results**: C(G) ≈ θ(G) with ratio {analysis['basic_stats']['mean_ratio']:.2f}\n")

            # Statistical significance
            if 'wilcoxon' in analysis:
                if analysis['wilcoxon'].get('significant'):
                    f.write("2. **Statistically significant difference**: Wilcoxon test confirms θ < C (p < 0.05)\n")
                else:
                    f.write("2. **No significant difference**: Statistical tests show θ ≈ C\n")

            # Correlations
            if 'correlations' in analysis:
                corr = analysis['correlations']
                f.write(
                    f"3. **Strong correlation**: θ and C are highly correlated (r = {corr['theta_C_pearson']:.3f})\n")

            f.write("\n## Detailed Statistics\n\n")

            # Table of results by category
            f.write("### By Graph Size\n\n")
            f.write("| Size Category | Mean θ | Mean C | Mean C/θ | Count |\n")
            f.write("|--------------|--------|--------|----------|-------|\n")

            size_groups = df.groupby('size_cat').agg({
                'theta': 'mean',
                'C_G': 'mean',
                'ratio': 'mean',
                'graph': 'count'
            })

            for cat, row in size_groups.iterrows():
                f.write(f"| {cat} | {row['theta']:.1f} | {row['C_G']:.1f} | {row['ratio']:.3f} | {row['graph']} |\n")

            if 'density_cat' in df:
                f.write("\n### By Graph Density\n\n")
                f.write("| Density Category | Mean θ | Mean C | Mean C/θ | Count |\n")
                f.write("|-----------------|--------|--------|----------|-------|\n")

                density_groups = df.groupby('density_cat').agg({
                    'theta': 'mean',
                    'C_G': 'mean',
                    'ratio': 'mean',
                    'graph': 'count'
                })

                for cat, row in density_groups.iterrows():
                    f.write(
                        f"| {cat} | {row['theta']:.1f} | {row['C_G']:.1f} | {row['ratio']:.3f} | {row['graph']} |\n")

            f.write("\n## Files Generated\n\n")
            f.write(f"- Comparison data: `wp4_comparison_{self.timestamp}.csv`\n")
            f.write(f"- Analysis plots: `wp4_analysis_{self.timestamp}.png`\n")
            f.write(f"- Summary plot: `wp4_summary_{self.timestamp}.png`\n")

        print(f"Report saved to: {report_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='WP4: Compare VCC and CE from CSV results')
    parser.add_argument('--vcc-csv', type=str,
                        default='results/WP1and2/evaluation_results_VCC.csv',
                        help='Path to VCC results CSV')
    parser.add_argument('--ce-stats-csv', type=str,
                        default='results/wp3/statistical_improvements_CE.csv',
                        help='Path to CE statistical results CSV')
    parser.add_argument('--ce-eff-csv', type=str,
                        default='results/wp3/effectiveness_results_CE.csv',
                        help='Path to CE effectiveness results CSV')
    parser.add_argument('--output-dir', type=str, default='results/wp4',
                        help='Output directory for results')

    args = parser.parse_args()

    print("=" * 80)
    print("WP4: COMPARISON OF VCC AND CE SOLUTIONS")
    print("=" * 80)

    # Initialize comparison framework
    comparator = WP4CSVComparison(args.output_dir)

    # Load and match results
    print("\nLoading and matching results from CSV files...")
    df = comparator.load_and_match_results(
        vcc_csv=args.vcc_csv,
        ce_stats_csv=args.ce_stats_csv,
        ce_eff_csv=args.ce_eff_csv
    )

    if df.empty:
        print("ERROR: No matching graphs found between VCC and CE results!")
        return

    # Analyze results
    print("\nAnalyzing comparison results...")
    analysis = comparator.analyze_results(df)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Graphs compared: {len(df)}")
    print(f"Mean θ(G): {analysis['basic_stats']['mean_theta']:.2f}")
    print(f"Mean C(G): {analysis['basic_stats']['mean_C_G']:.2f}")
    print(f"Mean C/θ ratio: {analysis['basic_stats']['mean_ratio']:.3f}")
    print(
        f"Invariant θ ≤ C satisfied: {'Yes' if analysis['invariant']['satisfied'] else f'No ({analysis["invariant"]["violations"]} violations)'}")

    # Create visualizations
    print("\nGenerating visualizations...")
    comparator.create_visualizations(df, analysis)

    # Generate report
    print("\nGenerating report...")
    comparator.generate_report(df, analysis)

    print("\n" + "=" * 60)
    print("WP4 ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
"""
Pre-Thoughts 120825
Main objective: Compare the solutions from the two conceptually similar problems - Vertex Clique Cover and Cluster Editing.
What needs to be done:

Compare the number of clusters obtained by both methods:
- C(G): number of clusters from cluster editing
- θ(G): vertex clique cover number

Analyze the quality of solutions between the two approaches to understand how well cluster editing
performs compared to vertex clique cover.
Bonus task: Develop a good heuristic to obtain a better solution for one problem from an
exact solution of the other problem (later)

Prerequisites for WP4:
- WP1: The vertex clique cover algorithms (Chalupa's heuristic and ILP)
- WP3: The cluster editing algorithms with kernelization
- WP0: Test instances to run comparisons on

approach
1. checkup for problem definition:
    - Vertex Clique cover
        - aim: cover all nodes with minimal number of cliques
        - output: θ(G) = minimal number of cliques
        - constraint: every node must be at least in one clique

    - Cluster Editing
        - aim: minimal edge modification to keep/maintain disjunct cliques
        - output:  C(G) = number of resulting clusters
        - constraint: resulting clusters are disjunct and completely connected

    - critical differences between VertexCliqueColoring and ClusterEditing:
        - VCC erlaubt overlaps → θ(G) ≤ C(G) immer
        - CE erzwingt Disjunktheit → kann mehr Cluster benötigen
        - Metriken unterscheiden sich: VCC zählt Cliquen, CE zählt Kantenmodifikationen

2. interface design ___  something like:

    class ClusteringResult:
    #Einheitliches Format für beide Probleme
    graph: nx.Graph
    clusters: List[Set[int]]  # Liste von Knotenmengen
    num_clusters: int
    method: str  # "vcc" oder "ce"
    metadata: Dict  # z.B. Laufzeit, Kantenmodifikationen

    def validate(self) -> bool:
        #Prüfe mathematische Korrektheit
        if self.method == "vcc":
            return self._validate_clique_cover()
        elif self.method == "ce":
            return self._validate_cluster_editing()

    def _validate_clique_cover(self) -> bool:
        # Jeder Knoten muss überdeckt sein
        covered = set().union(*self.clusters)
        if covered != set(self.graph.nodes()):
            return False
        # Jedes Cluster muss eine Clique sein
        for cluster in self.clusters:
            if not self._is_clique(cluster):
                return False
        return True

    def _validate_cluster_editing(self) -> bool:
        # Cluster müssen disjunkt sein
        for i, c1 in enumerate(self.clusters):
            for c2 in self.clusters[i+1:]:
                if c1 & c2:  # Schnittmenge nicht leer
                    return False
        # Jedes Cluster muss eine Clique sein
        for cluster in self.clusters:
            if not self._is_clique(cluster):
                return False
        return True

3. adapter for prerequisitites ___ idea:

    class SolverAdapter:
    #Adapter-Pattern für einheitliche Schnittstelle

    def __init__(self, vcc_solver, ce_solver):
        self.vcc_solver = vcc_solver  # aus WP1
        self.ce_solver = ce_solver    # aus WP3

    def solve_vcc(self, graph: nx.Graph, **kwargs) -> ClusteringResult:
        #Wrapper für VCC-Solver
        # Annahme über VCC-Solver Interface aus WP1
        result = self.vcc_solver.solve(graph, **kwargs)

        # Konvertiere in einheitliches Format
        return ClusteringResult(
            graph=graph,
            clusters=self._extract_vcc_clusters(result),
            num_clusters=result.get('theta'),
            method="vcc",
            metadata={
                'time': result.get('time'),
                'algorithm': result.get('algorithm', 'unknown')
            }
        )

    def solve_ce(self, graph: nx.Graph, **kwargs) -> ClusteringResult:
        #Wrapper für CE-Solver
        # Annahme über CE-Solver Interface aus WP3
        result = self.ce_solver.solve(graph, **kwargs)

        return ClusteringResult(
            graph=graph,
            clusters=self._extract_ce_clusters(result),
            num_clusters=len(result.get('clusters', [])),
            method="ce",
            metadata={
                'time': result.get('time'),
                'modifications': result.get('num_modifications'),
                'kernelization_used': result.get('kernelization', False)
            }
        )


4. framework for comparison ... like:

    class ComparisonFramework:
    #Hauptklasse für WP4

    def __init__(self, adapter: SolverAdapter):
        self.adapter = adapter
        self.results = []

    def compare_solutions(self, graph: nx.Graph) -> Dict:
        #Kernfunktion: Vergleiche VCC und CE Lösungen

        # Löse beide Probleme
        vcc_result = self.adapter.solve_vcc(graph)
        ce_result = self.adapter.solve_ce(graph)

        # Validierung
        assert vcc_result.validate(), "VCC Lösung ungültig!"
        assert ce_result.validate(), "CE Lösung ungültig!"

        # Berechne Vergleichsmetriken
        comparison = {
            'graph_stats': self._compute_graph_stats(graph),
            'theta': vcc_result.num_clusters,
            'C': ce_result.num_clusters,
            'ratio': ce_result.num_clusters / vcc_result.num_clusters,
            'overlap_analysis': self._analyze_overlaps(vcc_result, ce_result),
            'quality_metrics': self._compute_quality_metrics(vcc_result, ce_result)
        }

        return comparison

    def _analyze_overlaps(self, vcc: ClusteringResult, ce: ClusteringResult) -> Dict:
        #Analysiere strukturelle Unterschiede

        # Wie viele VCC-Cliquen überlappen?
        overlap_matrix = np.zeros((len(vcc.clusters), len(vcc.clusters)))
        for i, c1 in enumerate(vcc.clusters):
            for j, c2 in enumerate(vcc.clusters):
                if i != j:
                    overlap_matrix[i,j] = len(c1 & c2) / min(len(c1), len(c2))

        # Vergleiche Cluster-Zuordnungen
        agreement = self._compute_rand_index(vcc.clusters, ce.clusters)

        return {
            'avg_overlap': np.mean(overlap_matrix[overlap_matrix > 0]),
            'max_overlap': np.max(overlap_matrix),
            'rand_index': agreement
        }


5. Bonus haha lets gooo
    class CrossOptimizationHeuristic:
    # Bonus: Nutze Lösungen gegenseitig

    def improve_ce_from_vcc(self, graph: nx.Graph,
                            vcc_solution: ClusteringResult) -> ClusteringResult:
        '''
        Idee: VCC-Cliquen als Startpunkt für CE
        1. Beginne mit VCC-Cliquen
        2. Löse Überlappungen durch lokale Optimierung
        3. Minimiere Kantenmodifikationen
        '''

        # Schritt 5.1: Konfliktgraph für überlappende Knoten
        conflicts = self._find_overlapping_nodes(vcc_solution.clusters)

        # Schritt 5.2: Zuordnung durch gewichtetes Matching
        assignment = {}
        for node in conflicts:
            best_cluster = self._find_best_cluster_assignment(
                node, vcc_solution.clusters, graph
            )
            assignment[node] = best_cluster

        # Schritt 5.3: Konstruiere disjunkte Cluster
        disjoint_clusters = self._make_disjoint(vcc_solution.clusters, assignment)

        # Schritt 5.4: Lokale Verbesserung
        optimized = self._local_search(disjoint_clusters, graph)

        return ClusteringResult(
            graph=graph,
            clusters=optimized,
            num_clusters=len(optimized),
            method="ce_from_vcc",
            metadata={'source': 'vcc_based_heuristic'}
        )

    def _find_best_cluster_assignment(self, node: int,
                                     clusters: List[Set],
                                     graph: nx.Graph) -> int:
        #Weise Knoten dem Cluster mit meisten Nachbarn zu
        best_score = -1
        best_cluster = 0

        for idx, cluster in enumerate(clusters):
            if node in cluster:
                # Zähle existierende Kanten zu diesem Cluster
                neighbors_in_cluster = sum(
                    1 for n in cluster
                    if n != node and graph.has_edge(node, n)
                )
                score = neighbors_in_cluster / len(cluster)

                if score > best_score:
                    best_score = score
                    best_cluster = idx

        return best_cluster

6. Testing and Validating
    class WP4Validator:
    #Stelle mathematische Korrektheit sicher

    def validate_comparison(self, comp_result: Dict) -> bool:
        #Prüfe mathematische Invarianten

        # θ(G) ≤ C(G) muss immer gelten
        if comp_result['theta'] > comp_result['C']:
            raise ValueError(f"Invariante verletzt: θ(G)={comp_result['theta']} > C(G)={comp_result['C']}")

        # Ratio sollte ≥ 1 sein
        if comp_result['ratio'] < 1.0 - 1e-9:  # Numerische Toleranz
            raise ValueError(f"Ungültiges Verhältnis: {comp_result['ratio']}")

        return True

    def test_on_known_instances(self):
        #Teste mit Graphen bekannter Eigenschaften

        # Test 1: Perfekte Clique
        K5 = nx.complete_graph(5)
        result = self.framework.compare_solutions(K5)
        assert result['theta'] == 1 and result['C'] == 1

        # Test 2: Disjunkte Cliquen
        G = nx.disjoint_union(nx.complete_graph(3), nx.complete_graph(4))
        result = self.framework.compare_solutions(G)
        assert result['theta'] == 2 and result['C'] == 2

        # Test 3: Graph mit erzwungenen Überlappungen
        # ...

7. Main integration
    def main():
    #WP4 Hauptausführung

    # 7.1. Lade Solver aus WP1 und WP3
    from wp1.vcc_solver import VCCSolver
    from wp3.ce_solver import ClusterEditingSolver

    # 7.2. Initialisiere Framework
    adapter = SolverAdapter(VCCSolver(), ClusterEditingSolver())
    framework = ComparisonFramework(adapter)
    heuristic = CrossOptimizationHeuristic()

    # 7.3. Lade Testinstanzen aus WP0
    from wp0.generator import TestInstanceGenerator
    test_graphs = TestInstanceGenerator().generate_benchmark_suite()

    # 7.4. Führe systematischen Vergleich durch
    results = []
    for graph in test_graphs:
        comp = framework.compare_solutions(graph)

        # Bonus: Teste Heuristik
        improved = heuristic.improve_ce_from_vcc(
            graph,
            adapter.solve_vcc(graph)
        )
        comp['heuristic_improvement'] = improved.num_clusters / comp['C']

        results.append(comp)

    # 7.5. Statistische Auswertung
    analyze_results(results)
8. Statistical Analysis
    - Correlation Analysis of Graph properties and θ/C - ratio
    - confidence intervals and significance tests (Wilcoxon-Test: Ist θ signifikant kleiner als C?)
    - analysis for graph categories (skewed, uniform..)
9. Performance-Metrics
    -Runtime comp VCC vs CE for different graph sizes
    - analysis of needed memory
    -differences in Skalierbarkeit
10. Quality Metrics for Solutions
    - Modularität of found clusters
    - average cluster density
    - ratio of inter/intra cluster edges
11. maybe some advanced heuristic evaluation (BONUS lets gooo)
    - bidirectional improvements (CE→VCC und VCC→CE)
    - iterative Verfeinerungen between both methods ?
    - Approximation Quality Analysis (https://de.wikipedia.org/wiki/G%C3%BCte_von_Approximationsalgorithmen)
12. Visualization
    - Scatter Plots θ vs. C for different Graph Classes
    - Heatmap for Overlapping Matrices
    - graphic portrayal of cluster differences
13. Analysis of robustness
    - sensitivity against pertubation strengh
    - stability for different start configurations
    - edge cases
14. Exports/Reporting
    - CSV export of comparison metrics
    - automatical summary of results wäre nice
"""
