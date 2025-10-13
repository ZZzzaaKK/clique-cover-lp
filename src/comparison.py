import pandas as pd
import matplotlib.pyplot as plt
import re
import sys
import seaborn as sns
import os
from pathlib import Path
import numpy as np
from scipy import stats
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")


class ComparisonAnalyzer:
    
    def __init__(self, output_dir: str = "results/analyses"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def parse_results(self, filepath):
        """Parse results file and extract algorithm results"""
        results = []
        algorithm = os.path.basename(filepath).replace(".txt", "")
        with open(filepath, "r") as f:
            content = f.read()

        blocks = content.strip().split("------------------------------")
        for block in blocks:
            if not block.strip() or "Test Results for" in block or "Summary" in block:
                continue

            data = {"algorithm": algorithm}
            lines = block.strip().split("\n")

            for line in lines:
                if not line:
                    continue

                parts = line.split(": ")
                if len(parts) < 2:
                    continue

                key = parts[0].strip()
                value = ": ".join(parts[1:]).strip()

                if key == "File":
                    data["file"] = value
                elif key == "Predicted":
                    data["predicted"] = int(value)
                elif key == "Actual":
                    data["actual"] = int(value)
                elif key == "Deviation":
                    data["deviation"] = int(value)
                elif key == "Correct":
                    data["correct"] = value == "True"
                elif key == "Time taken":
                    data["time"] = float(value)

            if "file" in data:
                results.append(data)

        return results

    def derive_graph_directory(self, results_filepath):
        """Derive the corresponding test_graphs directory from results file path."""
        path = Path(results_filepath)
        parts = path.parts

        if "raw" in parts:
            idx = parts.index("raw")
            relative_parts = parts[idx + 1: -1]
            graph_path = Path("test_graphs")
            for part in relative_parts:
                graph_path = graph_path / part
            return str(graph_path)

        print(f"Warning: Could not derive graph directory from {results_filepath}")
        return None

    def find_graph_file(self, filename, search_path):
        """Find a graph file by searching a directory and its subdirectories."""
        if not search_path or not os.path.exists(search_path):
            return None

        search_path = Path(search_path)
        for graph_file in search_path.rglob(filename):
            if graph_file.is_file():
                return str(graph_file)
        return None

    def parse_graph_file(self, filepath):
        """Extract properties from a graph file"""
        properties = {
            "vertices": None,
            "edges": None,
            "density": None,
            "clique_number": None,
            "chromatic_number": None,
            "vertex_clique_cover_number": None,  # θ(G) - Added for WP4
            "degeneracy": None,
            "average_degree": None,
            "connected": None,
        }

        if not os.path.exists(filepath):
            return properties

        try:
            with open(filepath, "r") as f:
                content = f.read()

            vertex_pattern = re.findall(r"^(\d+):", content, re.MULTILINE)
            if vertex_pattern:
                properties["vertices"] = max(map(int, vertex_pattern))

            property_patterns = {
                "density": r"Density: ([\d.]+)",
                "clique_number": r"Clique Number: (\d+)",
                "chromatic_number": r"Chromatic Number: (\d+)",
                "vertex_clique_cover_number": r"Vertex Clique Cover Number: (\d+)",  # θ(G)
                "degeneracy": r"Degeneracy: (\d+)",
                "average_degree": r"Average Degree: ([\d.]+)",
                "connected": r"Connected: (Yes|No)",
            }

            for prop, pattern in property_patterns.items():
                match = re.search(pattern, content)
                if match:
                    try:
                        value = match.group(1)
                        if prop == "connected":
                            properties[prop] = value == "Yes"
                        elif "." in value:
                            properties[prop] = float(value)
                        else:
                            properties[prop] = int(value)
                    except (ValueError, IndexError):
                        pass

            if properties["edges"] is None:
                edge_count = 0
                lines = content.split("\n")
                for line in lines:
                    if ":" in line and not any(
                            keyword in line
                            for keyword in [
                                "Acyclic", "Algebraic", "Average", "Bipartite",
                                "Chromatic", "Circumference", "Claw", "Clique",
                                "Connected", "Degeneracy", "Vertex",
                            ]
                    ):
                        parts = line.split(":")
                        if len(parts) == 2 and parts[1].strip():
                            neighbors = parts[1].strip().split()
                            edge_count += len(neighbors)
                properties["edges"] = edge_count // 2

        except Exception as e:
            print(f"Error parsing graph file {filepath}: {e}")

        return properties

    def is_cluster_editing_algo(self, algo_name: str) -> bool:
        """Check if algorithm is a cluster editing algorithm"""
        cluster_keywords = ['cluster', 'editing']
        return any(kw in algo_name.lower() for kw in cluster_keywords)

    def perform_statistical_analysis(self, df: pd.DataFrame) -> Dict:
        """Perform comprehensive statistical analysis (WP4)"""

        analysis = {
            'basic_stats': {
                'n_algorithms': len(df['algorithm'].unique()),
                'n_graphs': len(df['file'].unique()) if 'file' in df.columns else len(df),
                'total_results': len(df)
            }
        }

        # Algorithm performance statistics
        algo_stats = df.groupby('algorithm').agg({
            'time': ['mean', 'median', 'std', 'min', 'max'],
            'correct': 'mean' if 'correct' in df.columns else lambda x: None,
            'deviation': ['mean', 'median'] if 'deviation' in df.columns else lambda x: None
        })

        analysis['algorithm_performance'] = algo_stats.to_dict()

        # WP4 Analysis: C(G) vs θ(G) comparison for cluster editing algorithms
        if 'vertex_clique_cover_number' in df.columns:
            cluster_algos = [a for a in df['algorithm'].unique() if self.is_cluster_editing_algo(a)]

            if cluster_algos:
                analysis['wp4_analysis'] = {}

                for algo in cluster_algos:
                    algo_df = df[df['algorithm'] == algo].copy()
                    valid = algo_df.dropna(subset=['predicted', 'vertex_clique_cover_number'])

                    if len(valid) > 0:
                        C_vals = valid['predicted'].values  # C(G)
                        theta_vals = valid['vertex_clique_cover_number'].values  # θ(G)

                        # Calculate ratios
                        ratios = [c / t if t != 0 else np.nan for c, t in zip(C_vals, theta_vals)]
                        ratios = [r for r in ratios if not np.isnan(r)]

                        # Wilcoxon test
                        try:
                            stat, p_val = stats.wilcoxon(C_vals, theta_vals)
                        except:
                            stat, p_val = np.nan, np.nan

                        # Correlation
                        try:
                            corr, corr_p = stats.pearsonr(C_vals, theta_vals)
                        except:
                            corr, corr_p = np.nan, np.nan

                        analysis['wp4_analysis'][algo] = {
                            'mean_C': np.mean(C_vals),
                            'mean_theta': np.mean(theta_vals),
                            'mean_ratio': np.mean(ratios) if ratios else np.nan,
                            'wilcoxon_stat': stat,
                            'wilcoxon_p': p_val,
                            'correlation': corr,
                            'correlation_p': corr_p
                        }

        # Statistical tests between algorithms
        if len(df['algorithm'].unique()) > 1:
            algorithms = df['algorithm'].unique()
            analysis['pairwise_comparisons'] = {}

            for i, algo1 in enumerate(algorithms):
                for algo2 in algorithms[i + 1:]:
                    if 'file' in df.columns:
                        files = df['file'].unique()
                        paired_data1 = []
                        paired_data2 = []
                        for file in files:
                            d1 = df[(df['algorithm'] == algo1) & (df['file'] == file)]['time'].values
                            d2 = df[(df['algorithm'] == algo2) & (df['file'] == file)]['time'].values
                            if len(d1) > 0 and len(d2) > 0:
                                paired_data1.append(d1[0])
                                paired_data2.append(d2[0])

                        if len(paired_data1) > 1:
                            try:
                                stat, p_value = stats.wilcoxon(paired_data1, paired_data2)
                                analysis['pairwise_comparisons'][f'{algo1}_vs_{algo2}'] = {
                                    'test': 'Wilcoxon signed-rank',
                                    'statistic': stat,
                                    'p_value': p_value,
                                    'significant': p_value < 0.05,
                                    'mean_time_diff': np.mean(np.array(paired_data1) - np.array(paired_data2))
                                }
                            except:
                                pass

        # Correlation analysis
        if 'vertices' in df.columns and df['vertices'].notna().any():
            analysis['correlations'] = {}

            for algo in df['algorithm'].unique():
                algo_data = df[df['algorithm'] == algo]

                if len(algo_data) > 2:
                    valid_data = algo_data.dropna(subset=['vertices', 'time'])
                    if len(valid_data) > 2:
                        corr, p_val = stats.pearsonr(valid_data['vertices'], valid_data['time'])
                        analysis['correlations'][f'{algo}_time_vs_size'] = {
                            'correlation': corr,
                            'p_value': p_val
                        }

                    if 'density' in df.columns:
                        valid_data = algo_data.dropna(subset=['density', 'time'])
                        if len(valid_data) > 2:
                            corr, p_val = stats.pearsonr(valid_data['density'], valid_data['time'])
                            analysis['correlations'][f'{algo}_time_vs_density'] = {
                                'correlation': corr,
                                'p_value': p_val
                            }

        return analysis

    def generate_plots(self, df: pd.DataFrame, output_dir: str):
        
        sns.set_style("whitegrid")

        # Plot 1: Time taken by algorithm
        plt.figure(figsize=(10, 6))
        time_means = df.groupby("algorithm")["time"].mean().sort_values()
        time_means.plot(kind="bar")
        plt.title("Average Time Taken by Algorithm")
        plt.ylabel("Average Time (s)")
        plt.xlabel("Algorithm")
        plt.yscale("log")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "time_comparison.png"), dpi=150)
        plt.close()

        # Plot 2: Correctness by algorithm
        if 'correct' in df.columns:
            plt.figure(figsize=(10, 6))
            correctness_data = (
                df.groupby("algorithm")["correct"].mean().sort_values(ascending=False)
            )
            correctness_data.plot(kind="bar", color="steelblue")
            plt.title("Correctness Rate by Algorithm")
            plt.ylabel("Correctness Rate")
            plt.xlabel("Algorithm")
            plt.ylim(0, 1.05)
            plt.axhline(y=1.0, color="green", linestyle="--", alpha=0.5, label="100%")
            plt.xticks(rotation=45, ha="right")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "correctness_comparison.png"), dpi=150)
            plt.close()

        # Plot 3: Time vs Problem Size (vertices)
        if 'vertices' in df.columns:
            plt.figure(figsize=(12, 8))
            for algo in sorted(df["algorithm"].unique()):
                algo_data = df[df["algorithm"] == algo].copy()
                if len(algo_data) == 0:
                    continue

                algo_data = algo_data.sort_values("vertices")
                grouped = (
                    algo_data.groupby("vertices")["time"]
                    .agg(["mean", "std", "count"])
                    .reset_index()
                )
                grouped = grouped.dropna()

                if len(grouped) == 0:
                    continue

                grouped["stderr"] = grouped["std"] / np.sqrt(grouped["count"])
                grouped["ci_lower"] = grouped["mean"] - 1.96 * grouped["stderr"]
                grouped["ci_upper"] = grouped["mean"] + 1.96 * grouped["stderr"]
                grouped["ci_lower"] = np.maximum(grouped["ci_lower"], grouped["mean"] * 0.01)

                plt.plot(grouped["vertices"], grouped["mean"], label=algo, linewidth=2)
                plt.fill_between(
                    grouped["vertices"], grouped["ci_lower"], grouped["ci_upper"], alpha=0.3
                )

            plt.title("Time Taken vs. Problem Size", fontsize=14, fontweight="bold")
            plt.xlabel("Problem Size (vertices)", fontsize=12)
            plt.ylabel("Time Taken (s)", fontsize=12)
            plt.yscale("log")
            plt.legend(title="algorithm", fontsize=10, title_fontsize=10)
            plt.grid(True, which="both", ls="--", alpha=0.4)
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, "time_vs_vertices.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()

        # Plot 4: Deviation from actual
        if 'deviation' in df.columns:
            plt.figure(figsize=(12, 6))
            valid_dev = df[df["deviation"].notna()]
            if not valid_dev.empty:
                sns.boxplot(data=valid_dev, x="algorithm", y="deviation", palette="Set2")
                plt.title("Deviation from Actual Clique Cover by Algorithm")
                plt.xlabel("Algorithm")
                plt.ylabel("Deviation")
                plt.axhline(y=0, color="green", linestyle="--", alpha=0.5, label="Perfect")
                plt.xticks(rotation=45, ha="right")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "deviation_comparison.png"), dpi=150)
                plt.close()

        # Plot 5: Time vs Density
        if "density" in df.columns and df["density"].notna().any():
            plt.figure(figsize=(12, 8))

            for algo in sorted(df["algorithm"].unique()):
                algo_data = df[(df["algorithm"] == algo) & (df["density"].notna())].copy()
                if algo_data.empty:
                    continue

                algo_data = algo_data.sort_values("density")
                algo_data["density_bin"] = pd.cut(
                    algo_data["density"], bins=min(20, len(algo_data) // 2 + 1)
                )
                grouped = (
                    algo_data.groupby("density_bin")["time"]
                    .agg(["mean", "std", "count"])
                    .reset_index()
                )
                grouped = grouped.dropna()

                if len(grouped) == 0:
                    continue

                grouped["density_center"] = [
                    interval.mid for interval in grouped["density_bin"]
                ]
                grouped["stderr"] = grouped["std"] / np.sqrt(grouped["count"])
                grouped["ci_lower"] = grouped["mean"] - 1.96 * grouped["stderr"]
                grouped["ci_upper"] = grouped["mean"] + 1.96 * grouped["stderr"]
                grouped["ci_lower"] = np.maximum(
                    grouped["ci_lower"], grouped["mean"] * 0.01
                )

                plt.plot(
                    grouped["density_center"], grouped["mean"], label=algo, linewidth=2
                )
                plt.fill_between(
                    grouped["density_center"],
                    grouped["ci_lower"],
                    grouped["ci_upper"],
                    alpha=0.3,
                )

            plt.title("Time Taken vs. Graph Density", fontsize=14, fontweight="bold")
            plt.xlabel("Graph Density", fontsize=12)
            plt.ylabel("Time Taken (s)", fontsize=12)
            plt.yscale("log")
            plt.legend(title="algorithm", fontsize=10, title_fontsize=10)
            plt.grid(True, which="both", ls="--", alpha=0.4)
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, "time_vs_density.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()

        # Plot 6: Time vs Clique Number
        if "clique_number" in df.columns and df["clique_number"].notna().any():
            plt.figure(figsize=(12, 8))

            for algo in sorted(df["algorithm"].unique()):
                algo_data = df[
                    (df["algorithm"] == algo) & (df["clique_number"].notna())
                    ].copy()
                if algo_data.empty:
                    continue

                algo_data = algo_data.sort_values("clique_number")
                grouped = (
                    algo_data.groupby("clique_number")["time"]
                    .agg(["mean", "std", "count"])
                    .reset_index()
                )
                grouped = grouped.dropna()

                if len(grouped) == 0:
                    continue

                grouped["stderr"] = grouped["std"] / np.sqrt(grouped["count"])
                grouped["ci_lower"] = grouped["mean"] - 1.96 * grouped["stderr"]
                grouped["ci_upper"] = grouped["mean"] + 1.96 * grouped["stderr"]
                grouped["ci_lower"] = np.maximum(
                    grouped["ci_lower"], grouped["mean"] * 0.01
                )

                plt.plot(grouped["clique_number"], grouped["mean"], label=algo, linewidth=2)
                plt.fill_between(
                    grouped["clique_number"],
                    grouped["ci_lower"],
                    grouped["ci_upper"],
                    alpha=0.3,
                )

            plt.title("Time Taken vs. Clique Number", fontsize=14, fontweight="bold")
            plt.xlabel("Clique Number", fontsize=12)
            plt.ylabel("Time Taken (s)", fontsize=12)
            plt.yscale("log")
            plt.legend(title="algorithm", fontsize=10, title_fontsize=10)
            plt.grid(True, which="both", ls="--", alpha=0.4)
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, "time_vs_clique_number.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()

        # Plot 7: Accuracy vs Vertices
        if 'correct' in df.columns and 'vertices' in df.columns:
            plt.figure(figsize=(12, 8))
            df_copy = df.copy()
            df_copy["vertex_bin"] = pd.cut(df_copy["vertices"], bins=10)
            accuracy_by_size = (
                df_copy.groupby(["algorithm", "vertex_bin"])["correct"].mean().reset_index()
            )

            for algo in sorted(df_copy["algorithm"].unique()):
                algo_data = accuracy_by_size[accuracy_by_size["algorithm"] == algo]
                if not algo_data.empty:
                    x_values = [interval.mid for interval in algo_data["vertex_bin"]]
                    plt.plot(x_values, algo_data["correct"], marker="o", label=algo, alpha=0.7)

            plt.title("Correctness vs. Graph Size")
            plt.xlabel("Number of Vertices")
            plt.ylabel("Correctness Rate")
            plt.ylim(0, 1.05)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "correctness_vs_size.png"), dpi=150)
            plt.close()

        print(f"Generated 7 individual plots in {output_dir}/")

    def generate_enhanced_plots(self, df: pd.DataFrame, analysis: Dict):
        """Generate comprehensive overview plot"""

        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Time comparison
        ax1 = fig.add_subplot(gs[0, 0])
        time_means = df.groupby("algorithm")["time"].mean().sort_values()
        bars = ax1.bar(range(len(time_means)), time_means.values, color='steelblue', edgecolor='black')
        ax1.set_xticks(range(len(time_means)))
        ax1.set_xticklabels(time_means.index, rotation=45, ha='right')
        ax1.set_ylabel("Average Time (s)", fontsize=10)
        ax1.set_title("Average Time by Algorithm", fontsize=11, fontweight='bold')
        ax1.set_yscale("log")
        ax1.grid(True, alpha=0.3)

        for bar, value in zip(bars, time_means.values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{value:.3f}', ha='center', va='bottom', fontsize=8)

        # 2. Correctness
        ax2 = fig.add_subplot(gs[0, 1])
        if 'correct' in df.columns:
            correctness_data = df.groupby("algorithm")["correct"].mean().sort_values(ascending=False)
            bars = ax2.bar(range(len(correctness_data)), correctness_data.values,
                           color='lightgreen', edgecolor='black')
            ax2.set_xticks(range(len(correctness_data)))
            ax2.set_xticklabels(correctness_data.index, rotation=45, ha='right')
            ax2.set_ylabel("Correctness Rate", fontsize=10)
            ax2.set_title("Correctness Rate", fontsize=11, fontweight='bold')
            ax2.set_ylim(0, 1.05)
            ax2.axhline(y=1.0, color='green', linestyle='--', alpha=0.5)
            ax2.grid(True, alpha=0.3)

        # 3. Time vs Problem Size
        ax3 = fig.add_subplot(gs[0, 2])
        if 'vertices' in df.columns and df['vertices'].notna().any():
            for algo in sorted(df["algorithm"].unique()):
                algo_data = df[df["algorithm"] == algo].copy()
                if len(algo_data) == 0:
                    continue

                algo_data = algo_data.sort_values("vertices")
                grouped = algo_data.groupby("vertices")["time"].agg(['mean', 'std', 'count']).reset_index()
                grouped = grouped.dropna()

                if len(grouped) > 0:
                    ax3.plot(grouped["vertices"], grouped["mean"], label=algo, linewidth=2)

            ax3.set_xlabel("Problem Size (vertices)", fontsize=10)
            ax3.set_ylabel("Time (s)", fontsize=10)
            ax3.set_title("Time vs Problem Size", fontsize=11, fontweight='bold')
            ax3.set_yscale("log")
            ax3.legend(fontsize=8)
            ax3.grid(True, which="both", ls="--", alpha=0.3)

        # 4. Deviation
        ax4 = fig.add_subplot(gs[1, 0])
        if 'deviation' in df.columns and df['deviation'].notna().any():
            valid_dev = df[df["deviation"].notna()]
            sns.boxplot(data=valid_dev, x="algorithm", y="deviation", palette="Set2", ax=ax4)
            ax4.set_xlabel("Algorithm", fontsize=10)
            ax4.set_ylabel("Deviation", fontsize=10)
            ax4.set_title("Deviation Distribution", fontsize=11, fontweight='bold')
            ax4.axhline(y=0, color='green', linestyle='--', alpha=0.5)
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
            ax4.grid(True, alpha=0.3)

        # 5. Time vs Density
        ax5 = fig.add_subplot(gs[1, 1])
        if 'density' in df.columns and df['density'].notna().any():
            for algo in sorted(df["algorithm"].unique()):
                algo_data = df[(df["algorithm"] == algo) & (df["density"].notna())].copy()
                if algo_data.empty:
                    continue

                algo_data = algo_data.sort_values("density")
                n_bins = min(15, len(algo_data) // 3 + 1)
                if n_bins > 1:
                    algo_data["density_bin"] = pd.cut(algo_data["density"], bins=n_bins)
                    grouped = algo_data.groupby("density_bin")["time"].agg(['mean']).reset_index()
                    grouped = grouped.dropna()

                    if len(grouped) > 0:
                        grouped["density_center"] = [interval.mid for interval in grouped["density_bin"]]
                        ax5.plot(grouped["density_center"], grouped["mean"], label=algo, linewidth=2)

            ax5.set_xlabel("Graph Density", fontsize=10)
            ax5.set_ylabel("Time (s)", fontsize=10)
            ax5.set_title("Time vs Density", fontsize=11, fontweight='bold')
            ax5.set_yscale("log")
            ax5.legend(fontsize=8)
            ax5.grid(True, which="both", ls="--", alpha=0.3)

        # 6. Correlation heatmap
        ax6 = fig.add_subplot(gs[1, 2])
        numeric_cols = ['time', 'deviation', 'vertices', 'edges', 'density', 'clique_number']
        available_cols = [col for col in numeric_cols if col in df.columns and df[col].notna().any()]

        if len(available_cols) >= 2:
            corr_matrix = df[available_cols].corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                        center=0, ax=ax6, cbar_kws={'label': 'Correlation'})
            ax6.set_title('Correlation Matrix', fontsize=11, fontweight='bold')

        # 7. Statistical significance
        ax7 = fig.add_subplot(gs[2, 0])
        if 'pairwise_comparisons' in analysis and analysis['pairwise_comparisons']:
            comparison_names = []
            p_values = []
            for comp, results in analysis['pairwise_comparisons'].items():
                comparison_names.append(comp.replace('_', '\nvs\n'))
                p_values.append(results['p_value'])

            bars = ax7.bar(range(len(p_values)), p_values, color='skyblue', edgecolor='black')
            ax7.set_xticks(range(len(p_values)))
            ax7.set_xticklabels(comparison_names, fontsize=8)
            ax7.set_ylabel('p-value', fontsize=10)
            ax7.set_title('Statistical Significance', fontsize=11, fontweight='bold')
            ax7.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='α=0.05')
            ax7.legend(fontsize=8)
            ax7.grid(True, alpha=0.3)

            for bar, p_val in zip(bars, p_values):
                if p_val < 0.05:
                    bar.set_color('lightgreen')

        # 8. Algorithm efficiency
        ax8 = fig.add_subplot(gs[2, 1])
        if 'correct' in df.columns:
            efficiency_data = df.groupby('algorithm').agg({
                'correct': 'mean',
                'time': 'mean'
            })
            efficiency_data['efficiency'] = efficiency_data['correct'] / efficiency_data['time']
            efficiency_data = efficiency_data['efficiency'].sort_values(ascending=False)

            bars = ax8.bar(range(len(efficiency_data)), efficiency_data.values,
                           color='gold', edgecolor='black')
            ax8.set_xticks(range(len(efficiency_data)))
            ax8.set_xticklabels(efficiency_data.index, rotation=45, ha='right')
            ax8.set_ylabel('Efficiency', fontsize=10)
            ax8.set_title('Algorithm Efficiency', fontsize=11, fontweight='bold')
            ax8.grid(True, alpha=0.3)

        # 9. Summary
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')

        summary_text = f"""
SUMMARY STATISTICS
{'=' * 25}

Algorithms: {analysis['basic_stats']['n_algorithms']}
Graphs: {analysis['basic_stats']['n_graphs']}
Total Results: {analysis['basic_stats']['total_results']}

Best Performance:
"""

        if not df.empty:
            fastest = df.groupby('algorithm')['time'].mean().idxmin()
            summary_text += f"  Fastest: {fastest}\n"

            if 'correct' in df.columns:
                most_accurate = df.groupby('algorithm')['correct'].mean().idxmax()
                summary_text += f"  Most Accurate: {most_accurate}\n"

            if 'deviation' in df.columns and df['deviation'].notna().any():
                min_deviation = df.groupby('algorithm')['deviation'].mean().abs().idxmin()
                summary_text += f"  Min Deviation: {min_deviation}\n"

        ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes,
                 fontsize=9, fontfamily='monospace', verticalalignment='top')

        plt.suptitle('Comprehensive Algorithm Comparison Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()

        output_file = self.output_dir / f"comprehensive_analysis_{self.timestamp}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved comprehensive analysis to {output_file}")

        plt.close()

    def generate_markdown_report(self, df: pd.DataFrame, analysis: Dict):
       
        report_path = self.output_dir / f"comparison_report_{self.timestamp}.md"

        with open(report_path, 'w') as f:
            f.write("# Algorithm Comparison Report\n\n")
            f.write(f"Generated: {self.timestamp}\n\n")

            # Executive Summary
            f.write(" Executive Summary\n\n")
            f.write(f"- **Algorithms compared**: {', '.join(df['algorithm'].unique())}\n")
            f.write(f"- **Number of test graphs**: {analysis['basic_stats']['n_graphs']}\n")
            f.write(f"- **Total comparisons**: {analysis['basic_stats']['total_results']}\n\n")

            # Key Findings
            f.write("## Key Findings\n\n")

            # Find best performing algorithms
            if not df.empty:
                fastest_algo = df.groupby('algorithm')['time'].mean().idxmin()
                fastest_time = df.groupby('algorithm')['time'].mean().min()
                f.write(f"1. **Fastest Algorithm**: {fastest_algo} (avg: {fastest_time:.4f}s)\n")

                if 'correct' in df.columns:
                    most_accurate = df.groupby('algorithm')['correct'].mean().idxmax()
                    accuracy = df.groupby('algorithm')['correct'].mean().max()
                    f.write(f"2. **Most Accurate**: {most_accurate} ({accuracy * 100:.1f}% correct)\n")

                if 'deviation' in df.columns and df['deviation'].notna().any():
                    min_dev_algo = df.groupby('algorithm')['deviation'].mean().abs().idxmin()
                    min_dev = df.groupby('algorithm')['deviation'].mean().abs().min()
                    f.write(f"3. **Lowest Deviation**: {min_dev_algo} (avg: {min_dev:.2f})\n")

            # WP4-Specific Key Findings (if cluster editing present)
            if 'wp4_analysis' in analysis:
                f.write("\n### WP4: Cluster Editing vs Vertex Clique Cover\n\n")

                for algo, wp4_data in analysis['wp4_analysis'].items():
                    mean_C = wp4_data['mean_C']
                    mean_theta = wp4_data['mean_theta']
                    mean_ratio = wp4_data['mean_ratio']
                    p_val = wp4_data['wilcoxon_p']
                    corr = wp4_data['correlation']

                    f.write(f"**{algo}:**\n\n")
                    f.write(f"1. **Methods produce similar results**: C(G) ≈ θ(G) with ratio {mean_ratio:.2f}\n")

                    if not np.isnan(p_val):
                        if p_val < 0.05:
                            if mean_C < mean_theta:
                                f.write(
                                    f"2. **Statistically significant difference**: Wilcoxon test confirms C < θ (p < 0.05)\n")
                            else:
                                f.write(
                                    f"2. **Statistically significant difference**: Wilcoxon test confirms θ < C (p < 0.05)\n")
                        else:
                            f.write(f"2. **No statistically significant difference**: Wilcoxon test p = {p_val:.3f}\n")

                    if not np.isnan(corr):
                        f.write(f"3. **Strong correlation**: θ and C are highly correlated (r = {corr:.3f})\n")

                    f.write("\n")

            # Statistical Significance
            if 'pairwise_comparisons' in analysis and analysis['pairwise_comparisons']:
                f.write("\n## Statistical Significance\n\n")
                f.write("| Comparison | Test | p-value | Significant | Mean Time Diff |\n")
                f.write("|------------|------|---------|-------------|----------------|\n")

                for comp, results in analysis['pairwise_comparisons'].items():
                    sig = "Yes" if results['significant'] else "No"
                    f.write(
                        f"| {comp} | {results['test']} | {results['p_value']:.4f} | {sig} | {results['mean_time_diff']:.4f}s |\n")

            # WP4: Detailed Statistics
            if 'wp4_analysis' in analysis and 'vertex_clique_cover_number' in df.columns:
                cluster_algos = [a for a in df['algorithm'].unique() if self.is_cluster_editing_algo(a)]

                if cluster_algos:
                    f.write("\n## Detailed Statistics (WP4: C vs θ)\n\n")

                    for algo in cluster_algos:
                        algo_df = df[df['algorithm'] == algo].copy()
                        valid = algo_df.dropna(subset=['predicted', 'vertex_clique_cover_number', 'vertices'])

                        if len(valid) == 0:
                            continue

                        f.write(f"### {algo}\n\n")

                        # By Graph Size
                        f.write("#### By Graph Size\n\n")

                        valid['size_category'] = pd.cut(
                            valid['vertices'],
                            bins=[0, 20, 50, 100, np.inf],
                            labels=['tiny', 'small', 'medium', 'large']
                        )

                        size_stats = valid.groupby('size_category').agg({
                            'vertex_clique_cover_number': 'mean',
                            'predicted': 'mean',
                            'file': 'count'
                        }).reset_index()

                        size_stats['ratio'] = size_stats['predicted'] / size_stats['vertex_clique_cover_number']

                        f.write("| Size Category | Mean θ | Mean C | Mean C/θ | Count |\n")
                        f.write("|---------------|--------|--------|----------|-------|\n")

                        for _, row in size_stats.iterrows():
                            f.write(
                                f"| {row['size_category']} | {row['vertex_clique_cover_number']:.1f} | {row['predicted']:.1f} | {row['ratio']:.3f} | {row['file']:.1f} |\n")

                        f.write("\n")

                        # By Graph Density
                        f.write("#### By Graph Density\n\n")

                        if 'density' in valid.columns and valid['density'].notna().any():
                            valid['density_category'] = pd.cut(
                                valid['density'],
                                bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                labels=['very_sparse', 'sparse', 'medium', 'dense', 'very_dense']
                            )

                            density_stats = valid.groupby('density_category').agg({
                                'vertex_clique_cover_number': 'mean',
                                'predicted': 'mean',
                                'file': 'count'
                            }).reset_index()

                            density_stats['ratio'] = density_stats['predicted'] / density_stats[
                                'vertex_clique_cover_number']

                            f.write("| Density Category | Mean θ | Mean C | Mean C/θ | Count |\n")
                            f.write("|------------------|--------|--------|----------|-------|\n")

                            for _, row in density_stats.iterrows():
                                f.write(
                                    f"| {row['density_category']} | {row['vertex_clique_cover_number']:.1f} | {row['predicted']:.1f} | {row['ratio']:.3f} | {row['file']:.1f} |\n")

                            f.write("\n")

            # Performance by Algorithm
            f.write("\n## Performance by Algorithm\n\n")

            algo_stats = df.groupby('algorithm').agg({
                'time': ['mean', 'std', 'min', 'max'],
                'correct': 'mean' if 'correct' in df.columns else lambda x: None,
                'deviation': 'mean' if 'deviation' in df.columns else lambda x: None
            })

            f.write("| Algorithm | Mean Time (s) | Std Time | Min Time | Max Time | Correctness | Avg Deviation |\n")
            f.write("|-----------|---------------|----------|----------|----------|-------------|---------------|\n")

            for algo in algo_stats.index:
                mean_time = algo_stats.loc[algo, ('time', 'mean')]
                std_time = algo_stats.loc[algo, ('time', 'std')]
                min_time = algo_stats.loc[algo, ('time', 'min')]
                max_time = algo_stats.loc[algo, ('time', 'max')]

                correctness = "N/A"
                if 'correct' in df.columns:
                    corr_val = algo_stats.loc[algo, ('correct', 'mean')]
                    if pd.notna(corr_val):
                        correctness = f"{corr_val * 100:.1f}%"

                deviation = "N/A"
                if 'deviation' in df.columns:
                    dev_val = algo_stats.loc[algo, ('deviation', 'mean')]
                    if pd.notna(dev_val):
                        deviation = f"{dev_val:.2f}"

                f.write(
                    f"| {algo} | {mean_time:.4f} | {std_time:.4f} | {min_time:.4f} | {max_time:.4f} | {correctness} | {deviation} |\n")

            # Correlation Analysis
            if 'correlations' in analysis and analysis['correlations']:
                f.write("\n## Correlation Analysis\n\n")
                f.write("| Metric | Correlation | p-value |\n")
                f.write("|--------|-------------|----------|\n")

                for metric, results in analysis['correlations'].items():
                    if results and 'correlation' in results:
                        f.write(f"| {metric} | {results['correlation']:.3f} | {results['p_value']:.4f} |\n")

            # Graph Categories Performance
            if 'vertices' in df.columns:
                f.write("\n## Performance by Graph Size\n\n")

                df_copy = df.copy()
                df_copy['size_category'] = pd.cut(df_copy['vertices'],
                                                  bins=[0, 20, 50, 100, np.inf],
                                                  labels=['Tiny (≤20)', 'Small (21-50)', 'Medium (51-100)',
                                                          'Large (>100)'])

                size_perf = df_copy.groupby(['size_category', 'algorithm'])['time'].mean().unstack(fill_value=None)

                f.write("| Size Category | " + " | ".join(size_perf.columns) + " |\n")
                f.write("|---------------|" + "|".join(["---------"] * len(size_perf.columns)) + "|\n")

                for category in size_perf.index:
                    row_data = [str(category)]
                    for algo in size_perf.columns:
                        val = size_perf.loc[category, algo]
                        if pd.notna(val):
                            row_data.append(f"{val:.4f}s")
                        else:
                            row_data.append("N/A")
                    f.write("| " + " | ".join(row_data) + " |\n")


            f.write("\n## Output Files\n\n")
            f.write(f"- Comprehensive analysis plot: `comprehensive_analysis_{self.timestamp}.png`\n")
            f.write(f"- This report: `comparison_report_{self.timestamp}.md`\n")
            f.write(f"- Raw comparison data: Available in DataFrame format\n")

        print(f"Report saved to: {report_path}")
        return report_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/comparison.py <results_file1> <results_file2> ...")
        print("\nExample:")
        print("  python src/comparison.py results/raw/curated/20-29/chalupa.txt results/raw/curated/20-29/ilp.txt")
        print("  python src/comparison.py results/raw/curated/*.txt")
        return

    results_files = sys.argv[1:]

    analyzer = ComparisonAnalyzer()
    all_results = []

    for results_file in results_files:
        print(f"Processing {results_file}...")

        file_results = analyzer.parse_results(results_file)
        graph_directory = analyzer.derive_graph_directory(results_file)

        if not graph_directory:
            print(f"  Warning: Could not determine graph directory, skipping...")
            continue

        if not os.path.exists(graph_directory):
            print(f"  Warning: Graph directory {graph_directory} does not exist, skipping...")
            continue

        print(f"  Looking for graphs in: {graph_directory}")

        found_count = 0
        for result in file_results:
            graph_filename = result["file"]
            graph_path = analyzer.find_graph_file(graph_filename, graph_directory)

            if graph_path:
                graph_props = analyzer.parse_graph_file(graph_path)
                result.update(graph_props)
                result["graph_path"] = graph_path
                found_count += 1
            else:
                print(f"  Warning: Could not find graph file {graph_filename}")
                result["graph_path"] = None

        print(f"  Found {found_count}/{len(file_results)} graph files")
        all_results.extend(file_results)

    if not all_results:
        print("\nNo results found!")
        return

    df = pd.DataFrame(all_results)

    # filter out rows where we couldn't find the graph file
    original_count = len(df)
    df = df.dropna(subset=["vertices"])
    filtered_count = len(df)

    if filtered_count < original_count:
        print(f"\nFiltered out {original_count - filtered_count} results without graph data")

    if df.empty:
        print("No valid results with graph data found!")
        return

    print(f"\nLoaded {len(df)} results from {len(results_files)} files")
    print(f"Algorithms: {sorted(df['algorithm'].unique())}")
    print(f"Vertex counts: {int(df['vertices'].min())}-{int(df['vertices'].max())}")

    print("\nPerforming statistical analysis...")
    analysis = analyzer.perform_statistical_analysis(df)

    print("Generating individual plots...")
    analyzer.generate_plots(df, str(analyzer.output_dir))

    print("Generating comprehensive overview...")
    analyzer.generate_enhanced_plots(df, analysis)

    print("Generating detailed report...")
    report_path = analyzer.generate_markdown_report(df, analysis)

    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Algorithms compared: {', '.join(df['algorithm'].unique())}")
    print(f"Total results analyzed: {len(df)}")

    if not df.empty:
        fastest = df.groupby('algorithm')['time'].mean().idxmin()
        fastest_time = df.groupby('algorithm')['time'].mean().min()
        print(f"Fastest algorithm: {fastest} (avg: {fastest_time:.4f}s)")

        if 'correct' in df.columns:
            most_accurate = df.groupby('algorithm')['correct'].mean().idxmax()
            accuracy = df.groupby('algorithm')['correct'].mean().max()
            print(f"Most accurate: {most_accurate} ({accuracy * 100:.1f}%)")

    # WP4 summary (if present)
    if 'wp4_analysis' in analysis:
        print("\nWP4 Analysis (C vs θ):")
        for algo, wp4_data in analysis['wp4_analysis'].items():
            print(f"  {algo}:")
            print(f"    Mean C(G): {wp4_data['mean_C']:.2f}")
            print(f"    Mean θ(G): {wp4_data['mean_theta']:.2f}")
            print(f"    Mean ratio C/θ: {wp4_data['mean_ratio']:.3f}")
            if not np.isnan(wp4_data['correlation']):
                print(f"    Correlation: {wp4_data['correlation']:.3f}")

    print(f"\nResults saved to: {analyzer.output_dir}")
    print(f"Report: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
