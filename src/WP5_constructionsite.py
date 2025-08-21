"""
WP5: Real Data Analysis on Rfam RNA Families

Applies the complete pipeline (WP0-4) to real RNA data with shift-alignment predictions.

Functions:
    1 data processing
        load tsv data with RNA alignments
        convert similarity-scores in graphs
        two use methods: threshold-based and KNN-based

    2 integration with WP1-4:
        uses VCC-solver from WP1/2 (Chalupa + ILP)
        uses CE-solver from WP3 (with kernelization)
        uses comparison frameworks from WP4

    3 biological analysis:
        Shift-Correlation: analysis, whether Shift-Events occur at Cluster-borders
        Conservation Analysis: misst similarities within clusters
        Interpretation: biological results

    4 visualizations
        6-Panel-visualization for each RNA-family
        Original Graph with Shift-Highlighting
        VCC and CE Clustering-results
        statistical overview

    5 reporting
        detailed markdown report for each family
        overview report for Batch-Processing
        CSV-Export of the Metriken
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from dataclasses import dataclass, field

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import components from other WPs
try:
    from wrapperV2 import (
        interactive_reduced_ilp_wrapper,
        reduced_ilp_wrapper,
        ilp_wrapper
    )
    from algorithms.chalupa import ChalupaHeuristic
    from algorithms.ilp_solver import solve_ilp_clique_cover
    from algorithms.cluster_editing_solver import ClusterEditingSolver
    from algorithms.cluster_editing_kernelization import ClusterEditingInstance
    from WP4_comparison_VCC_CE import (
        ComparisonFramework,
        SolverAdapter,
        ClusteringResult
    )
    from utils import txt_to_networkx
except ImportError as e:
    print(f"Warning: Some imports failed: {e}")
    print("Make sure all WP modules are available")


@dataclass
class RNAClusteringResult:
    """Extended clustering result for RNA data."""
    clusters: List[Set[str]]  # RNA IDs in clusters
    num_clusters: int
    method: str
    shift_correlation: Dict
    runtime: float
    metadata: Dict = field(default_factory=dict)


class WP5RfamAnalysis:
    """
    Main class for WP5: Analysis of Rfam RNA families with shift events.
    """

    def __init__(self, output_dir: str = "results/wp5"):
        """Initialize WP5 analyzer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = self.output_dir / "processed_graphs"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_rfam_data(self, tsv_path: str) -> nx.Graph:
        """
        Load RNA similarity data from TSV into NetworkX graph.

        Args:
            tsv_path: Path to TSV file with RNA pairwise alignments

        Returns:
            NetworkX graph with RNA sequences as nodes
        """
        print(f"Loading RNA data from {tsv_path}...")
        df = pd.read_csv(tsv_path, sep="\t")

        # Create weighted graph
        G = nx.Graph()

        # Get unique RNA IDs
        rna_ids = set(df['idA'].unique()) | set(df['idB'].unique())
        G.add_nodes_from(rna_ids)

        # Add edges with attributes
        edges_with_shifts = 0
        for _, row in df.iterrows():
            # Skip self-loops for clustering
            if row["idA"] != row["idB"]:
                # Calculate normalized similarity (0-1 scale, 1 = most similar)
                # Using exponential transformation for better separation
                max_score = df['score'].max()
                min_score = df['score'].min()
                normalized_score = (row["score"] - min_score) / (max_score - min_score) if max_score != min_score else 0
                similarity = np.exp(-2 * normalized_score)  # Exponential decay

                G.add_edge(
                    row["idA"],
                    row["idB"],
                    weight=row["score"],  # Original score
                    similarity=similarity,  # Normalized similarity [0,1]
                    shifts=row["shifts"],
                    has_shift=row["shifts"] > 0
                )

                if row["shifts"] > 0:
                    edges_with_shifts += 1

        print(f"  Loaded {G.number_of_nodes()} RNAs with {G.number_of_edges()} pairwise alignments")
        print(f"  Shift events detected in {edges_with_shifts} pairs ({edges_with_shifts/G.number_of_edges()*100:.1f}%)")

        return G

    def prepare_for_clustering(self, G: nx.Graph, method: str = 'threshold') -> nx.Graph:
        """
        Convert similarity graph to format expected by VCC/CE algorithms.

        Args:
            G: Original graph with similarity scores
            method: 'threshold' or 'knn' for edge selection

        Returns:
            Binary graph suitable for clustering
        """
        if method == 'threshold':
            return self._threshold_based_graph(G)
        elif method == 'knn':
            return self._knn_based_graph(G)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _threshold_based_graph(self, G: nx.Graph, percentile: float = 30) -> nx.Graph:
        """Create binary graph by thresholding similarity scores."""
        # Analyze score distribution
        scores = [e[2]['weight'] for e in G.edges(data=True)]
        threshold = np.percentile(scores, percentile)  # Keep top (100-percentile)% similar pairs

        # Create binary graph for clustering
        G_binary = nx.Graph()
        G_binary.add_nodes_from(G.nodes())

        edges_kept = 0
        for u, v, data in G.edges(data=True):
            if data['weight'] <= threshold:  # Lower score = more similar
                G_binary.add_edge(u, v)
                edges_kept += 1

        print(f"  Threshold method: kept {edges_kept}/{G.number_of_edges()} edges (threshold={threshold:.0f})")
        return G_binary

    def _knn_based_graph(self, G: nx.Graph, k: int = 3) -> nx.Graph:
        """Create binary graph by keeping k nearest neighbors for each node."""
        G_binary = nx.Graph()
        G_binary.add_nodes_from(G.nodes())

        for node in G.nodes():
            # Get all neighbors with scores
            neighbors = []
            for neighbor in G.neighbors(node):
                score = G[node][neighbor]['weight']
                neighbors.append((neighbor, score))

            # Sort by score (lower is better) and keep top k
            neighbors.sort(key=lambda x: x[1])
            for neighbor, _ in neighbors[:k]:
                G_binary.add_edge(node, neighbor)

        print(f"  KNN method: created graph with {G_binary.number_of_edges()} edges (k={k})")
        return G_binary

    def save_as_txt(self, G: nx.Graph, filepath: str) -> None:
        """Save graph in WP0-compatible text format."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Create node mapping to integers
        node_to_id = {node: i for i, node in enumerate(G.nodes())}

        with open(filepath, 'w') as f:
            # Write header
            f.write(f"# RNA Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")
            f.write(f"# Node mapping:\n")
            for node, id in node_to_id.items():
                f.write(f"# {id}: {node}\n")
            f.write("\n")

            # Write edges
            for u, v in G.edges():
                f.write(f"{node_to_id[u]} {node_to_id[v]}\n")

        print(f"  Saved graph to {filepath}")

    def apply_vcc(self, G: nx.Graph, method: str = 'heuristic') -> RNAClusteringResult:
        """
        Apply Vertex Clique Cover using WP1/2 methods.

        Args:
            G: Binary graph
            method: 'exact', 'heuristic', or 'reduced'
        """
        start_time = time.time()

        if method == 'exact':
            # Use ILP solver
            result = solve_ilp_clique_cover(G, time_limit=300, return_assignment=True)
            clusters = self._extract_clusters_from_coloring(G, result.get('assignment', {}))
            theta = result.get('chromatic_number', len(clusters))

        elif method == 'heuristic':
            # Use Chalupa heuristic
            complement = nx.complement(G)
            heuristic = ChalupaHeuristic(complement)
            coloring = heuristic.iterated_greedy_clique_covering(iterations=1000)
            clusters = self._extract_clusters_from_coloring(G, coloring)
            theta = len(clusters)

        elif method == 'reduced':
            # Use kernelization + ILP
            # Save temp file for wrapper
            temp_file = self.data_dir / "temp_graph.txt"
            self.save_as_txt(G, str(temp_file))
            result = interactive_reduced_ilp_wrapper(str(temp_file))
            if isinstance(result, dict):
                clusters = self._extract_clusters_from_coloring(G, result.get('assignment', {}))
                theta = result.get('chromatic_number', len(clusters))
            else:
                # Handle case where wrapper returns non-dict (e.g., just the number)
                clusters = []
                theta = result if result else 0
        else:
            raise ValueError(f"Unknown VCC method: {method}")

        runtime = time.time() - start_time

        return RNAClusteringResult(
            clusters=clusters,
            num_clusters=theta,
            method=f"vcc_{method}",
            shift_correlation={},  # Will be filled later
            runtime=runtime,
            metadata={'algorithm': method}
        )

    def apply_ce(self, G: nx.Graph, use_kernelization: bool = True) -> RNAClusteringResult:
        """
        Apply Cluster Editing using WP3 methods.

        Args:
            G: Binary graph
            use_kernelization: Whether to use kernelization
        """
        start_time = time.time()

        solver = ClusterEditingSolver(G)
        result = solver.solve(
            use_kernelization=use_kernelization,
            kernelization_type='optimized' if use_kernelization else None,
            clustering_algorithm='greedy_improved'
        )

        clusters = [set(cluster) for cluster in result.get('clusters', [])]
        runtime = time.time() - start_time

        return RNAClusteringResult(
            clusters=clusters,
            num_clusters=len(clusters),
            method=f"ce_{'kernelized' if use_kernelization else 'basic'}",
            shift_correlation={},  # Will be filled later
            runtime=runtime,
            metadata={
                'editing_cost': result.get('cost', 0),
                'kernelization': use_kernelization
            }
        )

    def _extract_clusters_from_coloring(self, G: nx.Graph, coloring: Dict) -> List[Set]:
        """Convert node coloring to list of clusters."""
        if not coloring:
            return []

        clusters_dict = defaultdict(set)

        # Handle different coloring formats
        if isinstance(coloring, dict):
            for node, color in coloring.items():
                clusters_dict[color].add(node)

        return list(clusters_dict.values())

    def analyze_shift_correlation(self, G_full: nx.Graph, clusters: List[Set]) -> Dict:
        """
        Analyze how shift events correlate with cluster boundaries.

        Key metrics:
        - Inter-cluster shift ratio: fraction of shifts between different clusters
        - Shift density: average shifts per edge within/between clusters
        """
        inter_cluster_shifts = 0
        intra_cluster_shifts = 0
        inter_cluster_edges = 0
        intra_cluster_edges = 0

        # Map nodes to clusters
        node_to_cluster = {}
        for i, cluster in enumerate(clusters):
            for node in cluster:
                node_to_cluster[node] = i

        # Analyze each edge
        for u, v, data in G_full.edges(data=True):
            if u in node_to_cluster and v in node_to_cluster:
                same_cluster = node_to_cluster[u] == node_to_cluster[v]
                has_shift = data.get('shifts', 0) > 0

                if same_cluster:
                    intra_cluster_edges += 1
                    if has_shift:
                        intra_cluster_shifts += 1
                else:
                    inter_cluster_edges += 1
                    if has_shift:
                        inter_cluster_shifts += 1

        total_shifts = inter_cluster_shifts + intra_cluster_shifts

        return {
            'total_shifts': total_shifts,
            'inter_cluster_shifts': inter_cluster_shifts,
            'intra_cluster_shifts': intra_cluster_shifts,
            'inter_cluster_ratio': inter_cluster_shifts/total_shifts if total_shifts > 0 else 0,
            'inter_cluster_edges': inter_cluster_edges,
            'intra_cluster_edges': intra_cluster_edges,
            'shift_density_inter': inter_cluster_shifts/inter_cluster_edges if inter_cluster_edges > 0 else 0,
            'shift_density_intra': intra_cluster_shifts/intra_cluster_edges if intra_cluster_edges > 0 else 0
        }

    def analyze_cluster_conservation(self, G_full: nx.Graph, clusters: List[Set]) -> Dict:
        """
        Analyze conservation within clusters based on similarity scores.
        """
        conservation_scores = []

        for cluster in clusters:
            if len(cluster) < 2:
                continue

            # Calculate average similarity within cluster
            similarities = []
            cluster_list = list(cluster)
            for i in range(len(cluster_list)):
                for j in range(i+1, len(cluster_list)):
                    if G_full.has_edge(cluster_list[i], cluster_list[j]):
                        sim = G_full[cluster_list[i]][cluster_list[j]].get('similarity', 0)
                        similarities.append(sim)

            if similarities:
                conservation_scores.append(np.mean(similarities))

        return {
            'mean_conservation': np.mean(conservation_scores) if conservation_scores else 0,
            'std_conservation': np.std(conservation_scores) if conservation_scores else 0,
            'min_conservation': np.min(conservation_scores) if conservation_scores else 0,
            'max_conservation': np.max(conservation_scores) if conservation_scores else 0
        }

    def compare_methods(self, G_binary: nx.Graph, vcc_result: RNAClusteringResult,
                       ce_result: RNAClusteringResult) -> Dict:
        """
        Compare VCC and CE solutions using WP4 framework.
        """
        comparison = {
            'theta': vcc_result.num_clusters,
            'C': ce_result.num_clusters,
            'ratio': ce_result.num_clusters / vcc_result.num_clusters if vcc_result.num_clusters > 0 else float('inf'),
            'vcc_runtime': vcc_result.runtime,
            'ce_runtime': ce_result.runtime,
            'speedup': vcc_result.runtime / ce_result.runtime if ce_result.runtime > 0 else float('inf')
        }

        # Check mathematical invariant
        if comparison['theta'] > comparison['C']:
            print(f"  WARNING: Invariant violated! θ(G)={comparison['theta']} > C(G)={comparison['C']}")

        # Clustering agreement metrics
        vcc_labels = self._clusters_to_labels(vcc_result.clusters, G_binary.nodes())
        ce_labels = self._clusters_to_labels(ce_result.clusters, G_binary.nodes())

        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        comparison['adjusted_rand_index'] = adjusted_rand_score(vcc_labels, ce_labels)
        comparison['nmi_score'] = normalized_mutual_info_score(vcc_labels, ce_labels)

        return comparison

    def _clusters_to_labels(self, clusters: List[Set], nodes: List) -> np.ndarray:
        """Convert cluster assignment to label array."""
        label_dict = {}
        for i, cluster in enumerate(clusters):
            for node in cluster:
                label_dict[node] = i
        return np.array([label_dict.get(node, -1) for node in nodes])

    def visualize_results(self, G_full: nx.Graph, G_binary: nx.Graph,
                         vcc_result: RNAClusteringResult, ce_result: RNAClusteringResult,
                         family_name: str):
        """
        Create comprehensive visualizations of clustering results.
        """
        fig = plt.figure(figsize=(20, 12))

        # Use spring layout for consistency
        pos = nx.spring_layout(G_binary, k=2, iterations=50, seed=42)

        # 1. Original graph with shift highlighting
        ax1 = plt.subplot(2, 3, 1)
        self._plot_graph_with_shifts(G_full, pos, ax1, "Original RNA Similarity Network")

        # 2. Binary graph for clustering
        ax2 = plt.subplot(2, 3, 2)
        nx.draw_networkx_nodes(G_binary, pos, node_color='lightblue', node_size=500, ax=ax2)
        nx.draw_networkx_edges(G_binary, pos, edge_color='gray', alpha=0.5, ax=ax2)
        nx.draw_networkx_labels(G_binary, pos, font_size=8, ax=ax2)
        ax2.set_title("Binary Graph for Clustering")
        ax2.axis('off')

        # 3. VCC clustering
        ax3 = plt.subplot(2, 3, 3)
        self._plot_clustering(G_binary, pos, vcc_result.clusters, ax3,
                            f"VCC Clustering (θ={vcc_result.num_clusters})")

        # 4. CE clustering
        ax4 = plt.subplot(2, 3, 4)
        self._plot_clustering(G_binary, pos, ce_result.clusters, ax4,
                            f"CE Clustering (C={ce_result.num_clusters})")

        # 5. Shift correlation heatmap
        ax5 = plt.subplot(2, 3, 5)
        self._plot_shift_correlation(vcc_result, ce_result, ax5)

        # 6. Summary statistics
        ax6 = plt.subplot(2, 3, 6)
        self._plot_summary_stats(vcc_result, ce_result, ax6)

        plt.suptitle(f"RNA Family {family_name} - Clustering Analysis", fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_path = self.figures_dir / f"{family_name}_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved visualization to {output_path}")

    def _plot_graph_with_shifts(self, G: nx.Graph, pos: Dict, ax, title: str):
        """Plot graph with shift events highlighted."""
        # Separate edges by shift status
        shift_edges = [(u,v) for u,v,d in G.edges(data=True) if d.get('has_shift', False)]
        normal_edges = [(u,v) for u,v,d in G.edges(data=True) if not d.get('has_shift', False)]

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500, ax=ax)

        # Draw edges
        nx.draw_networkx_edges(G, pos, edgelist=normal_edges, edge_color='gray',
                              alpha=0.3, width=1, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=shift_edges, edge_color='red',
                              width=2, alpha=0.8, ax=ax)

        # Labels
        labels = {node: node.split('/')[0].split('.')[-1][:8] for node in G.nodes()}  # Shortened labels
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)

        ax.set_title(title)
        ax.axis('off')

        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='gray', alpha=0.3, label='Normal alignment'),
            Line2D([0], [0], color='red', linewidth=2, label='Shift event')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

    def _plot_clustering(self, G: nx.Graph, pos: Dict, clusters: List[Set], ax, title: str):
        """Plot graph with cluster coloring."""
        # Create color map for clusters
        node_colors = {}
        for i, cluster in enumerate(clusters):
            for node in cluster:
                node_colors[node] = i

        # Use a colormap
        colors = [node_colors.get(node, -1) for node in G.nodes()]

        # Draw graph
        nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=500,
                              cmap='tab20', vmin=0, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.3, ax=ax)

        # Shortened labels
        labels = {node: node.split('/')[0].split('.')[-1][:8] for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)

        ax.set_title(title)
        ax.axis('off')

    def _plot_shift_correlation(self, vcc_result: RNAClusteringResult,
                               ce_result: RNAClusteringResult, ax):
        """Plot shift correlation comparison."""
        methods = ['VCC', 'CE']
        inter_ratios = [
            vcc_result.shift_correlation.get('inter_cluster_ratio', 0),
            ce_result.shift_correlation.get('inter_cluster_ratio', 0)
        ]

        bars = ax.bar(methods, inter_ratios, color=['steelblue', 'coral'])
        ax.set_ylabel('Inter-cluster Shift Ratio')
        ax.set_title('Shift Events vs Cluster Boundaries')
        ax.set_ylim([0, 1])

        # Add value labels on bars
        for bar, ratio in zip(bars, inter_ratios):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{ratio:.2f}', ha='center', va='bottom')

        # Add horizontal line at 0.5
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.text(0.5, 0.52, 'Random expectation', ha='center', fontsize=9, alpha=0.7)

    def _plot_summary_stats(self, vcc_result: RNAClusteringResult,
                           ce_result: RNAClusteringResult, ax):
        """Plot summary statistics table."""
        ax.axis('tight')
        ax.axis('off')

        # Create summary data
        data = [
            ['Metric', 'VCC', 'CE'],
            ['Number of clusters', f'{vcc_result.num_clusters}', f'{ce_result.num_clusters}'],
            ['Runtime (s)', f'{vcc_result.runtime:.2f}', f'{ce_result.runtime:.2f}'],
            ['Inter-cluster shifts',
             f'{vcc_result.shift_correlation.get("inter_cluster_shifts", 0)}',
             f'{ce_result.shift_correlation.get("inter_cluster_shifts", 0)}'],
            ['Intra-cluster shifts',
             f'{vcc_result.shift_correlation.get("intra_cluster_shifts", 0)}',
             f'{ce_result.shift_correlation.get("intra_cluster_shifts", 0)}'],
            ['Shift correlation',
             f'{vcc_result.shift_correlation.get("inter_cluster_ratio", 0):.2f}',
             f'{ce_result.shift_correlation.get("inter_cluster_ratio", 0):.2f}']
        ]

        table = ax.table(cellText=data, cellLoc='center', loc='center',
                        colWidths=[0.4, 0.3, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        # Style header row
        for i in range(3):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        ax.set_title('Summary Statistics', fontweight='bold', pad=20)

    def generate_report(self, family_name: str, G_full: nx.Graph,
                       vcc_result: RNAClusteringResult, ce_result: RNAClusteringResult,
                       comparison: Dict, bio_analysis: Dict):
        """
        Generate comprehensive markdown report for RNA family analysis.
        """
        report_path = self.output_dir / f"{family_name}_report.md"

        with open(report_path, 'w') as f:
            f.write(f"# WP5 Analysis Report: {family_name}\n\n")
            f.write(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Data summary
            f.write("## Data Summary\n\n")
            f.write(f"- **RNA sequences:** {G_full.number_of_nodes()}\n")
            f.write(f"- **Pairwise alignments:** {G_full.number_of_edges()}\n")
            shift_edges = sum(1 for _, _, d in G_full.edges(data=True) if d.get('has_shift', False))
            f.write(f"- **Alignments with shifts:** {shift_edges} ({shift_edges/G_full.number_of_edges()*100:.1f}%)\n\n")

            # Clustering results
            f.write("## Clustering Results\n\n")
            f.write("### Vertex Clique Cover (VCC)\n")
            f.write(f"- **Number of clusters (θ):** {vcc_result.num_clusters}\n")
            f.write(f"- **Runtime:** {vcc_result.runtime:.3f}s\n")
            f.write(f"- **Method:** {vcc_result.method}\n\n")

            f.write("### Cluster Editing (CE)\n")
            f.write(f"- **Number of clusters (C):** {ce_result.num_clusters}\n")
            f.write(f"- **Runtime:** {ce_result.runtime:.3f}s\n")
            f.write(f"- **Method:** {ce_result.method}\n\n")

            # Comparison
            f.write("## Method Comparison\n\n")
            f.write(f"- **C/θ ratio:** {comparison['ratio']:.3f}\n")
            f.write(f"- **Adjusted Rand Index:** {comparison['adjusted_rand_index']:.3f}\n")
            f.write(f"- **NMI Score:** {comparison['nmi_score']:.3f}\n")
            f.write(f"- **Runtime speedup (VCC/CE):** {comparison['speedup']:.2f}x\n\n")

            # Biological insights
            f.write("## Biological Insights\n\n")
            f.write("### Shift Event Analysis\n\n")

            # VCC shift analysis
            vcc_shift = vcc_result.shift_correlation
            f.write(f"**VCC Clustering:**\n")
            f.write(f"- Inter-cluster shifts: {vcc_shift.get('inter_cluster_shifts', 0)}\n")
            f.write(f"- Intra-cluster shifts: {vcc_shift.get('intra_cluster_shifts', 0)}\n")
            f.write(f"- Inter-cluster ratio: {vcc_shift.get('inter_cluster_ratio', 0):.3f}\n\n")

            # CE shift analysis
            ce_shift = ce_result.shift_correlation
            f.write(f"**CE Clustering:**\n")
            f.write(f"- Inter-cluster shifts: {ce_shift.get('inter_cluster_shifts', 0)}\n")
            f.write(f"- Intra-cluster shifts: {ce_shift.get('intra_cluster_shifts', 0)}\n")
            f.write(f"- Inter-cluster ratio: {ce_shift.get('inter_cluster_ratio', 0):.3f}\n\n")

            # Conservation analysis
            f.write("### Cluster Conservation\n\n")
            vcc_cons = bio_analysis.get('vcc_conservation', {})
            ce_cons = bio_analysis.get('ce_conservation', {})

            f.write(f"**VCC Clusters:**\n")
            f.write(f"- Mean conservation: {vcc_cons.get('mean_conservation', 0):.3f}\n")
            f.write(f"- Std deviation: {vcc_cons.get('std_conservation', 0):.3f}\n\n")

            f.write(f"**CE Clusters:**\n")
            f.write(f"- Mean conservation: {ce_cons.get('mean_conservation', 0):.3f}\n")
            f.write(f"- Std deviation: {ce_cons.get('std_conservation', 0):.3f}\n\n")

            # Interpretation
            f.write("## Interpretation\n\n")

            # Shift correlation interpretation
            if vcc_shift.get('inter_cluster_ratio', 0) > 0.7 or ce_shift.get('inter_cluster_ratio', 0) > 0.7:
                f.write(" **Strong correlation between shift events and cluster boundaries detected!**\n")
                f.write("  - This suggests that shift events occur primarily between evolutionarily distinct RNA modules.\n")
                f.write("  - The clustering successfully identifies RNA subfamilies with independent evolution.\n\n")
            elif vcc_shift.get('inter_cluster_ratio', 0) > 0.5 or ce_shift.get('inter_cluster_ratio', 0) > 0.5:
                f.write(" **Moderate correlation between shifts and clusters.**\n")
                f.write("  - Some association between evolutionary modules and shift events.\n")
                f.write("  - Further analysis may be needed to refine clustering.\n\n")
            else:
                f.write(" **Weak correlation between shifts and clusters.**\n")
                f.write("  - Shift events appear randomly distributed.\n")
                f.write("  - Alternative clustering approaches may be needed.\n\n")

            # Method comparison interpretation
            if comparison['ratio'] < 1.1:
                f.write("- **VCC and CE produce very similar clusterings** (C/θ ≈ 1)\n")
                f.write("  - Strong agreement suggests robust clustering structure.\n")
            elif comparison['ratio'] < 1.5:
                f.write("- **CE requires moderately more clusters than VCC**\n")
                f.write("  - Enforcing disjoint clusters leads to finer partitioning.\n")
            else:
                f.write("- **Significant difference between VCC and CE**\n")
                f.write("  - RNA relationships may not fit well with disjoint clustering.\n")

            f.write("\n---\n")
            f.write("*Report generated by WP5 RNA Analysis Pipeline*\n")

        print(f"  Saved report to {report_path}")

    def run_complete_pipeline(self, tsv_path: str, family_name: str = None):
        """
        Execute complete WP5 pipeline on RNA data.

        Args:
            tsv_path: Path to TSV file with RNA alignments
            family_name: Name of RNA family (extracted from filename if not provided)

        Returns:
            Dictionary with all results
        """
        # Extract family name from filename if not provided
        if family_name is None:
            family_name = Path(tsv_path).stem

        print(f"\n{'='*80}")
        print(f"WP5: Analyzing Rfam family {family_name}")
        print(f"{'='*80}\n")

        # 1. Load data
        print("1. Loading RNA similarity data...")
        G_full = self.load_rfam_data(tsv_path)

        # 2. Prepare for clustering
        print("\n2. Preparing graph for clustering algorithms...")
        G_binary = self.prepare_for_clustering(G_full, method='threshold')

        # Alternative: try KNN approach
        G_binary_knn = self.prepare_for_clustering(G_full, method='knn')

        # 3. Save in WP0 format
        self.save_as_txt(G_binary, self.data_dir / f"{family_name}.txt")

        # 4. Apply VCC
        print("\n3. Applying Vertex Clique Cover...")
        vcc_result = self.apply_vcc(G_binary, method='heuristic')
        print(f"  VCC found {vcc_result.num_clusters} clusters in {vcc_result.runtime:.3f}s")

        # 5. Apply CE
        print("\n4. Applying Cluster Editing...")
        ce_result = self.apply_ce(G_binary, use_kernelization=True)
        print(f"  CE found {ce_result.num_clusters} clusters in {ce_result.runtime:.3f}s")

        # 6. Analyze shift correlations
        print("\n5. Analyzing shift event correlations...")
        vcc_result.shift_correlation = self.analyze_shift_correlation(G_full, vcc_result.clusters)
        ce_result.shift_correlation = self.analyze_shift_correlation(G_full, ce_result.clusters)

        print(f"  VCC: {vcc_result.shift_correlation['inter_cluster_ratio']:.2%} shifts are inter-cluster")
        print(f"  CE:  {ce_result.shift_correlation['inter_cluster_ratio']:.2%} shifts are inter-cluster")

        # 7. Compare methods
        print("\n6. Comparing VCC and CE solutions...")
        comparison = self.compare_methods(G_binary, vcc_result, ce_result)
        print(f"  C/θ ratio: {comparison['ratio']:.3f}")
        print(f"  Agreement (ARI): {comparison['adjusted_rand_index']:.3f}")

        # 8. Biological analysis
        print("\n7. Performing biological analysis...")
        bio_analysis = {
            'vcc_conservation': self.analyze_cluster_conservation(G_full, vcc_result.clusters),
            'ce_conservation': self.analyze_cluster_conservation(G_full, ce_result.clusters)
        }

        # 9. Generate visualizations
        print("\n8. Creating visualizations...")
        self.visualize_results(G_full, G_binary, vcc_result, ce_result, family_name)

        # 10. Generate report
        print("\n9. Generating report...")
        self.generate_report(family_name, G_full, vcc_result, ce_result, comparison, bio_analysis)

        print(f"\n{'='*80}")
        print(f"Analysis complete for {family_name}!")
        print(f"{'='*80}\n")

        return {
            'graph_full': G_full,
            'graph_binary': G_binary,
            'vcc': vcc_result,
            'ce': ce_result,
            'comparison': comparison,
            'biological': bio_analysis
        }

    def batch_process_families(self, data_dir: str = "data"):
        """
        Process all RNA families in a directory.

        Args:
            data_dir: Directory containing TSV files
        """
        data_path = Path(data_dir)
        tsv_files = list(data_path.glob("RF*.tsv"))

        if not tsv_files:
            print(f"No RNA family files found in {data_dir}")
            return

        print(f"Found {len(tsv_files)} RNA families to process")

        all_results = []
        for tsv_file in tsv_files:
            try:
                results = self.run_complete_pipeline(str(tsv_file))
                all_results.append(results)
            except Exception as e:
                print(f"Error processing {tsv_file}: {e}")
                continue

        # Generate summary report
        if all_results:
            self.generate_summary_report(all_results)

        return all_results

    def generate_summary_report(self, all_results: List[Dict]):
        """
        Generate summary report across all RNA families.
        """
        summary_path = self.output_dir / "summary_report.md"

        with open(summary_path, 'w') as f:
            f.write("# WP5 Summary Report: All RNA Families\n\n")
            f.write(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Total families analyzed:** {len(all_results)}\n\n")

            # Aggregate statistics
            f.write("## Aggregate Statistics\n\n")

            # Collect metrics
            ratios = [r['comparison']['ratio'] for r in all_results]
            ari_scores = [r['comparison']['adjusted_rand_index'] for r in all_results]
            vcc_shift_ratios = [r['vcc'].shift_correlation['inter_cluster_ratio'] for r in all_results]
            ce_shift_ratios = [r['ce'].shift_correlation['inter_cluster_ratio'] for r in all_results]

            f.write("### Clustering Comparison (C/θ)\n")
            f.write(f"- Mean ratio: {np.mean(ratios):.3f}\n")
            f.write(f"- Std deviation: {np.std(ratios):.3f}\n")
            f.write(f"- Range: [{np.min(ratios):.3f}, {np.max(ratios):.3f}]\n\n")

            f.write("### Method Agreement (ARI)\n")
            f.write(f"- Mean ARI: {np.mean(ari_scores):.3f}\n")
            f.write(f"- Std deviation: {np.std(ari_scores):.3f}\n\n")

            f.write("### Shift Event Correlation\n")
            f.write(f"- VCC mean inter-cluster ratio: {np.mean(vcc_shift_ratios):.3f}\n")
            f.write(f"- CE mean inter-cluster ratio: {np.mean(ce_shift_ratios):.3f}\n\n")

            # Families with strong shift correlation
            f.write("## Families with Strong Shift-Cluster Correlation\n\n")
            strong_correlation = []
            for r in all_results:
                if r['vcc'].shift_correlation['inter_cluster_ratio'] > 0.7:
                    family = Path(r.get('family_name', 'Unknown')).stem
                    ratio = r['vcc'].shift_correlation['inter_cluster_ratio']
                    strong_correlation.append((family, ratio))

            if strong_correlation:
                for family, ratio in sorted(strong_correlation, key=lambda x: x[1], reverse=True):
                    f.write(f"- {family}: {ratio:.3f}\n")
            else:
                f.write("*No families with >0.7 inter-cluster shift ratio*\n")

            f.write("\n---\n")
            f.write("*Summary generated by WP5 RNA Analysis Pipeline*\n")

        print(f"  Saved summary report to {summary_path}")


def main():
    """
    Main entry point for WP5 RNA analysis.
    """
    import argparse

    parser = argparse.ArgumentParser(description='WP5: RNA Shift-Alignment Analysis')
    parser.add_argument('--input', type=str, default='data/RF02246.tsv',
                       help='Input TSV file or directory with RNA data')
    parser.add_argument('--output', type=str, default='results/wp5',
                       help='Output directory for results')
    parser.add_argument('--batch', action='store_true',
                       help='Process all TSV files in input directory')
    parser.add_argument('--method', type=str, default='heuristic',
                       choices=['exact', 'heuristic', 'reduced'],
                       help='VCC solving method')

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = WP5RfamAnalysis(output_dir=args.output)

    if args.batch:
        # Process all files in directory
        analyzer.batch_process_families(args.input)
    else:
        # Process single file
        analyzer.run_complete_pipeline(args.input)

    print("\nWP5 analysis complete!")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()