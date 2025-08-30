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

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from scipy import stats
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score #needs to be updated in uv / requirements
import warnings

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# VCC (Clique Cover)
from src.algorithms.ilp_solver import solve_ilp_clique_cover
from src.algorithms.chalupa import ChalupaHeuristic

# CE (Cluster Editing)
from src.algorithms.cluster_editing_solver import ClusterEditingSolver
from src.algorithms.cluster_editing_kernelization import (
    ClusterEditingInstance,
    ClusterEditingKernelization,
    OptimizedClusterEditingKernelization,
)

# Utilities & Generators
from src.utils import txt_to_networkx
from src.simulator import GraphGenerator, GraphConfig

# central helpers
from src.utils_metrics import (
    set_global_seeds, safe_ratio, rel_change,
    clean_for_plot, nanmean, safe_idxmax,
    should_kernelize, estimate_loglog_slope
)

set_global_seeds(33)

# ==================== Data Structures ====================

@dataclass
class ClusteringResult:
    """Unified format for both VCC and CE solutions."""
    graph: nx.Graph
    clusters: List[Set[int]]  # List of node sets
    num_clusters: int
    method: str  # "vcc", "ce", "vcc_heuristic", etc.
    metadata: Dict = field(default_factory=dict)

    def validate(self) -> bool:
        """Validate mathematical correctness of the solution."""
        if self.method in ["vcc", "vcc_heuristic"]:
            return self._validate_clique_cover()
        elif self.method in ["ce", "ce_kernelized", "ce_from_vcc", "vcc_from_ce"]:
            return self._validate_cluster_editing()
        return False

    def _validate_clique_cover(self) -> bool:
        """Validate VCC solution: all nodes covered, each cluster is a clique."""
        # Check coverage
        covered = set().union(*self.clusters) if self.clusters else set()
        if covered != set(self.graph.nodes()):
            print(f"VCC validation failed: not all nodes covered")
            return False

        # Check that each cluster is a clique
        for cluster in self.clusters:
            if not self._is_clique(cluster):
                print(f"VCC validation failed: cluster {cluster} is not a clique")
                return False
        return True

    def _validate_cluster_editing(self) -> bool:
        """Validate CE solution: disjoint clusters, each is a clique."""
        # Check disjointness
        all_nodes = set()
        for cluster in self.clusters:
            if all_nodes & cluster:
                print(f"CE validation failed: clusters not disjoint")
                return False
            all_nodes.update(cluster)

        # Check coverage
        if all_nodes != set(self.graph.nodes()):
            print(f"CE validation failed: not all nodes covered")
            return False

        # Check that each cluster is a clique
        for cluster in self.clusters:
            if not self._is_clique(cluster):
                print(f"CE validation failed: cluster {cluster} is not a clique")
                return False
        return True

    def _is_clique(self, nodes: Set[int]) -> bool:
        """Check if a set of nodes forms a clique."""
        if len(nodes) <= 1:
            return True
        subgraph = self.graph.subgraph(nodes)
        n = len(nodes)
        return subgraph.number_of_edges() == n * (n - 1) // 2


@dataclass
class ComparisonResult:
    """Result of comparing VCC and CE solutions."""
    graph_name: str
    graph_stats: Dict
    vcc_result: ClusteringResult
    ce_result: ClusteringResult
    theta: int  # θ(G)
    C: int  # C(G)
    ratio: float  # C/θ
    overlap_metrics: Dict
    quality_metrics: Dict
    heuristic_improvements: Dict = field(default_factory=dict)
    runtime_comparison: Dict = field(default_factory=dict)


# ==================== Solver Adapters ====================

class SolverAdapter:
    """Unified adapter for both VCC and CE solvers."""

    def __init__(self):
        self.vcc_exact_solver = None
        self.vcc_heuristic_solver = None
        self.ce_solver = None

    def solve_vcc(self, graph: nx.Graph, method: str = 'exact', **kwargs) -> ClusteringResult:
        """
        Solve VCC using specified method.

        Args:
            graph: Input graph
            method: 'exact' for ILP, 'heuristic' for Chalupa
        """
        if method == 'exact':
            return self._solve_vcc_exact(graph, **kwargs)
        elif method == 'heuristic':
            return self._solve_vcc_heuristic(graph, **kwargs)
        else:
            raise ValueError(f"Unknown VCC method: {method}")

    def _solve_vcc_exact(self, graph: nx.Graph, time_limit: int = 300) -> ClusteringResult:
        """Solve VCC exactly using ILP."""
        start_time = time.time()

        try:
            result = solve_ilp_clique_cover(
                graph,
                time_limit=time_limit,
                return_assignment=True
            )
            elapsed_time = time.time() - start_time

            # Extract clusters from coloring
            clusters = self._extract_vcc_clusters(result.get('assignment', {}))

            return ClusteringResult(
                graph=graph,
                clusters=clusters,
                num_clusters=result.get('theta', len(clusters)),
                method="vcc",
                metadata={
                    'time': elapsed_time,
                    'algorithm': 'ILP',
                    'status': result.get('status', 'unknown'),
                    'gap': result.get('gap', 0.0)
                }
            )
        except Exception as e:
            print(f"VCC exact solver failed: {e}")
            # Fallback to heuristic
            return self._solve_vcc_heuristic(graph)

    def _solve_vcc_heuristic(self, graph: nx.Graph, iterations: int = 1000) -> ClusteringResult:
        """Solve VCC using Chalupa heuristic."""
        start_time = time.time()

        try:
            # Use Chalupa on complement graph
            complement = nx.complement(graph)
            heuristic = ChalupaHeuristic(complement)
            coloring = heuristic.iterated_greedy_clique_covering(iterations=iterations)

            elapsed_time = time.time() - start_time

            # Extract clusters
            clusters = self._extract_vcc_clusters(coloring)

            return ClusteringResult(
                graph=graph,
                clusters=clusters,
                num_clusters=len(clusters),
                method="vcc_heuristic",
                metadata={
                    'time': elapsed_time,
                    'algorithm': 'Chalupa',
                    'iterations': iterations
                }
            )
        except Exception as e:
            print(f"VCC heuristic solver failed: {e}")
            # Return empty result
            return ClusteringResult(
                graph=graph,
                clusters=[],
                num_clusters=0,
                method="vcc_heuristic",
                metadata={'error': str(e)}
            )

    def solve_ce(self, graph: nx.Graph,
                 use_kernelization: bool = True,
                 kernelization_type: str = 'optimized',
                 algorithm: str = 'greedy_improved') -> ClusteringResult:
        """Solve CE with optional kernelization."""
        start_time = time.time()

        try:
            solver = ClusterEditingSolver(graph)
            result = solver.solve(
                use_kernelization=use_kernelization,
                kernelization_type=kernelization_type,
                clustering_algorithm=algorithm
            )

            elapsed_time = time.time() - start_time

            # Extract clusters
            clusters = [set(cluster) for cluster in result.get('clusters', [])]

            method = "ce_kernelized" if use_kernelization else "ce"

            return ClusteringResult(
                graph=graph,
                clusters=clusters,
                num_clusters=len(clusters),
                method=method,
                metadata={
                    'time': elapsed_time,
                    'algorithm': algorithm,
                    'kernelization': kernelization_type if use_kernelization else None,
                    'editing_cost': result.get('cost', 0),
                    'kernel_size': result.get('kernel_size', graph.number_of_nodes())
                }
            )
        except Exception as e:
            print(f"CE solver failed: {e}")
            return ClusteringResult(
                graph=graph,
                clusters=[],
                num_clusters=0,
                method="ce",
                metadata={'error': str(e)}
            )

    def _extract_vcc_clusters(self, coloring: Dict) -> List[Set[int]]:
        """Convert coloring to list of clusters."""
        if not coloring:
            return []

        clusters_dict = defaultdict(set)

        # Handle different coloring formats
        if isinstance(coloring, dict):
            first_key = next(iter(coloring.keys())) if coloring else None
            if first_key and isinstance(coloring[first_key], (list, set)):
                # Format: {color: [nodes]}
                for color, nodes in coloring.items():
                    clusters_dict[color] = set(nodes)
            else:
                # Format: {node: color}
                for node, color in coloring.items():
                    clusters_dict[color].add(node)

        return list(clusters_dict.values())


# ==================== Comparison Framework ====================

class ComparisonFramework:
    """Main framework for comparing VCC and CE solutions."""

    def __init__(self, adapter: SolverAdapter):
        self.adapter = adapter
        self.results = []

    def compare_solutions(self, graph: nx.Graph, graph_name: str = "unnamed") -> ComparisonResult:
        """
        Core comparison function: Compare VCC and CE solutions.

        Args:
            graph: Input graph
            graph_name: Name for identification

        Returns:
            ComparisonResult with all metrics
        """
        print(f"\nComparing solutions for {graph_name}...")

        # Solve both problems
        vcc_result = self.adapter.solve_vcc(graph, method='exact')
        if vcc_result.num_clusters == 0:  # Fallback to heuristic if exact failed
            vcc_result = self.adapter.solve_vcc(graph, method='heuristic')

        ce_result = self.adapter.solve_ce(graph)

        # Validate solutions
        if not vcc_result.validate():
            print(f"Warning: VCC solution invalid for {graph_name}")
        if not ce_result.validate():
            print(f"Warning: CE solution invalid for {graph_name}")

        # Compute metrics
        comparison = ComparisonResult(
            graph_name=graph_name,
            graph_stats=self._compute_graph_stats(graph),
            vcc_result=vcc_result,
            ce_result=ce_result,
            theta=vcc_result.num_clusters,
            C=ce_result.num_clusters,
            ratio=ce_result.num_clusters / vcc_result.num_clusters if vcc_result.num_clusters > 0 else float('inf'),
            overlap_metrics=self._analyze_overlaps(vcc_result, ce_result),
            quality_metrics=self._compute_quality_metrics(graph, vcc_result, ce_result),
            runtime_comparison={
                'vcc_time': vcc_result.metadata.get('time', 0),
                'ce_time': ce_result.metadata.get('time', 0),
                'speedup': vcc_result.metadata.get('time', 1) / ce_result.metadata.get('time', 1)
            }
        )

        # Check mathematical invariant
        if comparison.theta > comparison.C:
            print(f"WARNING: Invariant violated! θ(G)={comparison.theta} > C(G)={comparison.C}")

        self.results.append(comparison)
        return comparison

    def _compute_graph_stats(self, graph: nx.Graph) -> Dict:
        """Compute basic graph statistics."""
        return {
            'nodes': graph.number_of_nodes(),
            'edges': graph.number_of_edges(),
            'density': nx.density(graph),
            'avg_degree': 2 * graph.number_of_edges() / graph.number_of_nodes() if graph.number_of_nodes() > 0 else 0,
            'components': nx.number_connected_components(graph),
            'avg_clustering': nx.average_clustering(graph)
        }

    def _analyze_overlaps(self, vcc: ClusteringResult, ce: ClusteringResult) -> Dict:
        """Analyze structural differences between solutions."""
        # VCC cluster overlaps (only for VCC, as CE clusters are disjoint)
        vcc_overlap_matrix = np.zeros((len(vcc.clusters), len(vcc.clusters)))
        for i, c1 in enumerate(vcc.clusters):
            for j, c2 in enumerate(vcc.clusters):
                if i != j and len(c1) > 0 and len(c2) > 0:
                    vcc_overlap_matrix[i, j] = len(c1 & c2) / min(len(c1), len(c2))

        # Compute clustering agreement metrics
        vcc_labels = self._clusters_to_labels(vcc.clusters, vcc.graph.nodes())
        ce_labels = self._clusters_to_labels(ce.clusters, ce.graph.nodes())

        return {
            'vcc_avg_overlap': np.mean(vcc_overlap_matrix[vcc_overlap_matrix > 0]) if np.any(
                vcc_overlap_matrix > 0) else 0,
            'vcc_max_overlap': np.max(vcc_overlap_matrix),
            'adjusted_rand_index': adjusted_rand_score(vcc_labels, ce_labels),
            'nmi_score': normalized_mutual_info_score(vcc_labels, ce_labels),
            'jaccard_similarity': self._compute_jaccard_similarity(vcc.clusters, ce.clusters)
        }

    def _compute_quality_metrics(self, graph: nx.Graph,
                                 vcc: ClusteringResult,
                                 ce: ClusteringResult) -> Dict:
        """Compute clustering quality metrics."""
        metrics = {}

        # Modularity
        vcc_communities = [list(c) for c in vcc.clusters]
        ce_communities = [list(c) for c in ce.clusters]

        try:
            metrics['vcc_modularity'] = nx.algorithms.community.modularity(graph, vcc_communities)
            metrics['ce_modularity'] = nx.algorithms.community.modularity(graph, ce_communities)
        except:
            metrics['vcc_modularity'] = 0
            metrics['ce_modularity'] = 0

        # Conductance and density for each method
        metrics['vcc_avg_density'] = self._compute_avg_cluster_density(graph, vcc.clusters)
        metrics['ce_avg_density'] = self._compute_avg_cluster_density(graph, ce.clusters)

        metrics['vcc_avg_conductance'] = self._compute_avg_conductance(graph, vcc.clusters)
        metrics['ce_avg_conductance'] = self._compute_avg_conductance(graph, ce.clusters)

        # Silhouette coefficient (if applicable)
        if graph.number_of_nodes() > 2:
            metrics['vcc_silhouette'] = self._compute_silhouette(graph, vcc.clusters)
            metrics['ce_silhouette'] = self._compute_silhouette(graph, ce.clusters)

        return metrics

    def _clusters_to_labels(self, clusters: List[Set[int]], nodes: List) -> np.ndarray:
        """Convert cluster assignment to label array."""
        label_dict = {}
        for i, cluster in enumerate(clusters):
            for node in cluster:
                label_dict[node] = i

        return np.array([label_dict.get(node, -1) for node in nodes])

    def _compute_jaccard_similarity(self, clusters1: List[Set], clusters2: List[Set]) -> float:
        """Compute average Jaccard similarity between best-matching clusters."""
        if not clusters1 or not clusters2:
            return 0.0

        similarities = []
        for c1 in clusters1:
            best_sim = 0
            for c2 in clusters2:
                if len(c1 | c2) > 0:
                    sim = len(c1 & c2) / len(c1 | c2)
                    best_sim = max(best_sim, sim)
            similarities.append(best_sim)

        return np.mean(similarities)

    def _compute_avg_cluster_density(self, graph: nx.Graph, clusters: List[Set]) -> float:
        """Compute average density of clusters."""
        densities = []
        for cluster in clusters:
            if len(cluster) > 1:
                subgraph = graph.subgraph(cluster)
                densities.append(nx.density(subgraph))
        return np.mean(densities) if densities else 0

    def _compute_avg_conductance(self, graph: nx.Graph, clusters: List[Set]) -> float:
        """Compute average conductance of clusters."""
        conductances = []
        for cluster in clusters:
            if len(cluster) > 0 and len(cluster) < graph.number_of_nodes():
                conductance = nx.algorithms.cuts.conductance(graph, cluster)
                if conductance is not None:
                    conductances.append(conductance)
        return np.mean(conductances) if conductances else 0

    def _compute_silhouette(self, graph: nx.Graph, clusters: List[Set]) -> float:
        """Compute silhouette coefficient based on shortest paths."""
        # Simplified silhouette using graph distances
        try:
            if len(clusters) < 2:
                return 0

            # Compute distance matrix (using shortest paths)
            nodes = list(graph.nodes())
            n = len(nodes)
            if n > 100:  # Skip for large graphs
                return 0

            dist_matrix = np.full((n, n), np.inf)
            for i, u in enumerate(nodes):
                lengths = nx.single_source_shortest_path_length(graph, u)
                for j, v in enumerate(nodes):
                    if v in lengths:
                        dist_matrix[i][j] = lengths[v]

            # Compute silhouette for each node
            silhouettes = []
            node_to_cluster = {}
            for c_idx, cluster in enumerate(clusters):
                for node in cluster:
                    node_to_cluster[node] = c_idx

            for i, node in enumerate(nodes):
                if node not in node_to_cluster:
                    continue

                cluster_idx = node_to_cluster[node]

                # Average distance to nodes in same cluster
                same_cluster = [j for j, n in enumerate(nodes)
                                if n in clusters[cluster_idx] and j != i]
                if not same_cluster:
                    continue
                a = np.mean([dist_matrix[i][j] for j in same_cluster])

                # Minimum average distance to other clusters
                b = np.inf
                for other_idx, other_cluster in enumerate(clusters):
                    if other_idx != cluster_idx and len(other_cluster) > 0:
                        other_nodes = [j for j, n in enumerate(nodes) if n in other_cluster]
                        if other_nodes:
                            avg_dist = np.mean([dist_matrix[i][j] for j in other_nodes])
                            b = min(b, avg_dist)

                if b != np.inf:
                    silhouettes.append((b - a) / max(a, b))

            return np.mean(silhouettes) if silhouettes else 0

        except Exception as e:
            print(f"Silhouette computation failed: {e}")
            return 0


# ==================== Cross-Optimization Heuristics (Bonus) ====================

class CrossOptimizationHeuristic:
    """Heuristics to improve one solution using the other."""

    def improve_ce_from_vcc(self, graph: nx.Graph,
                            vcc_solution: ClusteringResult) -> ClusteringResult:
        """
        Use VCC solution to initialize CE.

        Strategy:
        1. Start with VCC cliques
        2. Resolve overlaps by assigning nodes to best cluster
        3. Apply local improvements
        """
        print("  Improving CE using VCC solution...")

        # Find overlapping nodes
        overlapping_nodes = self._find_overlapping_nodes(vcc_solution.clusters)

        # Create initial disjoint clustering
        disjoint_clusters = []
        assigned = set()

        # First, keep non-overlapping parts
        for cluster in vcc_solution.clusters:
            non_overlapping = cluster - overlapping_nodes
            if non_overlapping and non_overlapping not in disjoint_clusters:
                disjoint_clusters.append(non_overlapping)
                assigned.update(non_overlapping)

        # Assign overlapping nodes to best cluster
        for node in overlapping_nodes:
            best_cluster_idx = self._find_best_cluster_assignment(
                node, disjoint_clusters, graph
            )
            if best_cluster_idx >= 0:
                disjoint_clusters[best_cluster_idx].add(node)
            else:
                # Create new singleton cluster
                disjoint_clusters.append({node})

        # Apply local improvements
        improved_clusters = self._local_search_improvement(disjoint_clusters, graph)

        return ClusteringResult(
            graph=graph,
            clusters=improved_clusters,
            num_clusters=len(improved_clusters),
            method="ce_from_vcc",
            metadata={'source': 'vcc_based_heuristic'}
        )

    def improve_vcc_from_ce(self, graph: nx.Graph,
                            ce_solution: ClusteringResult) -> ClusteringResult:
        """
        Use CE solution to initialize VCC.

        Strategy:
        1. Start with CE clusters (already disjoint cliques)
        2. Try to merge clusters that form cliques
        3. Allow controlled overlaps for better coverage
        """
        print("  Improving VCC using CE solution...")

        # Start with CE clusters
        vcc_clusters = [set(c) for c in ce_solution.clusters]

        # Try to merge compatible clusters
        merged = True
        while merged:
            merged = False
            for i in range(len(vcc_clusters)):
                if i >= len(vcc_clusters):
                    break
                for j in range(i + 1, len(vcc_clusters)):
                    if j >= len(vcc_clusters):
                        break
                    # Check if union forms a clique
                    union = vcc_clusters[i] | vcc_clusters[j]
                    if self._is_clique(graph, union):
                        vcc_clusters[i] = union
                        vcc_clusters.pop(j)
                        merged = True
                        break

        # Allow strategic overlaps to reduce cluster count
        vcc_clusters = self._add_strategic_overlaps(graph, vcc_clusters)

        return ClusteringResult(
            graph=graph,
            clusters=vcc_clusters,
            num_clusters=len(vcc_clusters),
            method="vcc_from_ce",
            metadata={'source': 'ce_based_heuristic'}
        )

    def bidirectional_improvement(self, graph: nx.Graph,
                                  vcc_solution: ClusteringResult,
                                  ce_solution: ClusteringResult,
                                  max_iterations: int = 5) -> Tuple[ClusteringResult, ClusteringResult]:
        """
        Iteratively improve both solutions using each other.
        """
        print(f"  Bidirectional improvement (max {max_iterations} iterations)...")

        best_vcc = vcc_solution
        best_ce = ce_solution

        for iteration in range(max_iterations):
            # Improve CE using VCC
            new_ce = self.improve_ce_from_vcc(graph, best_vcc)
            if new_ce.num_clusters < best_ce.num_clusters:
                best_ce = new_ce

            # Improve VCC using CE
            new_vcc = self.improve_vcc_from_ce(graph, best_ce)
            if new_vcc.num_clusters < best_vcc.num_clusters:
                best_vcc = new_vcc

            # Check convergence
            if (new_ce.num_clusters == best_ce.num_clusters and
                    new_vcc.num_clusters == best_vcc.num_clusters):
                print(f"    Converged after {iteration + 1} iterations")
                break

        return best_vcc, best_ce

    def _find_overlapping_nodes(self, clusters: List[Set]) -> Set:
        """Find nodes that appear in multiple clusters."""
        node_counts = defaultdict(int)
        for cluster in clusters:
            for node in cluster:
                node_counts[node] += 1
        return {node for node, count in node_counts.items() if count > 1}

    def _find_best_cluster_assignment(self, node: int,
                                      clusters: List[Set],
                                      graph: nx.Graph) -> int:
        """Assign node to cluster with most connections."""
        best_score = -1
        best_idx = -1

        for idx, cluster in enumerate(clusters):
            # Count edges to cluster
            edges_to_cluster = sum(1 for n in cluster if graph.has_edge(node, n))
            # Normalize by cluster size
            score = edges_to_cluster / len(cluster) if len(cluster) > 0 else 0

            if score > best_score:
                best_score = score
                best_idx = idx

        return best_idx

    def _local_search_improvement(self, clusters: List[Set], graph: nx.Graph) -> List[Set]:
        """Apply local search to improve clustering."""
        improved = True
        iterations = 0
        max_iterations = 10

        while improved and iterations < max_iterations:
            improved = False
            iterations += 1

            # Try moving each node to a better cluster
            for i, cluster in enumerate(clusters):
                for node in list(cluster):
                    # Calculate current cost
                    current_cost = self._calculate_node_cost(node, cluster, graph)

                    # Try other clusters
                    for j, other_cluster in enumerate(clusters):
                        if i != j:
                            new_cost = self._calculate_node_cost(node, other_cluster, graph)
                            if new_cost < current_cost:
                                cluster.remove(node)
                                other_cluster.add(node)
                                improved = True
                                break

        # Remove empty clusters
        return [c for c in clusters if len(c) > 0]

    def _calculate_node_cost(self, node: int, cluster: Set, graph: nx.Graph) -> int:
        """Calculate cost of node in cluster (missing edges)."""
        missing_edges = 0
        for other in cluster:
            if other != node and not graph.has_edge(node, other):
                missing_edges += 1
        return missing_edges

    def _is_clique(self, graph: nx.Graph, nodes: Set) -> bool:
        """Check if nodes form a clique."""
        for u in nodes:
            for v in nodes:
                if u != v and not graph.has_edge(u, v):
                    return False
        return True

    def _add_strategic_overlaps(self, graph: nx.Graph, clusters: List[Set]) -> List[Set]:
        """Add strategic overlaps to reduce total cluster count."""
        # This is a simplified version - could be made more sophisticated
        new_clusters = []

        for cluster in clusters:
            # Check if cluster can be covered by existing new_clusters
            covered = False
            for new_cluster in new_clusters:
                if cluster.issubset(new_cluster):
                    covered = True
                    break

            if not covered:
                # Try to extend with neighboring nodes
                extended = set(cluster)
                for node in list(cluster):
                    for neighbor in graph.neighbors(node):
                        if neighbor not in extended:
                            # Check if adding neighbor maintains clique property
                            forms_clique = all(
                                graph.has_edge(neighbor, other)
                                for other in extended
                            )
                            if forms_clique:
                                extended.add(neighbor)

                new_clusters.append(extended)

        return new_clusters


# ==================== Statistical Analysis ====================

class StatisticalAnalyzer:
    """Statistical analysis of comparison results."""

    def __init__(self, results: List[ComparisonResult]):
        self.results = results
        self.df = self._results_to_dataframe(results)

    def _results_to_dataframe(self, results: List[ComparisonResult]) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        data = []
        for r in results:
            row = {
                'graph_name': r.graph_name,
                'nodes': r.graph_stats['nodes'],
                'edges': r.graph_stats['edges'],
                'density': r.graph_stats['density'],
                'theta': r.theta,
                'C': r.C,
                'ratio': r.ratio,
                'vcc_time': r.runtime_comparison.get('vcc_time', 0),
                'ce_time': r.runtime_comparison.get('ce_time', 0),
                'speedup': r.runtime_comparison.get('speedup', 1),
                'adjusted_rand_index': r.overlap_metrics.get('adjusted_rand_index', 0),
                'nmi_score': r.overlap_metrics.get('nmi_score', 0),
                'vcc_modularity': r.quality_metrics.get('vcc_modularity', 0),
                'ce_modularity': r.quality_metrics.get('ce_modularity', 0),
                'vcc_avg_density': r.quality_metrics.get('vcc_avg_density', 0),
                'ce_avg_density': r.quality_metrics.get('ce_avg_density', 0)
            }

            # Add heuristic improvements if available
            if r.heuristic_improvements:
                row.update({
                    'ce_from_vcc_clusters': r.heuristic_improvements.get('ce_from_vcc', 0),
                    'vcc_from_ce_clusters': r.heuristic_improvements.get('vcc_from_ce', 0),
                    'bidirectional_vcc': r.heuristic_improvements.get('bidirectional_vcc', 0),
                    'bidirectional_ce': r.heuristic_improvements.get('bidirectional_ce', 0)
                })

            data.append(row)

        return pd.DataFrame(data)

    def analyze_correlations(self) -> Dict:
        """Analyze correlations between graph properties and θ/C ratio."""
        correlations = {}

        if not self.df.empty:
            # Correlation with graph properties
            for prop in ['nodes', 'edges', 'density']:
                if prop in self.df.columns:
                    corr, p_value = stats.pearsonr(self.df[prop], self.df['ratio'])
                    correlations[f'{prop}_ratio_correlation'] = {
                        'correlation': corr,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }

            # Spearman correlation for non-linear relationships
            spearman_corr, spearman_p = stats.spearmanr(self.df['density'], self.df['ratio'])
            correlations['density_ratio_spearman'] = {
                'correlation': spearman_corr,
                'p_value': spearman_p
            }

        return correlations

    def test_significance(self) -> Dict:
        """Test if θ is significantly smaller than C."""
        results = {}

        if not self.df.empty and len(self.df) > 1:
            # Wilcoxon signed-rank test
            try:
                statistic, p_value = stats.wilcoxon(self.df['theta'], self.df['C'])
                results['wilcoxon'] = {
                    'statistic': statistic,
                    'p_value': p_value,
                    'theta_smaller': self.df['theta'].mean() < self.df['C'].mean(),
                    'significant': p_value < 0.05
                }
            except:
                results['wilcoxon'] = {'error': 'Not enough data for test'}

            # Paired t-test
            try:
                t_stat, t_p = stats.ttest_rel(self.df['theta'], self.df['C'])
                results['paired_ttest'] = {
                    't_statistic': t_stat,
                    'p_value': t_p,
                    'significant': t_p < 0.05
                }
            except:
                results['paired_ttest'] = {'error': 'Not enough data for test'}

            # Confidence interval for ratio
            ratios = self.df['ratio'].values
            results['ratio_ci'] = {
                'mean': np.mean(ratios),
                'std': np.std(ratios),
                'ci_95': (np.percentile(ratios, 2.5), np.percentile(ratios, 97.5))
            }

        return results

    def analyze_by_graph_type(self) -> Dict:
        """Analyze results by graph characteristics."""
        analysis = {}

        if not self.df.empty:
            # Categorize by size
            self.df['size_category'] = pd.cut(
                self.df['nodes'],
                bins=[0, 50, 100, 200, float('inf')],
                labels=['small', 'medium', 'large', 'very_large']
            )

            # Categorize by density
            self.df['density_category'] = pd.cut(
                self.df['density'],
                bins=[0, 0.1, 0.3, 0.5, 1.0],
                labels=['sparse', 'medium_sparse', 'medium_dense', 'dense']
            )

            # Analyze by categories
            for category in ['size_category', 'density_category']:
                if category in self.df.columns:
                    grouped = self.df.groupby(category)
                    analysis[category] = {
                        'mean_ratio': grouped['ratio'].mean().to_dict(),
                        'mean_theta': grouped['theta'].mean().to_dict(),
                        'mean_C': grouped['C'].mean().to_dict(),
                        'count': grouped.size().to_dict()
                    }

        return analysis

    def sensitivity_analysis(self) -> pd.DataFrame:
        """Analyze sensitivity to perturbation (if available in graph names)."""
        # Extract perturbation level from graph names if possible
        perturbation_data = []

        for _, row in self.df.iterrows():
            if 'perturbation' in row['graph_name'] or '_r' in row['graph_name']:
                # Try to extract perturbation level
                import re
                match = re.search(r'(?:perturbation|_r)(\d+)', row['graph_name'])
                if match:
                    pert_level = int(match.group(1)) / 100  # Convert to percentage
                    perturbation_data.append({
                        'perturbation': pert_level,
                        'ratio': row['ratio'],
                        'theta': row['theta'],
                        'C': row['C']
                    })

        return pd.DataFrame(perturbation_data) if perturbation_data else pd.DataFrame()


# ==================== Visualization ====================

class Visualizer:
    """Create visualizations for WP4 results."""

    def __init__(self, output_dir: str = "results/wp4/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

    def plot_theta_vs_c_scatter(self, df: pd.DataFrame):
        """Scatter plot of θ vs C with ideal line."""
        fig, ax = plt.subplots(figsize=(10, 8))

        # Color by density if available
        if 'density' in df.columns:
            scatter = ax.scatter(df['theta'], df['C'],
                                 c=df['density'],
                                 cmap='viridis',
                                 s=100, alpha=0.6)
            plt.colorbar(scatter, label='Graph Density')
        else:
            ax.scatter(df['theta'], df['C'], s=100, alpha=0.6)

        # Add diagonal line (θ = C)
        max_val = max(df['theta'].max(), df['C'].max())
        ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='θ = C')

        # Add trend line
        z = np.polyfit(df['theta'], df['C'], 1)
        p = np.poly1d(z)
        ax.plot(df['theta'], p(df['theta']), 'g-', alpha=0.5,
                label=f'Trend: C = {z[0]:.2f}θ + {z[1]:.2f}')

        ax.set_xlabel('θ(G) - Vertex Clique Cover Number', fontsize=12)
        ax.set_ylabel('C(G) - Cluster Editing Number', fontsize=12)
        ax.set_title('Comparison of VCC and CE Solutions', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'theta_vs_c_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_ratio_distribution(self, df: pd.DataFrame):
        """Distribution of C/θ ratios."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Histogram
        ax1.hist(df['ratio'], bins=20, edgecolor='black', alpha=0.7)
        ax1.axvline(1.0, color='r', linestyle='--', label='Ideal (C/θ = 1)')
        ax1.axvline(df['ratio'].mean(), color='g', linestyle='-',
                    label=f'Mean = {df["ratio"].mean():.2f}')
        ax1.set_xlabel('C/θ Ratio', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Distribution of C/θ Ratios', fontsize=14)
        ax1.legend()

        # Box plot by graph size category
        if 'nodes' in df.columns:
            df['size_cat'] = pd.cut(df['nodes'], bins=[0, 50, 100, 200, float('inf')],
                                    labels=['<50', '50-100', '100-200', '>200'])
            df.boxplot(column='ratio', by='size_cat', ax=ax2)
            ax2.set_xlabel('Graph Size (nodes)', fontsize=12)
            ax2.set_ylabel('C/θ Ratio', fontsize=12)
            ax2.set_title('C/θ Ratio by Graph Size', fontsize=14)
            plt.sca(ax2)
            plt.xticks(rotation=0)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'ratio_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_runtime_comparison(self, df: pd.DataFrame):
        """Compare runtimes of VCC and CE."""
        if 'vcc_time' not in df.columns or 'ce_time' not in df.columns:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Scatter plot of runtimes
        ax1.scatter(df['vcc_time'], df['ce_time'], alpha=0.6, s=50)

        # Add diagonal line
        max_time = max(df['vcc_time'].max(), df['ce_time'].max())
        ax1.plot([0, max_time], [0, max_time], 'r--', alpha=0.5, label='Equal time')

        ax1.set_xlabel('VCC Time (s)', fontsize=12)
        ax1.set_ylabel('CE Time (s)', fontsize=12)
        ax1.set_title('Runtime Comparison', fontsize=14)
        ax1.legend()
        ax1.set_xscale('log')
        ax1.set_yscale('log')

        # Speedup distribution
        speedups = df['vcc_time'] / df['ce_time'].replace(0, np.nan)
        ax2.hist(speedups.dropna(), bins=20, edgecolor='black', alpha=0.7)
        ax2.axvline(1.0, color='r', linestyle='--', label='No speedup')
        ax2.axvline(speedups.mean(), color='g', linestyle='-',
                    label=f'Mean = {speedups.mean():.2f}')
        ax2.set_xlabel('Speedup (VCC time / CE time)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Speedup Distribution', fontsize=14)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / 'runtime_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_quality_metrics(self, df: pd.DataFrame):
        """Compare quality metrics between VCC and CE."""
        metrics = ['modularity', 'avg_density']
        fig, axes = plt.subplots(1, len(metrics), figsize=(14, 6))

        for idx, metric in enumerate(metrics):
            vcc_col = f'vcc_{metric}'
            ce_col = f'ce_{metric}'

            if vcc_col in df.columns and ce_col in df.columns:
                ax = axes[idx] if len(metrics) > 1 else axes

                # Create paired plot
                x = range(len(df))
                ax.scatter(x, df[vcc_col], label='VCC', alpha=0.6, s=30)
                ax.scatter(x, df[ce_col], label='CE', alpha=0.6, s=30)

                # Connect pairs
                for i in x:
                    ax.plot([i, i], [df.iloc[i][vcc_col], df.iloc[i][ce_col]],
                            'k-', alpha=0.2)

                ax.set_xlabel('Graph Instance', fontsize=12)
                ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
                ax.set_title(f'{metric.replace("_", " ").title()} Comparison', fontsize=14)
                ax.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / 'quality_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_heatmap_overlap(self, overlap_matrix: np.ndarray, title: str = "Cluster Overlap"):
        """Plot heatmap of cluster overlaps."""
        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(overlap_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                    cbar_kws={'label': 'Overlap Ratio'},
                    ax=ax)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Cluster Index', fontsize=12)
        ax.set_ylabel('Cluster Index', fontsize=12)

        plt.tight_layout()
        plt.savefig(self.output_dir / f'{title.lower().replace(" ", "_")}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def plot_sensitivity_analysis(self, sensitivity_df: pd.DataFrame):
        """Plot sensitivity to perturbation strength."""
        if sensitivity_df.empty:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        # Group by perturbation level
        grouped = sensitivity_df.groupby('perturbation').agg({
            'ratio': ['mean', 'std'],
            'theta': 'mean',
            'C': 'mean'
        })

        # Plot ratio vs perturbation
        ax.errorbar(grouped.index, grouped['ratio']['mean'],
                    yerr=grouped['ratio']['std'],
                    marker='o', capsize=5, label='C/θ Ratio')

        ax.set_xlabel('Perturbation Strength', fontsize=12)
        ax.set_ylabel('C/θ Ratio', fontsize=12)
        ax.set_title('Sensitivity to Perturbation', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / 'sensitivity_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()


# ==================== Report Generation ====================

class ReportGenerator:
    """Generate comprehensive reports for WP4 results."""

    def __init__(self, output_dir: str = "results/wp4"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_markdown_report(self,
                                 results: List[ComparisonResult],
                                 stats_results: Dict,
                                 timestamp: str) -> str:
        """Generate comprehensive markdown report."""
        report_path = self.output_dir / f"wp4_report_{timestamp}.md"

        with open(report_path, 'w') as f:
            f.write("# WP4 Analysis Report: VCC vs CE Comparison\n\n")
            f.write(f"**Generated:** {timestamp}\n\n")

            # Executive Summary
            f.write("## Executive Summary\n\n")

            if results:
                avg_ratio = np.mean([r.ratio for r in results])
                f.write(f"- **Average C/θ ratio:** {avg_ratio:.3f}\n")
                f.write(f"- **Total instances analyzed:** {len(results)}\n")

                # Check invariant
                violations = sum(1 for r in results if r.theta > r.C)
                if violations > 0:
                    f.write(f"- ⚠️ **Invariant violations:** {violations} instances where θ > C\n")
                else:
                    f.write("- ✅ **Invariant satisfied:** θ ≤ C for all instances\n")

            # Statistical Significance
            f.write("\n## Statistical Analysis\n\n")
            if 'significance' in stats_results:
                sig = stats_results['significance']
                if 'wilcoxon' in sig:
                    f.write(f"### Wilcoxon Test\n")
                    f.write(f"- p-value: {sig['wilcoxon'].get('p_value', 'N/A'):.4f}\n")
                    f.write(f"- Significant: {'Yes' if sig['wilcoxon'].get('significant') else 'No'}\n\n")

            # Correlations
            if 'correlations' in stats_results:
                f.write("### Correlations with Graph Properties\n\n")
                f.write("| Property | Correlation with C/θ | p-value | Significant |\n")
                f.write("|----------|---------------------|---------|-------------|\n")

                for key, val in stats_results['correlations'].items():
                    if isinstance(val, dict) and 'correlation' in val:
                        f.write(f"| {key.replace('_', ' ')} | {val['correlation']:.3f} | ")
                        f.write(f"{val.get('p_value', 0):.4f} | ")
                        f.write(f"{'Yes' if val.get('significant') else 'No'} |\n")

            # Instance-specific analysis
            f.write("\n## Instance-Specific Results\n\n")
            if results:
                f.write("| Graph | Nodes | Edges | θ(G) | C(G) | C/θ | VCC Time | CE Time |\n")
                f.write("|-------|-------|-------|------|------|-----|----------|----------|\n")

                for r in results[:20]:  # Show first 20
                    f.write(f"| {r.graph_name[:20]} | {r.graph_stats['nodes']} | ")
                    f.write(f"{r.graph_stats['edges']} | {r.theta} | {r.C} | ")
                    f.write(f"{r.ratio:.2f} | {r.runtime_comparison.get('vcc_time', 0):.3f}s | ")
                    f.write(f"{r.runtime_comparison.get('ce_time', 0):.3f}s |\n")

                if len(results) > 20:
                    f.write(f"\n*... and {len(results) - 20} more instances*\n")

            # Recommendations
            f.write("\n## Recommendations\n\n")
            f.write(self._generate_recommendations(results, stats_results))

            f.write("\n---\n")
            f.write(f"*Report generated by WP4 Comparison Framework*\n")

        print(f"Report saved to {report_path}")
        return str(report_path)

    def generate_csv_export(self, df: pd.DataFrame, timestamp: str):
        """Export results to CSV."""
        csv_path = self.output_dir / f"wp4_results_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"CSV exported to {csv_path}")
        return str(csv_path)

    def _generate_recommendations(self, results: List[ComparisonResult], stats: Dict) -> str:
        """Generate recommendations based on analysis."""
        recommendations = []

        if results:
            avg_ratio = np.mean([r.ratio for r in results])

            if avg_ratio < 1.1:
                recommendations.append(
                    "- **Strong agreement**: VCC and CE produce very similar results. "
                    "Consider using CE for guaranteed disjoint clusters."
                )
            elif avg_ratio < 1.5:
                recommendations.append(
                    "- **Moderate agreement**: CE requires moderately more clusters. "
                    "Use VCC when overlaps are acceptable for better compression."
                )
            else:
                recommendations.append(
                    "- **Weak agreement**: Significant difference between methods. "
                    "Graph structure may not be well-suited for disjoint clustering."
                )

            # Runtime recommendations
            avg_speedup = np.mean([r.runtime_comparison.get('speedup', 1) for r in results])
            if avg_speedup > 2:
                recommendations.append(
                    "- **Performance**: CE is significantly faster than VCC. "
                    "Prefer CE for large-scale applications."
                )
            elif avg_speedup < 0.5:
                recommendations.append(
                    "- **Performance**: VCC is faster than CE. "
                    "Consider VCC heuristics for time-critical applications."
                )

        return "\n".join(recommendations) if recommendations else "No specific recommendations."


# ==================== Main Execution ====================

def main():
    """Main execution function for WP4."""
    parser = argparse.ArgumentParser(description='WP4: Compare VCC and CE solutions')
    parser.add_argument('--test-dir', type=str, default='test_graphs/generated',
                        help='Directory containing test graphs')
    parser.add_argument('--output-dir', type=str, default='results/wp4',
                        help='Output directory for results')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick test with fewer instances')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--skip-visualizations', action='store_true',
                        help='Skip generating visualizations')

    args = parser.parse_args()

    # Setup
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'=' * 80}")
    print(f"WP4: Comparison of Vertex Clique Cover and Cluster Editing")
    print(f"{'=' * 80}\n")

    # Initialize components
    adapter = SolverAdapter()
    framework = ComparisonFramework(adapter)
    heuristic = CrossOptimizationHeuristic()

    # Load test graphs
    test_graphs = load_test_graphs(args.test_dir, quick=args.quick)

    if not test_graphs:
        print("No test graphs found!")
        return

    print(f"Loaded {len(test_graphs)} test graphs\n")

    # Run comparisons
    all_results = []
    for graph_name, graph in test_graphs:
        try:
            # Basic comparison
            result = framework.compare_solutions(graph, graph_name)

            # Test heuristic improvements
            print("  Testing cross-optimization heuristics...")
            ce_improved = heuristic.improve_ce_from_vcc(graph, result.vcc_result)
            vcc_improved = heuristic.improve_vcc_from_ce(graph, result.ce_result)

            # Bidirectional improvement
            vcc_best, ce_best = heuristic.bidirectional_improvement(
                graph, result.vcc_result, result.ce_result
            )

            # Store improvement results
            result.heuristic_improvements = {
                'ce_from_vcc': ce_improved.num_clusters,
                'vcc_from_ce': vcc_improved.num_clusters,
                'bidirectional_vcc': vcc_best.num_clusters,
                'bidirectional_ce': ce_best.num_clusters
            }

            all_results.append(result)

            # Progress update
            print(f"  Completed: θ={result.theta}, C={result.C}, ratio={result.ratio:.3f}")
            print(f"  Improvements: CE from VCC={ce_improved.num_clusters}, "
                  f"VCC from CE={vcc_improved.num_clusters}\n")

        except Exception as e:
            print(f"  Error processing {graph_name}: {e}\n")
            continue

    # Statistical analysis
    print("\n" + "=" * 80)
    print("Statistical Analysis")
    print("=" * 80 + "\n")

    analyzer = StatisticalAnalyzer(all_results)
    stats_results = {
        'correlations': analyzer.analyze_correlations(),
        'significance': analyzer.test_significance(),
        'by_graph_type': analyzer.analyze_by_graph_type()
    }

    # Print summary statistics
    print(f"Summary Statistics:")
    print(f"  Average θ: {analyzer.df['theta'].mean():.2f}")
    print(f"  Average C: {analyzer.df['C'].mean():.2f}")
    print(f"  Average C/θ ratio: {analyzer.df['ratio'].mean():.3f}")
    print(f"  Ratio std dev: {analyzer.df['ratio'].std():.3f}")

    # Visualizations
    if not args.skip_visualizations:
        print("\nGenerating visualizations...")
        visualizer = Visualizer(f"{args.output_dir}/figures")
        visualizer.plot_theta_vs_c_scatter(analyzer.df)
        visualizer.plot_ratio_distribution(analyzer.df)
        visualizer.plot_runtime_comparison(analyzer.df)
        visualizer.plot_quality_metrics(analyzer.df)

        # Sensitivity analysis if applicable
        sensitivity_df = analyzer.sensitivity_analysis()
        if not sensitivity_df.empty:
            visualizer.plot_sensitivity_analysis(sensitivity_df)

    # Generate reports
    print("\nGenerating reports...")
    reporter = ReportGenerator(args.output_dir)
    report_path = reporter.generate_markdown_report(all_results, stats_results, timestamp)
    csv_path = reporter.generate_csv_export(analyzer.df, timestamp)

    # Final summary
    print("\n" + "=" * 80)
    print("WP4 Analysis Complete!")
    print("=" * 80)
    print(f"  Total instances analyzed: {len(all_results)}")
    print(f"  Report saved to: {report_path}")
    print(f"  CSV data saved to: {csv_path}")
    if not args.skip_visualizations:
        print(f"  Visualizations saved to: {args.output_dir}/figures/")
    print("=" * 80 + "\n")


def load_test_graphs(directory: str, quick: bool = False) -> List[Tuple[str, nx.Graph]]:
    """Load test graphs from directory."""
    graphs = []
    dir_path = Path(directory)

    if not dir_path.exists():
        print(f"Directory {directory} not found!")
        return graphs

    # Find all .txt files
    txt_files = list(dir_path.glob("**/*.txt"))

    if quick:
        # For quick test, use subset
        txt_files = txt_files[:10]

    for file_path in txt_files:
        try:
            graph = txt_to_networkx(str(file_path))
            graph_name = file_path.stem
            graphs.append((graph_name, graph))
        except Exception as e:
            print(f"Could not load {file_path}: {e}")
            continue

    return graphs


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
