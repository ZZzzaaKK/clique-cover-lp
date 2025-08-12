"""
Cluster editing solvers using kernelization.

This module provides:
- solver classes for cluster editing
- ClusterEditingSolver: Main solver interface
- Various clustering algorithms (greedy, pivot, etc.)
- Cost calculation utilities
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Set, Optional, Any, Tuple
import time

from .cluster_editing_kernelization import (
    ClusterEditingInstance,
    ClusterEditingKernelization,
    OptimizedClusterEditingKernelization
)


class ClusterEditingSolver:
    """
    Main solver for cluster editing problem with kernelization.

    Attributes:
        original_graph: Original input graph
        instance: Current working instance
        kernel: Kernelized instance
        solution: Final clustering solution
    """

    def __init__(self, graph: nx.Graph, weights: Optional[Dict] = None):
        """
        Initialize solver.

        Args:
            graph: Input graph
            weights: Optional edge weights (default: unit weights)
        """
        self.original_graph = graph.copy()

        if weights is None:
            weights = self._init_unit_weights(graph)

        self.instance = ClusterEditingInstance(graph.copy(), weights)
        self.kernel = None
        self.solution = None

    def _init_unit_weights(self, graph: nx.Graph) -> Dict:
        """
        Initialize unit weights (+1 for edges, -1 for non-edges).

        Args:
            graph: Input graph

        Returns:
            Dictionary of edge weights
        """
        weights = {}
        nodes = list(graph.nodes())

        # Efficient initialization based on density
        if graph.number_of_edges() > 0.25 * len(nodes) * (len(nodes) - 1) / 2:
            # Dense graph: use adjacency matrix
            adj_matrix = nx.adjacency_matrix(graph, nodes).todense()
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    weights[(nodes[i], nodes[j])] = 1.0 if adj_matrix[i, j] else -1.0
        else:
            # Sparse graph: use edge set
            edges_set = set(graph.edges())
            for i, u in enumerate(nodes):
                for v in nodes[i + 1:]:
                    edge = (u, v) if (u, v) in edges_set else (v, u)
                    weights[(min(u, v), max(u, v))] = 1.0 if edge in edges_set else -1.0

        return weights

    def solve(self,
              use_kernelization: bool = True,
              kernelization_type: str = 'optimized',
              clustering_algorithm: str = 'greedy_improved',
              **kwargs) -> Dict[str, Any]:
        """
        Solve the cluster editing problem.

        Args:
            use_kernelization: Whether to apply kernelization
            kernelization_type: 'standard' or 'optimized'
            clustering_algorithm: Algorithm to use for clustering
            **kwargs: Additional algorithm-specific parameters

        Returns:
            Solution dictionary with results and statistics
        """
        start_time = time.time()

        # Store original size
        original_stats = {
            'nodes': self.original_graph.number_of_nodes(),
            'edges': self.original_graph.number_of_edges()
        }

        # Apply kernelization if requested
        if use_kernelization:
            if kernelization_type == 'optimized':
                kernelizer = OptimizedClusterEditingKernelization(self.instance)
            else:
                kernelizer = ClusterEditingKernelization(self.instance)

            self.kernel = kernelizer.kernelize()
            kernel_stats = kernelizer.get_kernel_statistics()
        else:
            self.kernel = self.instance
            kernel_stats = None

        # Solve the (kernelized) instance
        if clustering_algorithm == 'greedy':
            clusters = self._greedy_clustering(self.kernel)
        elif clustering_algorithm == 'greedy_improved':
            clusters = self._greedy_clustering_improved(self.kernel)
        elif clustering_algorithm == 'pivot':
            clusters = self._pivot_clustering(self.kernel)
        elif clustering_algorithm == 'exact_small':
            clusters = self._solve_exact_small(self.kernel, **kwargs)
        else:
            raise ValueError(f"Unknown algorithm: {clustering_algorithm}")

        # Calculate cost
        editing_cost = self.calculate_editing_cost(self.kernel, clusters)

        # Prepare results
        elapsed_time = time.time() - start_time

        self.solution = {
            'clusters': clusters,
            'num_clusters': len(clusters),
            'editing_cost': editing_cost,
            'time_seconds': elapsed_time,
            'original_stats': original_stats,
            'kernel_stats': kernel_stats,
            'algorithm': clustering_algorithm,
            'use_kernelization': use_kernelization
        }

        return self.solution

    def _greedy_clustering(self, instance: ClusterEditingInstance) -> List[Set[int]]:
        """
        Basic greedy clustering algorithm.

        Args:
            instance: Cluster editing instance

        Returns:
            List of clusters (sets of vertices)
        """
        clusters = []
        unassigned = set(instance.graph.nodes())

        while unassigned:
            # Start new cluster with arbitrary vertex
            v = unassigned.pop()
            cluster = {v}

            # Add vertices with positive total weight to cluster
            candidates = list(unassigned)
            for u in candidates:
                total_weight = sum(
                    instance.get_weight(u, w) for w in cluster
                )
                if total_weight > 0:
                    cluster.add(u)
                    unassigned.remove(u)

            clusters.append(cluster)

        return clusters

    def _greedy_clustering_improved(self, instance: ClusterEditingInstance) -> List[Set[int]]:
        """
        Improved greedy clustering with pivot selection and local optimization.

        Args:
            instance: Cluster editing instance

        Returns:
            List of clusters
        """
        clusters = []
        unassigned = set(instance.graph.nodes())

        while unassigned:
            # Select best pivot (highest positive weight sum)
            pivot = self._select_best_pivot(instance, unassigned)
            cluster = {pivot}
            unassigned.remove(pivot)

            # Build cluster iteratively
            improved = True
            while improved:
                improved = False
                best_addition = None
                best_score = 0

                for u in list(unassigned):
                    score = sum(
                        instance.get_weight(u, v)
                        for v in cluster
                    )
                    if score > best_score:
                        best_score = score
                        best_addition = u

                if best_addition and best_score > 0:
                    cluster.add(best_addition)
                    unassigned.remove(best_addition)
                    improved = True

            clusters.append(cluster)

        # Apply local optimization
        self._optimize_clusters_local(instance, clusters)

        return clusters

    def _pivot_clustering(self, instance: ClusterEditingInstance) -> List[Set[int]]:
        """
        Pivot-based clustering algorithm.

        Processes vertices in order of their weight scores.

        Args:
            instance: Cluster editing instance

        Returns:
            List of clusters
        """
        clusters = []
        remaining = set(instance.graph.nodes())

        # Compute vertex scores
        scores = {}
        for v in remaining:
            scores[v] = instance.get_neighbor_sum(v)

        # Process vertices by decreasing score
        sorted_vertices = sorted(remaining, key=lambda x: scores[x], reverse=True)

        for pivot in sorted_vertices:
            if pivot not in remaining:
                continue

            # Form cluster around pivot
            cluster = {pivot}
            remaining.remove(pivot)

            # Add vertices with positive connection
            for v in list(remaining):
                if sum(instance.get_weight(v, u) for u in cluster) > 0:
                    cluster.add(v)
                    remaining.remove(v)

            clusters.append(cluster)

        return clusters

    def _solve_exact_small(self, instance: ClusterEditingInstance,
                          max_exact_size: int = 10) -> List[Set[int]]:
        """
        Exact solution for small instances using exhaustive search.

        Args:
            instance: Cluster editing instance
            max_exact_size: Maximum size for exact solution

        Returns:
            List of clusters
        """
        nodes = list(instance.graph.nodes())
        n = len(nodes)

        if n <= 1:
            return [set(nodes)] if nodes else []

        if n > max_exact_size:
            # Fall back to heuristic
            return self._greedy_clustering_improved(instance)

        # Try all partitions (Bell number B_n enumeration)
        best_cost = float('inf')
        best_clustering = None

        def generate_partitions(items, current=[]):
            """Generate all partitions of items."""
            if not items:
                yield current
            else:
                first = items[0]
                rest = items[1:]
                # Add to existing subset
                for i, subset in enumerate(current):
                    new_current = current[:i] + [subset | {first}] + current[i+1:]
                    yield from generate_partitions(rest, new_current)
                # Create new subset
                yield from generate_partitions(rest, current + [{first}])

        for partition in generate_partitions(nodes[1:], [{nodes[0]}]):
            cost = self.calculate_editing_cost(instance, partition)
            if cost < best_cost:
                best_cost = cost
                best_clustering = partition

        return best_clustering

    def _select_best_pivot(self, instance: ClusterEditingInstance,
                          candidates: Set[int]) -> int:
        """
        Select pivot vertex with maximum positive weight sum.

        Args:
            instance: Cluster editing instance
            candidates: Set of candidate vertices

        Returns:
            Best pivot vertex
        """
        best_pivot = None
        best_score = float('-inf')

        for v in candidates:
            score = sum(
                max(0, instance.get_weight(v, u))
                for u in instance.graph.neighbors(v)
                if u in candidates
            )
            if score > best_score:
                best_score = score
                best_pivot = v

        return best_pivot if best_pivot else next(iter(candidates))

    def _optimize_clusters_local(self, instance: ClusterEditingInstance,
                                 clusters: List[Set[int]],
                                 max_iterations: int = 10):
        """
        Local optimization by moving vertices between clusters.

        Args:
            instance: Cluster editing instance
            clusters: Current clustering
            max_iterations: Maximum optimization iterations
        """
        improved = True
        iteration = 0

        while improved and iteration < max_iterations:
            improved = False

            for i, cluster_i in enumerate(clusters):
                vertices_to_move = []

                for v in list(cluster_i):
                    # Cost of keeping v in current cluster
                    cost_stay = sum(
                        instance.get_weight(v, u)
                        for u in cluster_i if u != v
                    )

                    # Find best alternative cluster
                    best_cluster_idx = -1
                    best_cost_move = cost_stay

                    for j, cluster_j in enumerate(clusters):
                        if i != j:
                            cost_move = sum(
                                instance.get_weight(v, u)
                                for u in cluster_j
                            )
                            if cost_move > best_cost_move:
                                best_cost_move = cost_move
                                best_cluster_idx = j

                    if best_cluster_idx >= 0:
                        vertices_to_move.append((v, best_cluster_idx))

                # Apply moves
                for v, target_idx in vertices_to_move:
                    if v in cluster_i:  # Check if still there
                        cluster_i.remove(v)
                        clusters[target_idx].add(v)
                        improved = True

            # Remove empty clusters
            clusters[:] = [c for c in clusters if c]
            iteration += 1

    def calculate_editing_cost(self, instance: ClusterEditingInstance,
                               clusters: List[Set[int]]) -> float:
        """
        Calculate the cost of transforming the graph into the given clustering.

        Args:
            instance: Cluster editing instance
            clusters: Clustering solution

        Returns:
            Total editing cost
        """
        cost = 0.0

        # Cost of making each cluster complete
        for cluster in clusters:
            for u in cluster:
                for v in cluster:
                    if u < v and not instance.graph.has_edge(u, v):
                        cost += abs(instance.get_weight(u, v))

        # Cost of removing edges between clusters
        for i, cluster_i in enumerate(clusters):
            for j, cluster_j in enumerate(clusters[i+1:], i+1):
                for u in cluster_i:
                    for v in cluster_j:
                        if instance.graph.has_edge(u, v):
                            cost += abs(instance.get_weight(u, v))

        return cost

    def get_solution_quality_metrics(self) -> Dict[str, Any]:
        """
        Get quality metrics for the current solution.

        Returns:
            Dictionary with quality metrics
        """
        if not self.solution:
            return {}

        clusters = self.solution['clusters']

        # Calculate cluster size statistics
        cluster_sizes = [len(c) for c in clusters]

        # Calculate intra/inter cluster edge ratios
        intra_edges = sum(
            sum(1 for u in c for v in c if u < v and self.kernel.graph.has_edge(u, v))
            for c in clusters
        )

        inter_edges = sum(
            sum(1 for u in ci for v in cj if self.kernel.graph.has_edge(u, v))
            for i, ci in enumerate(clusters)
            for j, cj in enumerate(clusters[i+1:], i+1)
        )

        total_possible_intra = sum(len(c) * (len(c) - 1) // 2 for c in clusters)

        metrics = {
            'num_clusters': len(clusters),
            'avg_cluster_size': np.mean(cluster_sizes) if cluster_sizes else 0,
            'std_cluster_size': np.std(cluster_sizes) if cluster_sizes else 0,
            'min_cluster_size': min(cluster_sizes) if cluster_sizes else 0,
            'max_cluster_size': max(cluster_sizes) if cluster_sizes else 0,
            'singleton_clusters': sum(1 for c in clusters if len(c) == 1),
            'intra_cluster_edges': intra_edges,
            'inter_cluster_edges': inter_edges,
            'intra_cluster_density': intra_edges / total_possible_intra if total_possible_intra > 0 else 0,
            'editing_cost': self.solution.get('editing_cost', 0)
        }

        return metrics


def create_weighted_instance(graph: nx.Graph,
                            weight_function: str = 'unit') -> ClusterEditingInstance:
    """
    Create a weighted cluster editing instance.

    Args:
        graph: Input graph
        weight_function: Type of weight function ('unit', 'random', 'degree')

    Returns:
        ClusterEditingInstance with specified weights
    """
    import random

    weights = {}
    nodes = list(graph.nodes())

    for i, u in enumerate(nodes):
        for v in nodes[i+1:]:
            edge = (u, v)

            if weight_function == 'unit':
                # Unit weights
                weights[edge] = 1.0 if graph.has_edge(u, v) else -1.0

            elif weight_function == 'random':
                # Random weights in [-2, 2]
                if graph.has_edge(u, v):
                    weights[edge] = random.uniform(0.5, 2.0)
                else:
                    weights[edge] = random.uniform(-2.0, -0.5)

            elif weight_function == 'degree':
                # Weight based on vertex degrees
                deg_u = graph.degree(u)
                deg_v = graph.degree(v)
                if graph.has_edge(u, v):
                    weights[edge] = 1.0 + (deg_u + deg_v) / (2 * len(nodes))
                else:
                    weights[edge] = -1.0 - (deg_u + deg_v) / (4 * len(nodes))

            else:
                raise ValueError(f"Unknown weight function: {weight_function}")

    return ClusterEditingInstance(graph, weights)