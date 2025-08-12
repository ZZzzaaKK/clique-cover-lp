"""
OG skript for Cluster Editing, beeinhaltet aber auch noch die Evaluation, die wohl eher nicht in algorithms gehört
WP3: Kernelization for Cluster Editing Problem
===============================================
Implementation of kernelization techniques for the cluster editing problem
based on critical cliques and weighted reduction rules.

References:
- Böcker & Baumbach (2013): Cluster editing
- Cao & Chen (2012): Cluster editing: Kernelization based on edge cuts
- Guo (2009): A more effective linear kernelization for cluster editing
"""

import networkx as nx
import numpy as np
from typing import Dict, Tuple, Set, List, Optional, Any
from collections import defaultdict
import time
from dataclasses import dataclass


@dataclass
class ClusterEditingInstance:
    """Represents a weighted cluster editing instance"""
    graph: nx.Graph
    weights: Dict[Tuple[int, int], float]
    k: Optional[float] = None  # Parameter (budget) for modifications

    def copy(self):
        """Create a deep copy of the instance"""
        return ClusterEditingInstance(
            graph=self.graph.copy(),
            weights=self.weights.copy(),
            k=self.k
        )


class CriticalClique:
    """Manages critical cliques in a graph"""

    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.critical_cliques = []
        self._find_critical_cliques()

    def _find_critical_cliques(self):
        """
        Find all critical cliques in the graph.
        A critical clique is a maximal clique where all vertices have identical neighborhoods.
        """
        visited = set()

        for v in self.graph.nodes():
            if v in visited:
                continue

            # Find vertices with same closed neighborhood as v
            clique = {v}
            v_neighbors = set(self.graph.neighbors(v)) | {v}

            for u in self.graph.nodes():
                if u != v and u not in visited:
                    u_neighbors = set(self.graph.neighbors(u)) | {u}
                    if v_neighbors == u_neighbors:
                        clique.add(u)

            # Verify it's actually a clique
            if self._is_clique(clique):
                self.critical_cliques.append(clique)
                visited.update(clique)

    def _is_clique(self, vertices: Set[int]) -> bool:
        """Check if a set of vertices forms a clique"""
        for u in vertices:
            for v in vertices:
                if u != v and not self.graph.has_edge(u, v):
                    return False
        return True

    def merge_critical_cliques(self) -> Tuple[nx.Graph, Dict[int, Set[int]]]:
        """
        Merge vertices in each critical clique into a single vertex.
        Returns the reduced graph and a mapping from new vertices to original vertices.
        """
        reduced_graph = nx.Graph()
        vertex_mapping = {}
        new_vertex_id = 0
        old_to_new = {}

        # Create new vertices for each critical clique
        for clique in self.critical_cliques:
            vertex_mapping[new_vertex_id] = clique
            for v in clique:
                old_to_new[v] = new_vertex_id
            reduced_graph.add_node(new_vertex_id)
            new_vertex_id += 1

        # Add edges between new vertices
        for i, clique_i in enumerate(self.critical_cliques):
            for j, clique_j in enumerate(self.critical_cliques):
                if i < j:
                    # Check if there's an edge between the cliques
                    edge_exists = False
                    for u in clique_i:
                        for v in clique_j:
                            if self.graph.has_edge(u, v):
                                edge_exists = True
                                break
                        if edge_exists:
                            break

                    if edge_exists:
                        reduced_graph.add_edge(
                            old_to_new[next(iter(clique_i))],
                            old_to_new[next(iter(clique_j))]
                        )

        return reduced_graph, vertex_mapping


class ClusterEditingKernelization:
    """
    Implements kernelization rules for the cluster editing problem.
    """

    def __init__(self, instance: ClusterEditingInstance):
        self.instance = instance.copy()
        self.reduction_history = []

    def apply_rule_1_heavy_non_edge(self) -> bool:
        """
        Rule 1: Heavy non-edge rule
        Set an edge uv with s(uv) < 0 to forbidden if |s(uv)| >= sum of s(uw) for w in N(u)
        Returns True if any reduction was applied.
        """
        applied = False
        edges_to_forbid = []

        for u in self.instance.graph.nodes():
            for v in self.instance.graph.nodes():
                if u < v:  # Consider each pair once
                    edge = (u, v)
                    weight = self.instance.weights.get(edge, 0)

                    if weight < 0:  # Non-edge with negative weight
                        u_sum = sum(
                            self.instance.weights.get((min(u, w), max(u, w)), 0)
                            for w in self.instance.graph.neighbors(u)
                        )

                        if abs(weight) >= u_sum:
                            edges_to_forbid.append(edge)
                            applied = True

        # Mark edges as forbidden (remove from graph if they exist)
        for edge in edges_to_forbid:
            if self.instance.graph.has_edge(*edge):
                self.instance.graph.remove_edge(*edge)
            # Set weight to negative infinity to mark as forbidden
            self.instance.weights[edge] = float('-inf')
            self.reduction_history.append(('forbid_edge', edge))

        return applied

    def apply_rule_2_heavy_edge_single(self) -> bool:
        """
        Rule 2: Heavy edge rule (single end)
        Merge vertices u, v of an edge uv if s(uv) >= sum of |s(uw)| for all w != u,v
        Returns True if any reduction was applied.
        """
        applied = False
        merges = []

        for edge in list(self.instance.graph.edges()):
            u, v = edge
            weight = self.instance.weights.get((min(u, v), max(u, v)), 0)

            if weight > 0:
                total_sum = sum(
                    abs(self.instance.weights.get((min(u, w), max(u, w)), 0))
                    for w in self.instance.graph.nodes()
                    if w != u and w != v
                )

                if weight >= total_sum:
                    merges.append((u, v))
                    applied = True

        # Perform merges
        for u, v in merges:
            if u in self.instance.graph and v in self.instance.graph:
                self._merge_vertices(u, v)

        return applied

    def apply_rule_3_heavy_edge_both(self) -> bool:
        """
        Rule 3: Heavy edge rule (both ends)
        Merge vertices u, v if s(uv) >= sum s(uw) for w in N(u)\{v} + sum s(vw) for w in N(v)\{u}
        Returns True if any reduction was applied.
        """
        applied = False
        merges = []

        for edge in list(self.instance.graph.edges()):
            u, v = edge
            weight = self.instance.weights.get((min(u, v), max(u, v)), 0)

            if weight > 0:
                u_sum = sum(
                    self.instance.weights.get((min(u, w), max(u, w)), 0)
                    for w in self.instance.graph.neighbors(u)
                    if w != v
                )
                v_sum = sum(
                    self.instance.weights.get((min(v, w), max(v, w)), 0)
                    for w in self.instance.graph.neighbors(v)
                    if w != u
                )

                if weight >= u_sum + v_sum:
                    merges.append((u, v))
                    applied = True

        # Perform merges
        for u, v in merges:
            if u in self.instance.graph and v in self.instance.graph:
                self._merge_vertices(u, v)

        return applied

    def _merge_vertices(self, u: int, v: int):
        """
        Merge vertex v into vertex u.
        Updates the graph structure and edge weights accordingly.
        """
        if u == v or v not in self.instance.graph:
            return

        # Update edges: for each neighbor of v, update edge to u
        for neighbor in list(self.instance.graph.neighbors(v)):
            if neighbor != u:
                # Calculate new weight
                edge_uv = (min(u, neighbor), max(u, neighbor))
                edge_vn = (min(v, neighbor), max(v, neighbor))

                new_weight = self.instance.weights.get(edge_uv, 0)
                new_weight += self.instance.weights.get(edge_vn, 0)

                self.instance.weights[edge_uv] = new_weight

                # Add edge if it doesn't exist
                if not self.instance.graph.has_edge(u, neighbor):
                    self.instance.graph.add_edge(u, neighbor)

        # Remove vertex v
        self.instance.graph.remove_node(v)

        # Clean up weights dictionary
        keys_to_remove = [
            edge for edge in self.instance.weights
            if v in edge
        ]
        for edge in keys_to_remove:
            del self.instance.weights[edge]

        self.reduction_history.append(('merge', u, v))

    def apply_critical_clique_reduction(self) -> bool:
        """
        Apply critical clique reduction: merge all vertices in each critical clique.
        Returns True if any reduction was applied.
        """
        cc = CriticalClique(self.instance.graph)

        if not cc.critical_cliques or all(len(c) == 1 for c in cc.critical_cliques):
            return False

        # Merge vertices in each critical clique
        for clique in cc.critical_cliques:
            if len(clique) > 1:
                clique_list = list(clique)
                base_vertex = clique_list[0]
                for v in clique_list[1:]:
                    if v in self.instance.graph:
                        self._merge_vertices(base_vertex, v)

        return True

    def kernelize(self, max_iterations: int = 100) -> ClusterEditingInstance:
        """
        Apply all reduction rules iteratively until no more reductions are possible.
        Returns the kernelized instance.
        """
        iteration = 0

        while iteration < max_iterations:
            applied = False

            # Apply critical clique reduction first (most powerful)
            if self.apply_critical_clique_reduction():
                applied = True

            # Apply weighted reduction rules
            if self.apply_rule_1_heavy_non_edge():
                applied = True

            if self.apply_rule_2_heavy_edge_single():
                applied = True

            if self.apply_rule_3_heavy_edge_both():
                applied = True

            if not applied:
                break

            iteration += 1

        return self.instance

    def get_kernel_size(self) -> Tuple[int, int]:
        """Returns (number of vertices, number of edges) in the kernel"""
        return (
            self.instance.graph.number_of_nodes(),
            self.instance.graph.number_of_edges()
        )


class ClusterEditingSolver:
    """
    Solves the cluster editing problem using kernelization and exact/heuristic algorithms.
    """

    def __init__(self, graph: nx.Graph, weights: Optional[Dict] = None):
        """
        Initialize solver with a graph and optional edge weights.
        If weights are not provided, unit weights are assumed.
        """
        self.original_graph = graph.copy()

        if weights is None:
            # Create unit weights: +1 for existing edges, -1 for non-edges
            weights = {}
            for u in graph.nodes():
                for v in graph.nodes():
                    if u < v:
                        if graph.has_edge(u, v):
                            weights[(u, v)] = 1.0
                        else:
                            weights[(u, v)] = -1.0

        self.instance = ClusterEditingInstance(graph, weights)
        self.kernel = None
        self.solution = None

    def solve_with_kernelization(self, use_ilp: bool = False) -> Dict[str, Any]:
        """
        Solve cluster editing with kernelization preprocessing.

        Args:
            use_ilp: If True, use ILP solver on the kernel (not implemented here)
                    If False, use greedy heuristic

        Returns:
            Dictionary with solution details
        """
        start_time = time.time()

        # Store original size
        original_n = self.original_graph.number_of_nodes()
        original_m = self.original_graph.number_of_edges()

        # Apply kernelization
        kernelizer = ClusterEditingKernelization(self.instance)
        self.kernel = kernelizer.kernelize()

        kernel_n, kernel_m = kernelizer.get_kernel_size()

        # Solve the kernel (simplified greedy approach for now)
        clusters = self._greedy_clustering(self.kernel)

        # Calculate editing cost
        editing_cost = self._calculate_editing_cost(self.kernel, clusters)

        elapsed_time = time.time() - start_time

        return {
            'original_nodes': original_n,
            'original_edges': original_m,
            'kernel_nodes': kernel_n,
            'kernel_edges': kernel_m,
            'reduction_ratio': 1 - (kernel_n / original_n) if original_n > 0 else 0,
            'num_clusters': len(clusters),
            'clusters': clusters,
            'editing_cost': editing_cost,
            'time_seconds': elapsed_time,
            'reduction_history': kernelizer.reduction_history[:10]  # First 10 reductions
        }

    def _greedy_clustering(self, instance: ClusterEditingInstance) -> List[Set[int]]:
        """
        Simple greedy clustering: form clusters based on positive weight edges.
        """
        clusters = []
        unassigned = set(instance.graph.nodes())

        while unassigned:
            # Start a new cluster with an arbitrary vertex
            v = unassigned.pop()
            cluster = {v}

            # Add vertices connected with positive weight
            candidates = list(unassigned)
            for u in candidates:
                edge = (min(u, v), max(u, v))
                if instance.weights.get(edge, 0) > 0:
                    # Check if u is positively connected to all in cluster
                    add_to_cluster = True
                    for w in cluster:
                        edge_uw = (min(u, w), max(u, w))
                        if instance.weights.get(edge_uw, 0) <= 0:
                            add_to_cluster = False
                            break

                    if add_to_cluster:
                        cluster.add(u)
                        unassigned.remove(u)

            clusters.append(cluster)

        return clusters

    def _calculate_editing_cost(self, instance: ClusterEditingInstance,
                                clusters: List[Set[int]]) -> float:
        """
        Calculate the cost of transforming the graph into the given clustering.
        """
        cost = 0.0

        for i, cluster in enumerate(clusters):
            # Cost of making the cluster complete
            for u in cluster:
                for v in cluster:
                    if u < v:
                        edge = (u, v)
                        if not instance.graph.has_edge(u, v):
                            # Need to add this edge
                            cost += abs(instance.weights.get(edge, 1))

        # Cost of removing edges between clusters
        for i, cluster_i in enumerate(clusters):
            for j, cluster_j in enumerate(clusters):
                if i < j:
                    for u in cluster_i:
                        for v in cluster_j:
                            if instance.graph.has_edge(u, v):
                                edge = (min(u, v), max(u, v))
                                cost += abs(instance.weights.get(edge, 1))

        return cost


def evaluate_kernelization(graph_path: str = None, graph: nx.Graph = None) -> Dict[str, Any]:
    """
    Evaluate the effectiveness of kernelization on a graph instance.

    Args:
        graph_path: Path to graph file (if provided)
        graph: NetworkX graph object (if provided)

    Returns:
        Evaluation results dictionary
    """
    if graph is None and graph_path is None:
        # Create a test graph with planted clusters and noise
        graph = create_test_instance(num_cliques=5, clique_size=10, noise=0.1)
    elif graph_path is not None:
        # Load graph from file (implementation depends on file format)
        pass  # Placeholder for loading logic

    solver = ClusterEditingSolver(graph)
    results = solver.solve_with_kernelization()

    # Add evaluation metrics
    results['kernelization_effective'] = results['reduction_ratio'] > 0.5
    results['kernel_small_enough'] = results['kernel_nodes'] < 100

    return results


def create_test_instance(num_cliques: int = 3, clique_size: int = 5,
                         noise: float = 0.1) -> nx.Graph:
    """
    Create a test graph with planted cliques and noise.

    Args:
        num_cliques: Number of cliques to create
        clique_size: Size of each clique
        noise: Probability of flipping edges (adding/removing)

    Returns:
        NetworkX graph with planted structure
    """
    import random

    graph = nx.Graph()
    vertex_id = 0

    # Create disjoint cliques
    for _ in range(num_cliques):
        clique_vertices = list(range(vertex_id, vertex_id + clique_size))
        for i, u in enumerate(clique_vertices):
            for v in clique_vertices[i + 1:]:
                graph.add_edge(u, v)
        vertex_id += clique_size

    # Add noise
    all_vertices = list(graph.nodes())
    for u in all_vertices:
        for v in all_vertices:
            if u < v and random.random() < noise:
                if graph.has_edge(u, v):
                    graph.remove_edge(u, v)
                else:
                    graph.add_edge(u, v)

    return graph


# Example usage and testing
if __name__ == "__main__":
    # Test on a small example
    print("=" * 60)
    print("WP3: Cluster Editing Kernelization Test")
    print("=" * 60)

    # Create test instance
    test_graph = create_test_instance(num_cliques=4, clique_size=8, noise=0.15)

    print(f"\nOriginal graph:")
    print(f"  Nodes: {test_graph.number_of_nodes()}")
    print(f"  Edges: {test_graph.number_of_edges()}")

    # Solve with kernelization
    solver = ClusterEditingSolver(test_graph)
    results = solver.solve_with_kernelization()

    print(f"\nKernelization results:")
    print(f"  Kernel nodes: {results['kernel_nodes']} ({results['reduction_ratio']:.1%} reduction)")
    print(f"  Kernel edges: {results['kernel_edges']}")
    print(f"  Number of clusters: {results['num_clusters']}")
    print(f"  Editing cost: {results['editing_cost']:.2f}")
    print(f"  Time: {results['time_seconds']:.3f} seconds")

    print(f"\nFirst few reduction operations:")
    for op in results['reduction_history'][:5]:
        print(f"  - {op}")

    # Performance evaluation
    print("\n" + "=" * 60)
    print("Performance Evaluation on Different Instances")
    print("=" * 60)

    test_cases = [
        (3, 5, 0.05, "Low noise"),
        (5, 10, 0.1, "Medium noise"),
        (4, 8, 0.2, "High noise"),
        (10, 5, 0.1, "Many small cliques"),
        (2, 25, 0.1, "Few large cliques")
    ]

    for num_c, size_c, noise, description in test_cases:
        graph = create_test_instance(num_c, size_c, noise)
        results = evaluate_kernelization(graph=graph)

        print(f"\n{description} (n={num_c}, s={size_c}, noise={noise}):")
        print(f"  Original: {results['original_nodes']} nodes, {results['original_edges']} edges")
        print(f"  Kernel: {results['kernel_nodes']} nodes ({results['reduction_ratio']:.1%} reduction)")
        print(f"  Clusters found: {results['num_clusters']}")
        print(f"  Effective: {'Yes' if results['kernelization_effective'] else 'No'}")