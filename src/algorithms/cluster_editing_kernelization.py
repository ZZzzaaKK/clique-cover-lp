"""
Core implementation of cluster editing kernelization algorithms.

This module provides:
- Core-algorithms for kernelisation
- ClusterEditingInstance: Data structure for instances
- CriticalClique: Critical clique detection
- ClusterEditingKernelization: Kernelization rules
- OptimizedClusterEditingKernelization: Performance-optimized version

References:
- Böcker & Baumbach (2013): Cluster editing
- Cao & Chen (2012): Cluster editing: Kernelization based on edge cuts
- Chen & Meng (2012): A 2k kernel for the cluster editing problem
"""

import networkx as nx
import numpy as np
from typing import Dict, Tuple, Set, List, Optional, FrozenSet, Any
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class ClusterEditingInstance:
    """
    Represents a weighted cluster editing instance.

    Attributes:
        graph: NetworkX graph
        weights: Edge weights dictionary
        k: Optional parameter budget for modifications
    """
    graph: nx.Graph
    weights: Dict[Tuple[int, int], float] = field(default_factory=dict)
    k: Optional[float] = None

    # Cached computations for performance
    _neighbor_sums: Dict[int, float] = field(default_factory=dict, init=False)
    _weight_matrix: Optional[np.ndarray] = field(default=None, init=False)
    _use_matrix: bool = field(default=False, init=False)
    _node_to_idx: Dict[int, int] = field(default_factory=dict, init=False)

    def __post_init__(self):
        """Initialize cached structures based on graph density."""
        n = self.graph.number_of_nodes()
        m = self.graph.number_of_edges()

        # Use matrix representation for dense graphs (>25% density)
        if n > 0 and m > 0.25 * n * (n - 1) / 2:
            self._use_matrix = True
            self._init_weight_matrix()

        self._normalize_weights()
        self._update_neighbor_sums()

    def _init_weight_matrix(self):
        """Initialize weight matrix for dense graphs."""
        nodes = sorted(self.graph.nodes())
        n = len(nodes)
        self._weight_matrix = np.zeros((n, n))
        self._node_to_idx = {node: i for i, node in enumerate(nodes)}

        for (u, v), w in self.weights.items():
            if u in self._node_to_idx and v in self._node_to_idx:
                i, j = self._node_to_idx[u], self._node_to_idx[v]
                self._weight_matrix[i, j] = w
                self._weight_matrix[j, i] = w

    def _normalize_weights(self):
        """Ensure all edges use (min, max) convention."""
        normalized = {}
        for edge, weight in self.weights.items():
            u, v = edge
            normalized[(min(u, v), max(u, v))] = weight
        self.weights = normalized

    def _update_neighbor_sums(self, nodes: Optional[Set[int]] = None):
        """Update cached neighbor sums for specified nodes."""
        if nodes is None:
            nodes = self.graph.nodes()

        for u in nodes:
            self._neighbor_sums[u] = sum(
                self.get_weight(u, v)
                for v in self.graph.neighbors(u)
            )

    def get_weight(self, u: int, v: int) -> float:
        """Get edge weight efficiently."""
        if u == v:
            return 0

        if self._use_matrix and hasattr(self, '_node_to_idx'):
            if u in self._node_to_idx and v in self._node_to_idx:
                return self._weight_matrix[self._node_to_idx[u], self._node_to_idx[v]]

        return self.weights.get((min(u, v), max(u, v)), 0)

    def set_weight(self, u: int, v: int, weight: float):
        """Set edge weight with cache updates."""
        if u == v:
            return

        edge = (min(u, v), max(u, v))
        old_weight = self.weights.get(edge, 0)
        self.weights[edge] = weight

        if self._use_matrix and hasattr(self, '_node_to_idx'):
            if u in self._node_to_idx and v in self._node_to_idx:
                i, j = self._node_to_idx[u], self._node_to_idx[v]
                self._weight_matrix[i, j] = weight
                self._weight_matrix[j, i] = weight

        # Update affected neighbor sums
        if u in self.graph and v in self.graph and self.graph.has_edge(u, v):
            diff = weight - old_weight
            if u in self._neighbor_sums:
                self._neighbor_sums[u] += diff
            if v in self._neighbor_sums:
                self._neighbor_sums[v] += diff

    def get_neighbor_sum(self, u: int) -> float:
        """Get cached neighbor sum for vertex u."""
        return self._neighbor_sums.get(u, 0)

    def copy(self) -> 'ClusterEditingInstance':
        """Create a deep copy of the instance."""
        new_instance = ClusterEditingInstance(
            graph=self.graph.copy(),
            weights=self.weights.copy(),
            k=self.k
        )
        new_instance._neighbor_sums = self._neighbor_sums.copy()
        if self._weight_matrix is not None:
            new_instance._weight_matrix = self._weight_matrix.copy()
            new_instance._node_to_idx = self._node_to_idx.copy()
        new_instance._use_matrix = self._use_matrix
        return new_instance

    def get_statistics(self) -> Dict[str, Any]:
        """Get instance statistics."""
        return {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'uses_matrix': self._use_matrix,
            'num_weights': len(self.weights)
        }


class CriticalClique:
    """
    Manages critical cliques in a graph.

    A critical clique is a maximal clique where all vertices
    have identical closed neighborhoods.
    """

    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.critical_cliques: List[Set[int]] = []
        self.vertex_to_clique: Dict[int, int] = {}
        self._find_critical_cliques()

    def _get_closed_neighborhood(self, v: int) -> FrozenSet[int]:
        """Get hashable closed neighborhood of vertex v."""
        return frozenset(self.graph.neighbors(v)) | {v}

    def _find_critical_cliques(self):
        """Find all critical cliques using neighborhood hashing - O(n*m) complexity."""
        # Group vertices by closed neighborhood
        neighborhood_groups = defaultdict(set)

        for v in self.graph.nodes():
            closed_nbh = self._get_closed_neighborhood(v)
            neighborhood_groups[closed_nbh].add(v)

        # Verify each group forms a clique and store
        clique_id = 0
        for vertices in neighborhood_groups.values():
            if len(vertices) > 0 and self._is_clique(vertices):
                self.critical_cliques.append(vertices)
                for v in vertices:
                    self.vertex_to_clique[v] = clique_id
                clique_id += 1

    def _is_clique(self, vertices: Set[int]) -> bool:
        """Verify if vertices form a clique - O(|vertices|^2)."""
        vertices_list = list(vertices)
        n = len(vertices_list)

        if n <= 1:
            return True

        # Count edges - should be n*(n-1)/2 for complete clique
        edge_count = sum(
            1 for i in range(n) for j in range(i + 1, n)
            if self.graph.has_edge(vertices_list[i], vertices_list[j])
        )

        return edge_count == n * (n - 1) // 2

    def get_clique_sizes(self) -> List[int]:
        """Get sizes of all critical cliques."""
        return [len(clique) for clique in self.critical_cliques]


class ClusterEditingKernelization:
    """
    Implements kernelization rules for cluster editing.

    Provides three main reduction rules:
    1. Heavy non-edge rule
    2. Heavy edge rule (single end)
    3. Heavy edge rule (both ends)
    """

    def __init__(self, instance: ClusterEditingInstance):
        """
        Initialize kernelization with a cluster editing instance.

        Args:
            instance: ClusterEditingInstance to be reduced
        """
        self.instance = instance.copy()
        self.reduction_history: List[Tuple] = []
        self.stats = {
            'rules_applied': defaultdict(int),
            'vertices_removed': 0,
            'edges_modified': 0,
            'initial_size': (instance.graph.number_of_nodes(),
                             instance.graph.number_of_edges())
        }

    def apply_rule_1_heavy_non_edge(self) -> bool:
        """
        Rule 1: Heavy non-edge rule.
        Set an edge uv with s(uv) < 0 to forbidden if |s(uv)| >= sum of s(uw) for w in N(u).

        Returns:
            True if any reduction was applied
        """
        applied = False
        edges_to_forbid = []

        for u in self.instance.graph.nodes():
            u_sum = self.instance.get_neighbor_sum(u)

            for v in self.instance.graph.nodes():
                if u < v:
                    weight = self.instance.get_weight(u, v)

                    if weight < 0 and abs(weight) >= u_sum:
                        edges_to_forbid.append((u, v))
                        applied = True

        # Apply modifications
        for u, v in edges_to_forbid:
            if self.instance.graph.has_edge(u, v):
                self.instance.graph.remove_edge(u, v)
                self.stats['edges_modified'] += 1

            self.instance.set_weight(u, v, float('-inf'))
            self.reduction_history.append(('forbid_edge', u, v))
            self.stats['rules_applied']['heavy_non_edge'] += 1

        return applied

    def apply_rule_2_heavy_edge_single(self) -> bool:
        """
        Rule 2: Heavy edge rule (single end).
        Merge vertices u, v if s(uv) >= sum of |s(uw)| for all w != u,v.

        Returns:
            True if any reduction was applied
        """
        applied = False
        merges = []

        for edge in list(self.instance.graph.edges()):
            u, v = edge
            weight = self.instance.get_weight(u, v)

            if weight > 0:
                total_sum = sum(
                    abs(self.instance.get_weight(u, w)) +
                    abs(self.instance.get_weight(v, w))
                    for w in self.instance.graph.nodes()
                    if w != u and w != v
                ) / 2  # Divided by 2 as we count each weight twice

                if weight >= total_sum:
                    merges.append((u, v))
                    applied = True

        # Apply merges
        for u, v in merges:
            if u in self.instance.graph and v in self.instance.graph:
                self._merge_vertices(u, v)
                self.stats['rules_applied']['heavy_edge_single'] += 1

        return applied

    def apply_rule_3_heavy_edge_both(self) -> bool:
        """
        Rule 3: Heavy edge rule (both ends).
        Merge vertices u, v if s(uv) >= sum s(uw) for w in N(u)\{v} + sum s(vw) for w in N(v)\{u}.

        Returns:
            True if any reduction was applied
        """
        applied = False
        merges = []

        for edge in list(self.instance.graph.edges()):
            u, v = edge
            weight = self.instance.get_weight(u, v)

            if weight > 0:
                u_sum = sum(
                    self.instance.get_weight(u, w)
                    for w in self.instance.graph.neighbors(u)
                    if w != v
                )
                v_sum = sum(
                    self.instance.get_weight(v, w)
                    for w in self.instance.graph.neighbors(v)
                    if w != u
                )

                if weight >= u_sum + v_sum:
                    merges.append((u, v))
                    applied = True

        # Apply merges
        for u, v in merges:
            if u in self.instance.graph and v in self.instance.graph:
                self._merge_vertices(u, v)
                self.stats['rules_applied']['heavy_edge_both'] += 1

        return applied

    def _merge_vertices(self, u: int, v: int):
        """
        Merge vertex v into vertex u.

        Args:
            u: Target vertex (kept)
            v: Source vertex (removed)
        """
        if u == v or v not in self.instance.graph:
            return

        # Update edges and weights
        for neighbor in list(self.instance.graph.neighbors(v)):
            if neighbor != u:
                edge_un = (min(u, neighbor), max(u, neighbor))
                edge_vn = (min(v, neighbor), max(v, neighbor))

                new_weight = self.instance.weights.get(edge_un, 0)
                new_weight += self.instance.weights.get(edge_vn, 0)

                self.instance.weights[edge_un] = new_weight

                if not self.instance.graph.has_edge(u, neighbor):
                    self.instance.graph.add_edge(u, neighbor)

        # Remove vertex v
        self.instance.graph.remove_node(v)

        # Clean up weights
        self.instance.weights = {
            edge: w for edge, w in self.instance.weights.items()
            if v not in edge
        }

        # Update caches
        affected = {u} | set(self.instance.graph.neighbors(u))
        self.instance._update_neighbor_sums(affected)

        self.reduction_history.append(('merge', u, v))
        self.stats['vertices_removed'] += 1

    def apply_critical_clique_reduction(self) -> bool:
        """
        Apply critical clique reduction.
        Merges all vertices within each critical clique.

        Returns:
            True if any reduction was applied
        """
        cc = CriticalClique(self.instance.graph)

        if not cc.critical_cliques or all(len(c) == 1 for c in cc.critical_cliques):
            return False

        applied = False
        for clique in cc.critical_cliques:
            if len(clique) > 1:
                clique_list = list(clique)
                base_vertex = clique_list[0]

                for v in clique_list[1:]:
                    if v in self.instance.graph:
                        self._merge_vertices(base_vertex, v)
                        self.stats['rules_applied']['critical_clique'] += 1
                        applied = True

        return applied

    def kernelize(self, max_iterations: int = 100) -> ClusterEditingInstance:
        """
        Apply all reduction rules iteratively until fixpoint.

        Args:
            max_iterations: Maximum number of iterations

        Returns:
            Kernelized ClusterEditingInstance
        """
        iteration = 0

        while iteration < max_iterations:
            applied = False

            # Apply rules in order of effectiveness
            if self.apply_critical_clique_reduction():
                applied = True

            if self.apply_rule_1_heavy_non_edge():
                applied = True

            if self.apply_rule_2_heavy_edge_single():
                applied = True

            if self.apply_rule_3_heavy_edge_both():
                applied = True

            if not applied:
                break

            iteration += 1

        self.stats['iterations'] = iteration
        return self.instance

    def get_kernel_statistics(self) -> Dict[str, Any]:
        """
        Get detailed statistics about the kernelization.

        Returns:
            Dictionary with kernelization statistics
        """
        current_size = (self.instance.graph.number_of_nodes(),
                        self.instance.graph.number_of_edges())

        return {
            'initial_nodes': self.stats['initial_size'][0],
            'initial_edges': self.stats['initial_size'][1],
            'kernel_nodes': current_size[0],
            'kernel_edges': current_size[1],
            'reduction_ratio': 1 - (current_size[0] / self.stats['initial_size'][0])
            if self.stats['initial_size'][0] > 0 else 0,
            'rules_applied': dict(self.stats['rules_applied']),
            'vertices_removed': self.stats['vertices_removed'],
            'edges_modified': self.stats['edges_modified'],
            'iterations': self.stats.get('iterations', 0),
            'reduction_history_size': len(self.reduction_history)
        }


class OptimizedClusterEditingKernelization(ClusterEditingKernelization):
    """
    Performance-optimized version with batch processing and better algorithms.

    Improvements:
    - Batch application of rules
    - Union-find for merges
    - Early termination
    - Better caching
    """

    def apply_all_rules_batch(self) -> bool:
        """
        Apply all rules in batch mode for better performance.

        Returns:
            True if any modifications were made
        """
        modifications = {
            'merges': [],
            'forbidden_edges': []
        }

        # Collect all modifications
        self._collect_heavy_non_edges(modifications)
        self._collect_heavy_edges(modifications)

        # Apply in optimal order
        return self._apply_modifications_batch(modifications)

    def _collect_heavy_non_edges(self, mods: Dict):
        """Collect heavy non-edges to forbid."""
        for u in self.instance.graph.nodes():
            u_sum = self.instance.get_neighbor_sum(u)

            if u_sum <= 0:
                continue

            for v in self.instance.graph.nodes():
                if u < v:
                    weight = self.instance.get_weight(u, v)
                    if weight < 0 and abs(weight) >= u_sum:
                        mods['forbidden_edges'].append((u, v))

    def _collect_heavy_edges(self, mods: Dict):
        """Collect heavy edges for merging (combines rules 2 and 3)."""
        for u, v in self.instance.graph.edges():
            weight = self.instance.get_weight(u, v)

            if weight <= 0:
                continue

            # Check rule 2
            nodes = set(self.instance.graph.nodes())
            other_sum = sum(
                abs(self.instance.get_weight(u, w)) +
                abs(self.instance.get_weight(v, w))
                for w in nodes if w != u and w != v
            ) / 2

            if weight >= other_sum:
                mods['merges'].append((u, v, weight, 'rule2'))
                continue

            # Check rule 3
            u_sum = sum(
                self.instance.get_weight(u, w)
                for w in self.instance.graph.neighbors(u)
                if w != v
            )
            v_sum = sum(
                self.instance.get_weight(v, w)
                for w in self.instance.graph.neighbors(v)
                if w != u
            )

            if weight >= u_sum + v_sum:
                mods['merges'].append((u, v, weight, 'rule3'))

    def _apply_modifications_batch(self, mods: Dict) -> bool:
        """Apply collected modifications efficiently."""
        applied = False

        # Forbid edges
        for u, v in mods['forbidden_edges']:
            if self.instance.graph.has_edge(u, v):
                self.instance.graph.remove_edge(u, v)
            self.instance.set_weight(u, v, float('-inf'))
            self.stats['rules_applied']['heavy_non_edge'] += 1
            applied = True

        # Apply merges using union-find
        if mods['merges']:
            self._apply_merges_unionfind(mods['merges'])
            applied = True

        return applied

    def _apply_merges_unionfind(self, merges: List[Tuple]):
        """Apply merges efficiently using union-find structure."""
        # Sort by weight (highest first)
        merges.sort(key=lambda x: x[2], reverse=True)

        # Union-find structure
        parent = {}

        def find(x):
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[py] = px
                return True
            return False

        # Apply merges
        for merge_data in merges:
            u, v = merge_data[0], merge_data[1]
            rule = merge_data[3] if len(merge_data) > 3 else 'unknown'

            if u in self.instance.graph and v in self.instance.graph:
                if union(u, v):
                    self._merge_vertices(u, v)
                    self.stats['rules_applied'][f'heavy_edge_{rule}'] += 1

    def kernelize(self, max_iterations: int = 50) -> ClusterEditingInstance:
        """
        Optimized kernelization with early termination.

        Args:
            max_iterations: Maximum iterations

        Returns:
            Kernelized instance
        """
        iteration = 0
        last_size = (self.instance.graph.number_of_nodes(),
                     self.instance.graph.number_of_edges())
        no_change_count = 0

        while iteration < max_iterations:
            # Critical cliques first
            cc_applied = self.apply_critical_clique_reduction()

            # Batch application
            rules_applied = self.apply_all_rules_batch()

            current_size = (self.instance.graph.number_of_nodes(),
                            self.instance.graph.number_of_edges())

            # Check convergence
            if not cc_applied and not rules_applied:
                break

            if current_size == last_size:
                no_change_count += 1
                if no_change_count >= 2:
                    break
            else:
                no_change_count = 0
                last_size = current_size

            iteration += 1

        self.stats['iterations'] = iteration
        return self.instance