# src/algorithms/cluster_editing_solver.py
import time
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Tuple, Union

import networkx as nx
import numpy as np

from src.algorithms.cluster_editing_kernelization import (
    AdvancedClusterEditingInstance as ClusterEditingInstance,
    AdvancedKernelization as ClusterEditingKernelization,
)
from src.utils import txt_to_networkx
from src.algorithms.cluster_editing_ilp import solve_cluster_editing_ilp


class ClusterEditingSolver:
    """
    Main solver for cluster editing problem using (optional) kernelization + ILP..
    """

    def __init__(self, graph: nx.Graph, weights: Optional[Dict] = None):
        self.original_graph = graph.copy()

        if weights is None:
            weights = self._init_unit_weights(graph)

        self.instance = ClusterEditingInstance(graph.copy(), weights)
        self.kernel: Optional[ClusterEditingInstance] = None
        self.solution: Optional[Dict[str, Any]] = None

    def _init_unit_weights(self, graph: nx.Graph) -> Dict[Tuple[int, int], float]:
        """+1 für Kante, -1 für Nicht-Kante."""
        weights: Dict[Tuple[int, int], float] = {}
        nodes = list(graph.nodes())
        edges_set = set((min(u, v), max(u, v)) for u, v in graph.edges())

        for i, u in enumerate(nodes):
            for v in nodes[i + 1:]:
                e = (min(u, v), max(u, v))
                weights[e] = 1.0 if e in edges_set else -1.0
        return weights

    def _ensure_complete_weights(self, instance: ClusterEditingInstance) -> Dict[Tuple[int, int], float]:
        """Sorge dafür, dass für jedes Paar (i<j) ein Gewicht existiert."""
        w = dict(instance.weights)
        nodes = list(instance.graph.nodes())
        edges_set = set((min(u, v), max(u, v)) for u, v in instance.graph.edges())
        for i, u in enumerate(nodes):
            for v in nodes[i + 1:]:
                e = (min(u, v), max(u, v))
                if e not in w:
                    w[e] = 1.0 if e in edges_set else -1.0
        return w

    def solve(self,
              use_kernelization: bool = True,
              kernelization_type: str = 'optimized',
              clustering_algorithm: str = 'ilp',
              time_limit: Optional[float] = None,
              mip_gap: Optional[float] = None,
              threads: Optional[int] = None,
              gurobi_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Löse Cluster Editing (immer via ILP). Kernelization optional.
        """
        start_time = time.time()

        original_stats = {
            'nodes': self.original_graph.number_of_nodes(),
            'edges': self.original_graph.number_of_edges()
        }

        # Kernelization (optional)
        if use_kernelization:
            kernelizer = ClusterEditingKernelization(
                self.instance,
                use_preprocessing=True,
                use_smart_ordering=True
            )
            self.kernel = kernelizer.kernelize()
            kernel_stats = kernelizer.get_comprehensive_stats()
            instance_to_solve = self.kernel
        else:
            self.kernel = self.instance
            kernel_stats = None
            instance_to_solve = self.instance

        # ILP lösen
        weights_complete = self._ensure_complete_weights(instance_to_solve)
        clusters, ilp_stats = solve_cluster_editing_ilp(
            instance_to_solve.graph,
            weights_complete,
            time_limit=time_limit,
            mip_gap=mip_gap,
            threads=threads,
            gurobi_params=gurobi_params,
        )

        # Lifting & faire Kostenauswertung auf dem Original
        if use_kernelization:
            clusters_lifted = self._lift_clusters_to_original(instance_to_solve, clusters)
            editing_cost = self.calculate_editing_cost(self.instance, clusters_lifted)
            clusters_to_report = clusters_lifted
        else:
            editing_cost = self.calculate_editing_cost(instance_to_solve, clusters)
            clusters_to_report = clusters

        # Kosten berechnen
        editing_cost = self.calculate_editing_cost(instance_to_solve, clusters)

        elapsed_time = time.time() - start_time
        self.solution = {
            'clusters': clusters_to_report,
            'num_clusters': len(clusters_to_report),
            'editing_cost': editing_cost,
            'time_seconds': elapsed_time,
            'original_stats': original_stats,
            'kernel_stats': kernel_stats,
            'algorithm': 'ilp',
            'use_kernelization': use_kernelization,
            'ilp': ilp_stats,
        }

        return self.solution

    def _greedy_clustering(self, instance: ClusterEditingInstance) -> List[Set[int]]:
        """(derzeit nicht mehr verwendet – belassen für ggf. spätere Tests)"""
        clusters: List[Set[int]] = []
        unassigned = set(instance.graph.nodes())
        while unassigned:
            v = unassigned.pop()
            cluster = {v}
            for u in list(unassigned):
                if sum(instance.get_weight(u, w) for w in cluster) > 0:
                    cluster.add(u)
                    unassigned.remove(u)
            clusters.append(cluster)
        return clusters

    def calculate_editing_cost(self, instance: ClusterEditingInstance,
                               clusters: List[Set[int]]) -> float:
        """Kosten, um Graph in gegebene Clusterung zu transformieren."""
        cost = 0.0

        # Innerhalb der Cluster fehlende Kanten hinzufügen
        for cluster in clusters:
            cl = list(cluster)
            for i, u in enumerate(cl):
                for v in cl[i + 1:]:
                    if not instance.graph.has_edge(u, v):
                        cost += abs(instance.get_weight(u, v))

        # Zwischen Clustern vorhandene Kanten entfernen
        for i, ci in enumerate(clusters):
            for cj in clusters[i + 1:]:
                for u in ci:
                    for v in cj:
                        if instance.graph.has_edge(u, v):
                            cost += abs(instance.get_weight(u, v))

        return cost

    def _lift_clusters_to_original(self, kernel_instance, clusters_on_kernel: List[Set[int]]) -> List[Set[int]]:
        """
        Mappt Cluster aus dem Kernel zurück auf Originalknoten via kernel_instance.supernode_members.
        Fällt zurück auf Identität, falls Mapping fehlt.
        """
        lifted: List[Set[int]] = []
        mapping = getattr(kernel_instance, "supernode_members", None)
        if not mapping:
            return [set(C) for C in clusters_on_kernel]
        for C in clusters_on_kernel:
            S: Set[int] = set()
            for s in C:
                S.update(mapping.get(s, {s}))
            lifted.append(S)
        return lifted

    def get_solution_quality_metrics(self) -> Dict[str, Any]:
        if not self.solution:
            return {}
        clusters = self.solution['clusters']
        cluster_sizes = [len(c) for c in clusters]

        intra_edges = sum(
            sum(1 for u in c for v in c if u < v and self.kernel.graph.has_edge(u, v))
            for c in clusters
        )
        inter_edges = sum(
            sum(1 for u in ci for v in cj if self.kernel.graph.has_edge(u, v))
            for i, ci in enumerate(clusters)
            for cj in clusters[i + 1:]
        )
        total_possible_intra = sum(len(c) * (len(c) - 1) // 2 for c in clusters)

        return {
            'num_clusters': len(clusters),
            'avg_cluster_size': float(np.mean(cluster_sizes)) if cluster_sizes else 0.0,
            'std_cluster_size': float(np.std(cluster_sizes)) if cluster_sizes else 0.0,
            'min_cluster_size': min(cluster_sizes) if cluster_sizes else 0,
            'max_cluster_size': max(cluster_sizes) if cluster_sizes else 0,
            'singleton_clusters': sum(1 for c in clusters if len(c) == 1),
            'intra_cluster_edges': intra_edges,
            'inter_cluster_edges': inter_edges,
            'intra_cluster_density': (intra_edges / total_possible_intra) if total_possible_intra > 0 else 0.0,
            'editing_cost': self.solution.get('editing_cost', 0.0)
        }

    @classmethod
    def from_txt(cls, txt_path: Union[str, Path]):
        G = txt_to_networkx(str(Path(txt_path)))
        return cls(G, weights=None)
