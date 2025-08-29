# src/algorithms/cluster_editing_solver_fixed.py
import time
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Tuple, Union
import tracemalloc
import pandas as pd
import networkx as nx
import numpy as np

from src.algorithms.cluster_editing_kernelization import (
    AdvancedClusterEditingInstance,
    AdvancedKernelization,
)
from src.algorithms.cluster_editing_ilp import (
    solve_cluster_editing_ilp,
    validate_clustering,
    calculate_clustering_cost,
)
from src.utils import txt_to_networkx

logger = logging.getLogger(__name__)


class ClusterEditingSolver:
    """
    Production-ready solver for cluster editing problem using kernelization + ILP.

    This solver implements the complete approach from Böcker et al. with:
    - Correct reduction rules including min-cut and DP
    - 2-partition inequalities in ILP
    - Proper error handling and validation
    - Memory tracking and limits
    """

    def __init__(self,
                 graph: nx.Graph,
                 weights: Optional[Dict[Tuple[int, int], float]] = None,
                 memory_limit_mb: int = 4096):
        """
        Initialize solver.

        Args:
            graph: Input graph
            weights: Edge weights (None = unit weights)
            memory_limit_mb: Memory limit in MB
        """
        if graph is None or graph.number_of_nodes() == 0:
            raise ValueError("Invalid input graph")

        self.original_graph = graph.copy()
        self.memory_limit = memory_limit_mb * 1024 * 1024  # Convert to bytes

        if weights is None:
            weights = self._init_unit_weights(graph)
        else:
            weights = self._validate_weights(weights)

        self.instance = AdvancedClusterEditingInstance(graph.copy(), weights)
        self.kernel: Optional[AdvancedClusterEditingInstance] = None
        self.solution: Optional[Dict[str, Any]] = None

    def _init_unit_weights(self, graph: nx.Graph) -> Dict[Tuple[int, int], float]:
        """Initialize unit weights: +1 for edges, -1 for non-edges."""
        weights: Dict[Tuple[int, int], float] = {}
        nodes = list(graph.nodes())
        edges_set = set((min(u, v), max(u, v)) for u, v in graph.edges())

        for i, u in enumerate(nodes):
            for v in nodes[i + 1:]:
                e = (min(u, v), max(u, v))
                weights[e] = 1.0 if e in edges_set else -1.0
        return weights

    def _validate_weights(self, weights: Dict[Tuple[int, int], float]) -> Dict[Tuple[int, int], float]:
        """Validate and sanitize edge weights."""
        validated = {}
        for edge, weight in weights.items():
            if not isinstance(edge, tuple) or len(edge) != 2:
                logger.warning(f"Invalid edge format: {edge}")
                continue

            # Ensure proper edge format (min, max)
            u, v = edge
            if u > v:
                u, v = v, u

            # Validate weight
            if not np.isfinite(weight):
                logger.warning(f"Non-finite weight {weight} for edge ({u},{v}), using 0")
                weight = 0.0

            validated[(u, v)] = weight

        return validated

    def _ensure_complete_weights(self, instance: AdvancedClusterEditingInstance) -> Dict[Tuple[int, int], float]:
        """Ensure weights exist for all node pairs."""
        w = dict(instance.weights)
        nodes = list(instance.graph.nodes())
        edges_set = set((min(u, v), max(u, v)) for u, v in instance.graph.edges())

        for i, u in enumerate(nodes):
            for v in nodes[i + 1:]:
                e = (min(u, v), max(u, v))
                if e not in w:
                    w[e] = 1.0 if e in edges_set else -1.0
        return w

    def _check_memory_usage(self):
        """Check current memory usage against limit."""
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            if current > self.memory_limit:
                raise MemoryError(f"Memory usage {current / 1e6:.1f}MB exceeds limit {self.memory_limit / 1e6:.1f}MB")

    def solve(self,
              use_kernelization: bool = True,
              kernelization_type: str = 'optimized',
              time_limit: Optional[float] = None,
              mip_gap: Optional[float] = None,
              threads: Optional[int] = None,
              gurobi_params: Optional[Dict[str, Any]] = None,
              use_2partition: bool = True,
              track_memory: bool = False) -> Dict[str, Any]:
        """
        Solve Cluster Editing with all fixes applied.

        Args:
            use_kernelization: Whether to use kernelization
            kernelization_type: Type of kernelization ('standard' or 'optimized')
            time_limit: Time limit in seconds
            mip_gap: MIP gap tolerance
            threads: Number of threads for ILP
            gurobi_params: Additional Gurobi parameters
            use_2partition: Whether to use 2-partition inequalities
            track_memory: Whether to track memory usage

        Returns:
            Solution dictionary with clusters, cost, statistics, etc.
        """
        start_time = time.time()

        # Start memory tracking if requested
        if track_memory:
            tracemalloc.start()

        try:
            original_stats = {
                'nodes': self.original_graph.number_of_nodes(),
                'edges': self.original_graph.number_of_edges(),
                'density': nx.density(self.original_graph),
            }

            # Apply kernelization if requested
            kernel_stats = None
            if use_kernelization:
                logger.info("Applying kernelization...")
                self._check_memory_usage() if track_memory else None

                kernelizer = AdvancedKernelization(
                    self.instance,
                    use_preprocessing=True,
                    use_smart_ordering=(kernelization_type == 'optimized'),
                    use_parallel=True
                )
                self.kernel = kernelizer.kernelize()
                kernel_stats = kernelizer.get_comprehensive_stats()
                instance_to_solve = self.kernel

                logger.info(f"Kernelization reduced {original_stats['nodes']} -> "
                            f"{instance_to_solve.graph.number_of_nodes()} nodes")
            else:
                self.kernel = self.instance
                instance_to_solve = self.instance

            # Ensure complete weights for ILP
            weights_complete = self._ensure_complete_weights(instance_to_solve)

            # Check memory before ILP
            self._check_memory_usage() if track_memory else None

            # Solve using ILP with 2-partition inequalities
            logger.info("Solving ILP...")
            clusters, ilp_stats = solve_cluster_editing_ilp(
                instance_to_solve.graph,
                weights_complete,
                time_limit=time_limit,
                mip_gap=mip_gap,
                threads=threads,
                gurobi_params=gurobi_params,
                use_2partition=use_2partition,
            )

            # Lift clusters back to original graph if kernelized
            if use_kernelization:
                clusters_lifted = self._lift_clusters_to_original(instance_to_solve, clusters)
                clusters_to_report = clusters_lifted
            else:
                clusters_to_report = clusters

            # Validate clustering
            if not validate_clustering(self.original_graph, clusters_to_report):
                raise ValueError("Invalid clustering produced")

            # Calculate editing cost on original graph
            original_weights = self._ensure_complete_weights(self.instance)
            editing_cost = calculate_clustering_cost(
                self.original_graph,
                original_weights,
                clusters_to_report
            )

            # Get memory statistics if tracking
            memory_stats = None
            if track_memory:
                current, peak = tracemalloc.get_traced_memory()
                memory_stats = {
                    'peak_memory_mb': peak / 1e6,
                    'current_memory_mb': current / 1e6,
                }
                tracemalloc.stop()

            elapsed_time = time.time() - start_time

            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(
                self.original_graph,
                clusters_to_report
            )

            self.solution = {
                'clusters': clusters_to_report,
                'num_clusters': len(clusters_to_report),
                'editing_cost': editing_cost,
                'time_seconds': elapsed_time,
                'original_stats': original_stats,
                'kernel_stats': kernel_stats,
                'algorithm': 'ilp_2partition' if use_2partition else 'ilp',
                'use_kernelization': use_kernelization,
                'kernelization_type': kernelization_type if use_kernelization else None,
                'ilp': ilp_stats,
                'quality_metrics': quality_metrics,
                'memory_stats': memory_stats,
                'success': True,
            }

            return self.solution

        except Exception as e:
            logger.error(f"Solver failed: {e}")
            if track_memory and tracemalloc.is_tracing():
                tracemalloc.stop()

            # Return partial result with error information
            return {
                'clusters': [],
                'num_clusters': 0,
                'editing_cost': float('inf'),
                'time_seconds': time.time() - start_time,
                'error': str(e),
                'success': False,
            }

    def _lift_clusters_to_original(self,
                                   kernel_instance: AdvancedClusterEditingInstance,
                                   clusters_on_kernel: List[Set[int]]) -> List[Set[int]]:
        """
        Map clusters from kernel back to original nodes.

        Args:
            kernel_instance: Kernelized instance
            clusters_on_kernel: Clusters on kernel graph

        Returns:
            Clusters on original graph
        """
        lifted: List[Set[int]] = []
        mapping = getattr(kernel_instance, "supernode_members", None)

        if not mapping:
            # No mapping available, return as-is
            return [set(C) for C in clusters_on_kernel]

        for C in clusters_on_kernel:
            S: Set[int] = set()
            for supernode in C:
                # Each supernode represents one or more original nodes
                S.update(mapping.get(supernode, {supernode}))
            lifted.append(S)

        return lifted

    def _calculate_quality_metrics(self,
                                   graph: nx.Graph,
                                   clusters: List[Set[int]]) -> Dict[str, Any]:
        """Calculate quality metrics for the clustering."""
        if not clusters:
            return {}

        cluster_sizes = [len(c) for c in clusters]

        # Count intra-cluster and inter-cluster edges
        intra_edges = 0
        for cluster in clusters:
            subgraph = graph.subgraph(cluster)
            intra_edges += subgraph.number_of_edges()

        inter_edges = 0
        for i, ci in enumerate(clusters):
            for cj in clusters[i + 1:]:
                for u in ci:
                    for v in cj:
                        if graph.has_edge(u, v):
                            inter_edges += 1

        # Calculate maximum possible intra-cluster edges
        total_possible_intra = sum(len(c) * (len(c) - 1) // 2 for c in clusters)

        return {
            'num_clusters': len(clusters),
            'avg_cluster_size': float(np.mean(cluster_sizes)),
            'std_cluster_size': float(np.std(cluster_sizes)),
            'min_cluster_size': min(cluster_sizes),
            'max_cluster_size': max(cluster_sizes),
            'singleton_clusters': sum(1 for c in clusters if len(c) == 1),
            'intra_cluster_edges': intra_edges,
            'inter_cluster_edges': inter_edges,
            'intra_cluster_density': (intra_edges / total_possible_intra)
            if total_possible_intra > 0 else 0.0,
            'modularity': self._calculate_modularity(graph, clusters),
        }

    def _calculate_modularity(self, graph: nx.Graph, clusters: List[Set[int]]) -> float:
        """Calculate modularity of the clustering."""
        try:
            # Convert clusters to partition format for NetworkX
            partition = {}
            for i, cluster in enumerate(clusters):
                for node in cluster:
                    partition[node] = i

            from networkx.algorithms.community import modularity
            return modularity(graph, clusters)
        except:
            return 0.0

    def get_solution_summary(self) -> str:
        """Get a human-readable summary of the solution."""
        if not self.solution:
            return "No solution available"

        if not self.solution.get('success', False):
            return f"Solution failed: {self.solution.get('error', 'Unknown error')}"

        lines = [
            "=== Cluster Editing Solution ===",
            f"Original graph: {self.solution['original_stats']['nodes']} nodes, "
            f"{self.solution['original_stats']['edges']} edges",
        ]

        if self.solution.get('kernel_stats'):
            ks = self.solution['kernel_stats']
            lines.append(f"Kernel reduction: {ks['reduction_ratio']:.1%} "
                         f"({ks['final_graph']['nodes']} nodes)")

        lines.extend([
            f"Clusters found: {self.solution['num_clusters']}",
            f"Editing cost: {self.solution['editing_cost']:.2f}",
            f"Time: {self.solution['time_seconds']:.3f}s",
        ])

        if self.solution.get('memory_stats'):
            ms = self.solution['memory_stats']
            lines.append(f"Peak memory: {ms['peak_memory_mb']:.1f}MB")

        qm = self.solution.get('quality_metrics', {})
        if qm:
            lines.extend([
                f"Average cluster size: {qm['avg_cluster_size']:.1f}",
                f"Intra-cluster density: {qm['intra_cluster_density']:.3f}",
                f"Modularity: {qm.get('modularity', 0):.3f}",
            ])

        return "\n".join(lines)

    @classmethod
    def from_txt(cls, txt_path: Union[str, Path], **kwargs):
        """Create solver from text file."""
        G = txt_to_networkx(str(Path(txt_path)))
        return cls(G, weights=None, **kwargs)


def benchmark_solver(graphs: Dict[str, nx.Graph],
                     output_file: str = "benchmark_results.csv") -> pd.DataFrame:
    """
    Benchmark the solver on multiple graphs.

    Args:
        graphs: Dictionary of graph name -> graph
        output_file: Where to save results

    Returns:
        DataFrame with benchmark results
    """
    import pandas as pd

    results = []

    for name, graph in graphs.items():
        print(f"Benchmarking {name}...")

        # Test different configurations
        configs = [
            {'name': 'No kernel', 'use_kernelization': False, 'use_2partition': False},
            {'name': 'Kernel only', 'use_kernelization': True, 'use_2partition': False},
            {'name': 'Full (kernel+2part)', 'use_kernelization': True, 'use_2partition': True},
        ]

        for config in configs:
            solver = ClusterEditingSolver(graph)
            solution = solver.solve(
                use_kernelization=config['use_kernelization'],
                use_2partition=config['use_2partition'],
                time_limit=300,  # 5 minute limit
                track_memory=True
            )

            results.append({
                'graph': name,
                'n_nodes': graph.number_of_nodes(),
                'n_edges': graph.number_of_edges(),
                'config': config['name'],
                'time_seconds': solution['time_seconds'],
                'editing_cost': solution['editing_cost'],
                'num_clusters': solution['num_clusters'],
                'success': solution['success'],
                'peak_memory_mb': solution.get('memory_stats', {}).get('peak_memory_mb', None),
            })

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)

    # Print summary
    print("\nBenchmark Summary:")
    print(df.groupby('config')[['time_seconds', 'editing_cost']].mean())

    return df


def main():
    """Example usage and testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Fixed Cluster Editing Solver")
    parser.add_argument('graph_file', help='Path to graph file')
    parser.add_argument('--time-limit', type=float, help='Time limit in seconds')
    parser.add_argument('--no-kernel', action='store_true', help='Disable kernelization')
    parser.add_argument('--no-2partition', action='store_true', help='Disable 2-partition cuts')
    parser.add_argument('--threads', type=int, help='Number of threads')
    parser.add_argument('--memory-limit', type=int, default=4096, help='Memory limit in MB')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    # Create and run solver
    solver = ClusterEditingSolver.from_txt(
        args.graph_file,
        memory_limit_mb=args.memory_limit
    )

    solution = solver.solve(
        use_kernelization=not args.no_kernel,
        use_2partition=not args.no_2partition,
        time_limit=args.time_limit,
        threads=args.threads,
        track_memory=True
    )

    # Print results
    print(solver.get_solution_summary())

    if solution['success']:
        print(f"\nClusters:")
        for i, cluster in enumerate(solution['clusters'], 1):
            print(f"  Cluster {i}: {sorted(cluster)}")


if __name__ == "__main__":
    main()