"""
Graph generation and perturbation for clique covering experiments.
"""

import networkx as nx
import numpy as np
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class GraphConfig:
    """Configuration for graph generation."""
    num_cliques: int = 5
    distribution_type: str = "uniform"  # "uniform" or "skewed"
    min_size: int = 5
    max_size: int = 20
    uniform_size: int = 10
    edge_removal_prob: float = 0.2
    edge_addition_prob: float = 0.05
    skewness: str = "high"  # "low", "medium", or "high"

class GraphGenerator:
    """Generates graphs with clique structure for testing clique covering algorithms."""

    @staticmethod
    def create_disjoint_cliques(clique_sizes):
        """
        Create a graph with disjoint cliques of specified sizes.

        Args:
            clique_sizes (list): List of sizes for each clique

        Returns:
            G: NetworkX graph with disjoint cliques
            communities: Dictionary mapping node to its clique id
        """
        G = nx.Graph()
        communities = {}

        node_id = 0
        for i, size in enumerate(clique_sizes):
            # Create a clique of the given size
            clique_nodes = list(range(node_id, node_id + size))

            # Add all nodes and edges for this clique
            for u in clique_nodes:
                G.add_node(u)
                communities[u] = i  # Mark community membership
                for v in clique_nodes:
                    if u != v:  # Avoid self-loops
                        G.add_edge(u, v)

            node_id += size

        return G, communities

    @staticmethod
    def perturb_graph(G, communities, edge_removal_prob, edge_addition_prob):
        """
        Perturb the graph by removing edges within cliques and adding edges between cliques.

        Args:
            G: NetworkX graph with cliques
            communities: Dictionary mapping node to its clique id
            edge_removal_prob: Probability of removing an edge within a clique
            edge_addition_prob: Probability of adding an edge between different cliques

        Returns:
            G: The perturbed graph
        """
        # Make a copy of the graph to modify
        G_perturbed = G.copy()

        # Remove edges within cliques
        for u, v in list(G.edges()):
            if communities[u] == communities[v]:  # Same clique
                if random.random() < edge_removal_prob:
                    G_perturbed.remove_edge(u, v)

        # Add edges between cliques
        nodes = list(G.nodes())
        for i, u in enumerate(nodes):
            for v in nodes[i+1:]:
                if communities[u] != communities[v]:  # Different cliques
                    if not G_perturbed.has_edge(u, v) and random.random() < edge_addition_prob:
                        G_perturbed.add_edge(u, v)

        return G_perturbed

    @staticmethod
    def generate_uniform_clique_sizes(num_cliques, clique_size):
        """Generate uniform clique sizes."""
        return [clique_size] * num_cliques

    @staticmethod
    def generate_skewed_clique_sizes(num_cliques, min_size, max_size, skewness="high"):
        """
        Generate skewed clique sizes with few large and many small ones.

        Args:
            num_cliques: Number of cliques to generate
            min_size: Minimum size of a clique
            max_size: Maximum size of a clique
            skewness: "low", "medium", or "high" skewness

        Returns:
            List of clique sizes
        """
        if skewness == "low":
            # Slightly skewed - closer to uniform
            alpha = 3.0
        elif skewness == "medium":
            # Medium skewness
            alpha = 1.5
        else:  # high
            # Highly skewed
            alpha = 0.5

        # Generate sizes from a power law distribution
        sizes = np.random.power(alpha, size=num_cliques)

        # Scale to the desired range
        sizes = min_size + (max_size - min_size) * sizes

        # Round to integers
        return [int(size) for size in sizes]

    @staticmethod
    def analyze_graph(G, communities):
        """
        Analyze properties of the graph.

        Args:
        G: NetworkX graph
        communities: Dictionary mapping node to its clique id

        Returns:
        dict: Dictionary of graph properties
        """
        # Group nodes by community
        community_nodes = defaultdict(list)
        for node, comm in communities.items():
            community_nodes[comm].append(node)

        # Calculate statistics
        stats = {
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "avg_degree": sum(dict(G.degree()).values()) / G.number_of_nodes(),
            "num_components": nx.number_connected_components(G),
            "avg_clustering": nx.average_clustering(G),
            "density": nx.density(G),
            "community_sizes": [len(nodes) for comm, nodes in community_nodes.items()],
            "community_densities": []
        }

        # Calculate density within each community
        for comm, nodes in community_nodes.items():
            subgraph = G.subgraph(nodes)
            stats["community_densities"].append(nx.density(subgraph))

        return stats

    @classmethod
    def generate_test_case(cls, config: GraphConfig) -> Tuple[nx.Graph, nx.Graph, Dict, Dict, Dict]:
        """Generate a test case based on the configuration."""
        # Generate clique sizes
        if config.distribution_type == "uniform":
            clique_sizes = cls.generate_uniform_clique_sizes(config.num_cliques, config.uniform_size)
        else:
            clique_sizes = cls.generate_skewed_clique_sizes(
                config.num_cliques, config.min_size, config.max_size, config.skewness
            )

        # Create original graph with disjoint cliques
        G_original, communities = cls.create_disjoint_cliques(clique_sizes)

        # Perturb the graph
        G_perturbed = cls.perturb_graph(
            G_original, communities, config.edge_removal_prob, config.edge_addition_prob
        )

        # Analyze the graphs
        stats_original = cls.analyze_graph(G_original, communities)
        stats_perturbed = cls.analyze_graph(G_perturbed, communities)

        return G_original, G_perturbed, communities, stats_original, stats_perturbed
