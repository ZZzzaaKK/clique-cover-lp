import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict


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


def generate_uniform_clique_sizes(num_cliques, clique_size):
    """Generate uniform clique sizes."""
    return [clique_size] * num_cliques


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


def visualize_graph(G, communities, title):
    """
    Visualize the graph with different colors for different communities.

    Args:
        G: NetworkX graph
        communities: Dictionary mapping node to its community (clique) id
        title: Title for the plot
    """
    plt.figure(figsize=(10, 7))

    # Create a list of colors for each community
    unique_communities = set(communities.values())
    color_map = plt.cm.rainbow(np.linspace(0, 1, len(unique_communities)))
    community_colors = {comm: color_map[i] for i, comm in enumerate(unique_communities)}

    # Set node colors based on community
    node_colors = [community_colors[communities[node]] for node in G.nodes()]

    # Draw the graph
    pos = nx.spring_layout(G, seed=42)  # Position nodes using a spring layout
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color=node_colors, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5)

    plt.title(title)
    plt.axis('off')
    plt.show()


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


def run_simulation(num_cliques=5, distribution_type="uniform",
                  min_size=5, max_size=20, uniform_size=10,
                  edge_removal_prob=0.2, edge_addition_prob=0.05,
                  skewness="high", visualize=True):
    """
    Run a full simulation, creating cliques and perturbing them.

    Args:
        num_cliques: Number of cliques to create
        distribution_type: "uniform" or "skewed"
        min_size: Minimum clique size (for skewed distribution)
        max_size: Maximum clique size (for skewed distribution)
        uniform_size: Size of each clique (for uniform distribution)
        edge_removal_prob: Probability of removing edges within cliques
        edge_addition_prob: Probability of adding edges between cliques
        skewness: "low", "medium", or "high" (for skewed distribution)
        visualize: Whether to visualize the graphs

    Returns:
        tuple: (original_graph, perturbed_graph, communities, stats_original, stats_perturbed)
    """
    # Generate clique sizes
    if distribution_type == "uniform":
        clique_sizes = generate_uniform_clique_sizes(num_cliques, uniform_size)
    else:
        clique_sizes = generate_skewed_clique_sizes(num_cliques, min_size, max_size, skewness)

    print(f"Generated clique sizes: {clique_sizes}")

    # Create original graph with disjoint cliques
    G_original, communities = create_disjoint_cliques(clique_sizes)

    # Perturb the graph
    G_perturbed = perturb_graph(G_original, communities, edge_removal_prob, edge_addition_prob)

    # Analyze the graphs
    stats_original = analyze_graph(G_original, communities)
    stats_perturbed = analyze_graph(G_perturbed, communities)

    # Visualize if requested
    if visualize:
        visualize_graph(G_original, communities, "Original Disjoint Cliques")
        visualize_graph(G_perturbed, communities, "Perturbed Graph")

    return G_original, G_perturbed, communities, stats_original, stats_perturbed


def print_stats(stats_original, stats_perturbed):
    """Print comparative statistics for original and perturbed graphs."""
    print("\nGraph Statistics:")
    print("-" * 50)
    print(f"Number of nodes: {stats_original['num_nodes']}")
    print(f"Original edges: {stats_original['num_edges']} | Perturbed edges: {stats_perturbed['num_edges']}")
    print(f"Original avg degree: {stats_original['avg_degree']:.2f} | Perturbed: {stats_perturbed['avg_degree']:.2f}")
    print(f"Original clustering: {stats_original['avg_clustering']:.4f} | Perturbed: {stats_perturbed['avg_clustering']:.4f}")
    print(f"Original components: {stats_original['num_components']} | Perturbed: {stats_perturbed['num_components']}")
    print(f"Original density: {stats_original['density']:.4f} | Perturbed: {stats_perturbed['density']:.4f}")
    print(f"Community sizes: {stats_original['community_sizes']}")
    print("\nDensity within communities:")
    print(f"Original: {[f'{d:.4f}' for d in stats_original['community_densities']]}")
    print(f"Perturbed: {[f'{d:.4f}' for d in stats_perturbed['community_densities']]}")


if __name__ == "__main__":
    # Example 1: Uniform distribution with moderate perturbation
    print("\n=== Example 1: Uniform distribution ===")
    G_orig1, G_pert1, comm1, stats_orig1, stats_pert1 = run_simulation(
        num_cliques=5,
        distribution_type="uniform",
        uniform_size=8,
        edge_removal_prob=0.2,
        edge_addition_prob=0.05
    )
    print_stats(stats_orig1, stats_pert1)

    # Example 2: Highly skewed distribution
    print("\n=== Example 2: Highly skewed distribution ===")
    G_orig2, G_pert2, comm2, stats_orig2, stats_pert2 = run_simulation(
        num_cliques=8,
        distribution_type="skewed",
        min_size=3,
        max_size=25,
        edge_removal_prob=0.3,
        edge_addition_prob=0.1,
        skewness="high"
    )
    print_stats(stats_orig2, stats_pert2)

    # Example 3: Mildly skewed distribution
    print("\n=== Example 3: Mildly skewed distribution ===")
    G_orig3, G_pert3, comm3, stats_orig3, stats_pert3 = run_simulation(
        num_cliques=6,
        distribution_type="skewed",
        min_size=5,
        max_size=15,
        edge_removal_prob=0.15,
        edge_addition_prob=0.03,
        skewness="low"
    )
    print_stats(stats_orig3, stats_pert3)
