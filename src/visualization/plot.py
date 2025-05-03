import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

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

def visualize_solution_comparison(G, ground_truth_cliques, algorithm_cliques, title="Solution Comparison"):
    """
    Visualize comparison between ground truth and algorithm solution.

    Args:
        G: NetworkX graph
        ground_truth_cliques: List of cliques (each clique is a set of nodes)
        algorithm_cliques: List of cliques found by the algorithm
        title: Title for the plot
    """
    plt.figure(figsize=(15, 7))

    # Create two subplots
    plt.subplot(1, 2, 1)

    # Map nodes to their clique id for ground truth
    gt_communities = {}
    for i, clique in enumerate(ground_truth_cliques):
        for node in clique:
            gt_communities[node] = i

    # Visualize ground truth
    pos = nx.spring_layout(G, seed=42)  # Same layout for both

    # Draw ground truth
    unique_communities = set(gt_communities.values())
    color_map = plt.cm.rainbow(np.linspace(0, 1, len(unique_communities)))
    community_colors = {comm: color_map[i] for i, comm in enumerate(unique_communities)}
    node_colors = [community_colors[gt_communities[node]] for node in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_size=100, node_color=node_colors, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5)
    plt.title("Ground Truth")
    plt.axis('off')

    # Map nodes to their clique id for algorithm solution
    algo_communities = {}
    for i, clique in enumerate(algorithm_cliques):
        for node in clique:
            algo_communities[node] = i

    # Draw algorithm solution
    plt.subplot(1, 2, 2)
    unique_communities = set(algo_communities.values())
    color_map = plt.cm.rainbow(np.linspace(0, 1, len(unique_communities)))
    community_colors = {comm: color_map[i] for i, comm in enumerate(unique_communities)}
    node_colors = [community_colors[algo_communities[node]] for node in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_size=100, node_color=node_colors, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5)
    plt.title(f"Algorithm Solution (Found {len(algorithm_cliques)} cliques)")
    plt.axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
