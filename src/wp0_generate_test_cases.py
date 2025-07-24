"""
Generate and save test cases for clique covering experiments.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from pathlib import Path
from simulator import GraphGenerator, GraphConfig

def visualize_graph(G, communities, subdirectory, title):
    """
    Visualize the graph with different colors for different communities.

    Args:
        G: NetworkX graph
        communities: Dictionary mapping node to its community (clique) id
        subdirectory: Subdirectory to save the figure in
        title: Title for the plot
    """
    plt.figure(figsize=(10, 7))

    # Create a list of colors for each community
    unique_communities = set(communities.values())
    color_map = plt.cm.get_cmap('rainbow')(np.linspace(0, 1, len(unique_communities)))
    community_colors = {comm: color_map[i] for i, comm in enumerate(unique_communities)}

    # Set node colors based on community
    node_colors = [community_colors[communities[node]] for node in G.nodes()]

    # Draw the graph
    pos = nx.spring_layout(G, seed=42)  # Position nodes using a spring layout
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color=node_colors, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5)

    plt.title(title)
    plt.axis('off')
    Path("figures").mkdir(exist_ok=True)
    Path(f"figures/{subdirectory}").mkdir(exist_ok=True)
    plt.savefig(f"figures/{subdirectory}/{title}.png")
    print(f"Saved figure to figures/{subdirectory}/{title}.png")

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
    color_map = plt.cm.get_cmap('rainbow')(np.linspace(0, 1, len(unique_communities)))
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
    color_map = plt.cm.get_cmap('rainbow')(np.linspace(0, 1, len(unique_communities)))
    community_colors = {comm: color_map[i] for i, comm in enumerate(unique_communities)}
    node_colors = [community_colors[algo_communities[node]] for node in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_size=100, node_color=node_colors, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5)
    plt.title(f"Algorithm Solution (Found {len(algorithm_cliques)} cliques)")
    plt.axis('off')

    plt.suptitle(title)
    plt.tight_layout()

def save_test_case_as_txt(G_original, G_perturbed, communities, stats_original, stats_perturbed, case_name, output_dir="test_cases/generated"):
    """Save test cases in the same txt format as curated graphs."""
    os.makedirs(output_dir, exist_ok=True)

    def write_graph_txt(G, communities, stats, filename):
        """Write a single graph in txt format"""
        with open(filename, 'w') as f:
            # Write adjacency list (sorted by vertex number)
            vertices = sorted(G.nodes())
            for vertex in vertices:
                neighbors = sorted(G.neighbors(vertex))
                if neighbors:
                    neighbors_str = ' '.join(map(str, neighbors))
                    f.write(f"{vertex}: {neighbors_str}\n")
                else:
                    f.write(f"{vertex}:\n")

            f.write("\n")  # Empty line between graph and attributes

            # Write graph attributes
            f.write(f"Connected: {'Yes' if nx.is_connected(G) else 'No'}\n")
            f.write(f"Number of Vertices: {G.number_of_nodes()}\n")
            f.write(f"Number of Edges: {G.number_of_edges()}\n")
            f.write(f"Average Degree: {2 * G.number_of_edges() / G.number_of_nodes():.3f}\n")
            f.write(f"Density: {nx.density(G):.3f}\n")
            f.write(f"Number of Components: {nx.number_connected_components(G)}\n")

            # Add clique cover information
            f.write(f"Ground Truth Clique Cover: {len(set(communities.values()))}\n")
            f.write(f"Maximum Degree: {max(dict(G.degree()).values()) if G.nodes() else 0}\n")
            f.write(f"Minimum Degree: {min(dict(G.degree()).values()) if G.nodes() else 0}\n")

            # Add generation parameters
            f.write("Generated: Yes\n")
            f.write(f"Original Cliques: {stats.get('num_cliques', 'N/A')}\n")
            f.write(f"Edge Removal Probability: {stats.get('edge_removal_prob', 'N/A')}\n")
            f.write(f"Edge Addition Probability: {stats.get('edge_addition_prob', 'N/A')}\n")
            f.write(f"Distribution Type: {stats.get('distribution_type', 'N/A')}\n")

    # Save original graph
    original_filename = f"{output_dir}/{case_name}_original.txt"
    write_graph_txt(G_original, communities, stats_original, original_filename)

    # Save perturbed graph
    perturbed_filename = f"{output_dir}/{case_name}_perturbed.txt"
    write_graph_txt(G_perturbed, communities, stats_perturbed, perturbed_filename)

    print(f"Saved {case_name} as txt files to {output_dir}")

def generate_test_suite():
    """Generate a comprehensive test suite with different parameters."""
    # Generate uniform distribution examples
    for size in [5, 10, 20]:
        for num_cliques in [3, 5, 10]:
            for removal_prob in [0.1, 0.3]:
                config = GraphConfig(
                    num_cliques=num_cliques,
                    distribution_type="uniform",
                    uniform_size=size,
                    edge_removal_prob=removal_prob,
                    edge_addition_prob=removal_prob/4
                )

                result = GraphGenerator.generate_test_case(config)
                G_original, G_perturbed, communities, stats_original, stats_perturbed = result

                case_name = f"uniform_n{num_cliques}_s{size}_r{int(removal_prob*100)}"

                # Save as txt files instead of pickle
                save_test_case_as_txt(G_original, G_perturbed, communities, stats_original, stats_perturbed, case_name)

                # Visualize the first few examples
                if size <= 10 and num_cliques <= 5:
                    visualize_graph(G_original, communities, "original", case_name)
                    visualize_graph(G_perturbed, communities, "perturbed", case_name)

    # Generate skewed distribution examples
    for min_size in [2, 6, 10]:
        for max_size in [14, 18, 22]:
            for num_cliques in [3, 5, 10]:
                for removal_prob in [0.1, 0.3]:
                    config = GraphConfig(
                        num_cliques=num_cliques,
                        distribution_type="skewed",
                        min_size=min_size,
                        max_size=max_size,
                        edge_removal_prob=removal_prob,
                        edge_addition_prob=removal_prob/4
                    )

                    result = GraphGenerator.generate_test_case(config)
                    G_original, G_perturbed, communities, stats_original, stats_perturbed = result

                    case_name = f"skewed_n{num_cliques}_min{min_size}_max{max_size}_r{int(removal_prob*100)}"

                    # Save as txt files instead of pickle
                    save_test_case_as_txt(G_original, G_perturbed, communities, stats_original, stats_perturbed, case_name)

                    visualize_graph(G_original, communities, "original", case_name)
                    visualize_graph(G_perturbed, communities, "perturbed", case_name)

if __name__ == "__main__":
    generate_test_suite()
