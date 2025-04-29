import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt

def generate_test_graph(num_vertices, num_cliques, clique_size_distribution='uniform',
                        within_clique_noise=0.1, between_clique_noise=0.05, seed=None):
    """
    Generate a test graph for clique cover problems.

    Parameters:
    num_vertices (int): Total number of vertices in the graph
    num_cliques (int): Number of cliques to start with
    clique_size_distribution (str): 'uniform', 'skewed', or 'power_law'
    within_clique_noise (float): Probability of removing an edge within a clique
    between_clique_noise (float): Probability of adding an edge between different cliques
    seed (int): Random seed for reproducibility

    Returns:
    G (nx.Graph): The generated graph
    clique_assignments (list): List of vertex sets for the original cliques
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Create empty graph
    G = nx.Graph()

    # Determine clique sizes based on the specified distribution
    if clique_size_distribution == 'uniform':
        # All cliques have roughly the same size
        base_size = num_vertices // num_cliques
        remainder = num_vertices % num_cliques
        clique_sizes = [base_size + (1 if i < remainder else 0) for i in range(num_cliques)]

    elif clique_size_distribution == 'skewed':
        # Few large cliques, many small ones
        total = 0
        clique_sizes = []
        # Set the first few cliques to be large
        large_cliques = min(num_cliques // 3, 1)
        for i in range(large_cliques):
            size = num_vertices // (large_cliques + 1)
            clique_sizes.append(size)
            total += size

        # Distribute remaining vertices among small cliques
        remaining = num_vertices - total
        small_clique_count = num_cliques - large_cliques
        min_size = 2  # Minimum size for a clique

        if small_clique_count > 0:
            if remaining >= small_clique_count * min_size:
                base_small_size = remaining // small_clique_count
                small_remainder = remaining % small_clique_count
                for i in range(small_clique_count):
                    clique_sizes.append(base_small_size + (1 if i < small_remainder else 0))
            else:
                # Can't satisfy minimum size requirement
                for i in range(small_clique_count - 1):
                    clique_sizes.append(min_size)
                    remaining -= min_size
                clique_sizes.append(remaining)  # Last clique gets what's left

    elif clique_size_distribution == 'power_law':
        # Use a power law distribution
        alpha = 2.5  # Parameter for power law
        raw_sizes = np.random.power(alpha, num_cliques) + 1  # +1 to avoid zero-sized cliques
        # Scale to get the desired total
        total = sum(raw_sizes)
        scaling_factor = num_vertices / total
        clique_sizes = [max(2, int(s * scaling_factor)) for s in raw_sizes]

        # Adjust to get exactly num_vertices
        while sum(clique_sizes) > num_vertices:
            idx = random.randrange(len(clique_sizes))
            if clique_sizes[idx] > 2:
                clique_sizes[idx] -= 1

        while sum(clique_sizes) < num_vertices:
            idx = random.randrange(len(clique_sizes))
            clique_sizes[idx] += 1

    # Create vertices for each clique
    vertex_id = 0
    clique_assignments = []

    for size in clique_sizes:
        clique_vertices = list(range(vertex_id, vertex_id + size))
        clique_assignments.append(set(clique_vertices))

        # Add all vertices to the graph
        G.add_nodes_from(clique_vertices)

        # Add all possible edges within the clique
        for u in clique_vertices:
            for v in clique_vertices:
                if u < v:  # Avoid self-loops and duplicate edges
                    G.add_edge(u, v)

        vertex_id += size

    # Apply noise: remove edges within cliques
    edges_to_remove = []
    for clique_vertices in clique_assignments:
        for u in clique_vertices:
            for v in clique_vertices:
                if u < v and (u, v) in G.edges() and random.random() < within_clique_noise:
                    edges_to_remove.append((u, v))

    G.remove_edges_from(edges_to_remove)

    # Apply noise: add edges between different cliques
    all_vertices = list(G.nodes())
    for i, clique1 in enumerate(clique_assignments):
        for j, clique2 in enumerate(clique_assignments):
            if i < j:  # Only consider unique pairs of cliques
                for u in clique1:
                    for v in clique2:
                        if random.random() < between_clique_noise:
                            G.add_edge(u, v)

    return G, clique_assignments

def visualize_graph(G, clique_assignments=None, title="Test Graph"):
    """
    Visualize the graph with different colors for each original clique.

    Parameters:
    G (nx.Graph): The graph to visualize
    clique_assignments (list): List of vertex sets for the original cliques
    title (str): Title for the plot
    """
    plt.figure(figsize=(10, 8))

    if clique_assignments:
        # Color nodes based on their original clique
        color_map = []
        color_palette = plt.cm.tab20(np.linspace(0, 1, len(clique_assignments)))

        node_colors = {}
        for i, clique in enumerate(clique_assignments):
            for node in clique:
                node_colors[node] = color_palette[i]

        color_map = [node_colors[node] for node in G.nodes()]
        nx.draw(G, node_color=color_map, with_labels=True, node_size=300, font_weight='bold')
    else:
        nx.draw(G, with_labels=True, node_size=300, font_weight='bold')

    plt.title(title)
    plt.show()

def save_graph_to_file(G, filename):
    """
    Save graph to an adjacency list file.

    Parameters:
    G (nx.Graph): The graph to save
    filename (str): Output filename
    """
    with open(filename, 'w') as f:
        f.write(f"{G.number_of_nodes()} {G.number_of_edges()}\n")
        for edge in G.edges():
            f.write(f"{edge[0]} {edge[1]}\n")

def get_graph_stats(G, original_cliques):
    """
    Calculate statistics for the generated graph.

    Parameters:
    G (nx.Graph): The graph
    original_cliques (list): List of vertex sets for the original cliques

    Returns:
    dict: Dictionary with graph statistics
    """
    # Calculate statistics
    stats = {
        "num_vertices": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "num_original_cliques": len(original_cliques),
        "density": nx.density(G),
        "average_degree": sum(dict(G.degree()).values()) / G.number_of_nodes(),
        "clique_sizes": [len(clique) for clique in original_cliques]
    }

    # Calculate edge distribution
    intra_clique_edges = 0
    total_possible_intra = 0

    for clique in original_cliques:
        clique = list(clique)
        for i in range(len(clique)):
            for j in range(i+1, len(clique)):
                total_possible_intra += 1
                if G.has_edge(clique[i], clique[j]):
                    intra_clique_edges += 1

    stats["intra_clique_edges"] = intra_clique_edges
    stats["intra_clique_edge_ratio"] = intra_clique_edges / total_possible_intra if total_possible_intra > 0 else 0

    inter_clique_edges = G.number_of_edges() - intra_clique_edges
    total_possible_inter = (G.number_of_nodes() * (G.number_of_nodes() - 1)) // 2 - total_possible_intra

    stats["inter_clique_edges"] = inter_clique_edges
    stats["inter_clique_edge_ratio"] = inter_clique_edges / total_possible_inter if total_possible_inter > 0 else 0

    return stats

# Example usage
def main():
    # Generate test graphs with different parameters
    distributions = ["uniform", "skewed", "power_law"]

    for dist in distributions:
        # Create small graph for visualization
        G_small, cliques_small = generate_test_graph(
            num_vertices=20,
            num_cliques=4,
            clique_size_distribution=dist,
            within_clique_noise=0.1,
            between_clique_noise=0.05,
            seed=42
        )

        # Visualize
        visualize_graph(G_small, cliques_small, f"Test Graph with {dist} Distribution")

        # Print statistics
        stats = get_graph_stats(G_small, cliques_small)
        print(f"\nStats for {dist} distribution:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # Create larger graph for algorithm testing
        G_large, cliques_large = generate_test_graph(
            num_vertices=100,
            num_cliques=10,
            clique_size_distribution=dist,
            within_clique_noise=0.1,
            between_clique_noise=0.05,
            seed=42
        )

        # Save to file
        save_graph_to_file(G_large, f"test_graph_{dist}.txt")

        # Generate test cases with varying noise levels
        for within_noise in [0.05, 0.1, 0.2]:
            for between_noise in [0.02, 0.05, 0.1]:
                G_noise, cliques_noise = generate_test_graph(
                    num_vertices=100,
                    num_cliques=10,
                    clique_size_distribution=dist,
                    within_clique_noise=within_noise,
                    between_clique_noise=between_noise,
                    seed=100 + int(within_noise*100) + int(between_noise*100)
                )

                filename = f"test_graph_{dist}_w{int(within_noise*100)}_b{int(between_noise*100)}.txt"
                save_graph_to_file(G_noise, filename)
                print(f"Generated {filename}")

if __name__ == "__main__":
    main()
