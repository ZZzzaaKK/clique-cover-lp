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
