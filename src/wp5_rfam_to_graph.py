import pandas as pd
import networkx as nx

# Load the TSV file into a pandas DataFrame
df = pd.read_csv("test_graphs/rfam/RF02246.tsv", sep="\t")
# Convert 'shifts' to a boolean indicating whether any shift occurred
df["shift_event"] = df["shifts"] > 0
# Create an undirected graph
G = nx.Graph()
# Iterate through the DataFrame and add edges
for _, row in df.iterrows():
    # Skip self-loops for now because it messes with ILP
    if row["idA"] == row["idB"]:
        continue
    G.add_edge(
        row["idA"],
        row["idB"],
        score=row["score"],
        shifts=row["shifts"],
        shift_event=row["shift_event"]
    )

# Create a mapping from node names to integers
nodes = list(G.nodes())
node_to_int = {node: i for i, node in enumerate(nodes)}
int_to_node = {i: node for i, node in enumerate(nodes)}

# Save the graph in adjacency list format
output_file = "test_graphs/rfam/RF02246_graph.txt"
with open(output_file, "w") as f:
    for node in sorted(G.nodes()):
        node_int = node_to_int[node]
        neighbors = sorted([node_to_int[neighbor] for neighbor in G.neighbors(node)])
        neighbors_str = " ".join(map(str, neighbors))
        f.write(f"{node_int}: {neighbors_str}\n")

    # Add graph statistics
    f.write(f"\nConnected: {'Yes' if nx.is_connected(G) else 'No'}\n")
    f.write(f"Number of Vertices: {G.number_of_nodes()}\n")
    f.write(f"Number of Edges: {G.number_of_edges()}\n")
    f.write(f"Density: {nx.density(G):.3f}\n")
    f.write(f"Number of Components: {nx.number_connected_components(G)}\n")

print(f"Graph saved to {output_file}")
print(f"Node mapping saved for reference:")
for i, node in int_to_node.items():
    print(f"{i}: {node}")
