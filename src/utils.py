import networkx as nx

def get_value(txt_filepath, attribute_name="Chromatic Number"):
    """Extract a value from a txt file"""
    with open(txt_filepath, 'r') as f:
        for line in f:
            if line.startswith(f"{attribute_name}:"):
                value_str = line.split(':', 1)[1].strip()
                return value_str
    return None

def txt_to_networkx(txt_filepath):
    """Convert txt adjacency list format to NetworkX graph"""
    G = nx.Graph()

    with open(txt_filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if ':' in line and line.split(':')[0].strip().isdigit():
                parts = line.split(':')
                vertex = int(parts[0].strip())
                G.add_node(vertex)

                if len(parts) > 1 and parts[1].strip():
                    neighbors = [int(x) for x in parts[1].strip().split()]
                    for neighbor in neighbors:
                        G.add_edge(vertex, neighbor)
            elif line and not line.split(':')[0].strip().isdigit():
                # Hit the attributes section, stop parsing
                break

    # Convert to 0-indexed for Chalupa algorithm
    mapping = {old: new for new, old in enumerate(sorted(G.nodes()))}
    G = nx.relabel_nodes(G, mapping)

    return G
