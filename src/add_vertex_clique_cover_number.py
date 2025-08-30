#!/usr/bin/env python3
from pathlib import Path
from utils import get_value, txt_to_networkx
import sys
import networkx as nx
from algorithms.ilp_solver import solve_ilp_clique_cover

def add_vertex_clique_cover_number_if_missing(directory):
    """Add vertex clique cover number only to files that don't already have it"""
    path = Path(directory)

    for txt_file in path.glob("**/*.txt"):
        # Check if vertex clique cover number already exists
        existing_value = get_value(txt_file, "Vertex Clique Cover Number")

        if existing_value is None:
            print(f"Computing vertex clique cover number for {txt_file.name}...")
            G = txt_to_networkx(str(txt_file))
            G_complement = nx.complement(G)
            result = solve_ilp_clique_cover(G_complement, time_limit=300, require_optimal=True)

            if 'error' not in result:
                vcc_number = result['chromatic_number']
                # Append to file
                with open(txt_file, 'a') as f:
                    f.write("\n# Calculated by ILP on complement graph\n")
                    f.write(f"Vertex Clique Cover Number: {vcc_number}\n")
                print(f"  Added: Vertex Clique Cover Number: {vcc_number}")
            else:
                print(f"  Failed to compute for {txt_file.name}")
        else:
            print(f"Vertex Clique Cover Number already exists for {txt_file.name}: {existing_value}")

if __name__ == "__main__":
    graph_dir = sys.argv[1] if len(sys.argv) > 1 else "test_graphs/generated/perturbed"
    add_vertex_clique_cover_number_if_missing(graph_dir)
