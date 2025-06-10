"""
Simple test for ILP solver - just to verify it works.
"""

import networkx as nx
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src" / "algorithms"))

from src.algorithms.ilp_solver import solve_ilp_clique_cover

def test_simple_graphs():
    """Test with a few simple graphs."""

    print("Testing ILP solver with simple graphs...")

    # Test 1: Triangle (should need 3 colors)
    print("\n1. Triangle (K3):")
    G1 = nx.complete_graph(3)
    result1 = solve_ilp_clique_cover(G1)
    print(f"   Nodes: {result1['n_nodes']}, Edges: {result1['n_edges']}")
    print(f"   Chromatic number: {result1['chromatic_number']}")
    print(f"   Coloring: {result1['coloring']}")

    # Test 2: Path of 4 nodes (should need 2 colors)
    print("\n2. Path graph (4 nodes):")
    G2 = nx.path_graph(4)  # 0-1-2-3
    result2 = solve_ilp_clique_cover(G2)
    print(f"   Nodes: {result2['n_nodes']}, Edges: {result2['n_edges']}")
    print(f"   Chromatic number: {result2['chromatic_number']}")
    print(f"   Coloring: {result2['coloring']}")

    # Test 3: Single node
    print("\n3. Single node:")
    G3 = nx.Graph()
    G3.add_node(1)
    result3 = solve_ilp_clique_cover(G3)
    print(f"   Nodes: {result3['n_nodes']}, Edges: {result3['n_edges']}")
    print(f"   Chromatic number: {result3['chromatic_number']}")
    print(f"   Coloring: {result3['coloring']}")

    print("\nAll tests completed!")

if __name__ == "__main__":
    test_simple_graphs()
