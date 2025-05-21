import unittest
import networkx as nx
from reductions import apply_all_reductions

class TestReductions(unittest.TestCase):

    def test_isolated_vertex(self):
        G = nx.Graph()
        G.add_nodes_from([1, 2, 3])
        G.add_edge(1, 2)
        G_reduced, trace = apply_all_reductions(G.copy(), verbose=False, timing=False)
        self.assertFalse(3 in G_reduced.nodes)
        self.assertTrue(any("isolated" in name.lower() for name, _ in trace))

    def test_degree_two_folding(self):
        G = nx.Graph()
        G.add_edges_from([(1, 2), (2, 3)])
        G_reduced, trace = apply_all_reductions(G.copy(), verbose=False, timing=False)
        self.assertNotIn(2, G_reduced.nodes)
        self.assertTrue((1, 3) in G_reduced.edges)

    def test_twin_removal(self):
        G = nx.Graph()
        G.add_edges_from([(1, 4), (2, 4), (3, 4)])
        G.add_edges_from([(1, 5), (2, 5), (3, 5)])
        G.add_edges_from([(1, 6), (2, 6), (3, 6)])
        G_reduced, trace = apply_all_reductions(G.copy(), verbose=False, timing=False)
        self.assertLess(len(G_reduced.nodes), 6)

    def test_domination(self):
        G = nx.Graph()
        G.add_edges_from([(1, 3), (2, 3), (1, 2)])
        G_reduced, trace = apply_all_reductions(G.copy(), verbose=False, timing=False)
        self.assertLess(len(G_reduced.nodes), 3)

    def test_twin_folding(self):
        G = nx.Graph()
        G.add_edges_from([(1, 4), (1, 5), (1, 6),
                          (2, 4), (2, 5), (2, 6)])
        G_reduced, trace = apply_all_reductions(G.copy(), verbose=False, timing=False)
        twin_folded = any("twin_folding" in name.lower() for name, _ in trace)
        self.assertTrue(twin_folded)

    def test_crown_reduction(self):
        G = nx.Graph()
        G.add_edges_from([(1, 4), (2, 4), (3, 4), (1, 5), (2, 5), (3, 5)])
        G_reduced, trace = apply_all_reductions(G.copy(), verbose=False, timing=False)
        crown_reduced = any("crown" in name.lower() for name, _ in trace)
        self.assertTrue(crown_reduced)

if __name__ == '__main__':
    unittest.main()
