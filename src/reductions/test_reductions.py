import unittest
import sys
import os
import networkx as nx
from reductions import apply_isolated_vertex_reduction, apply_degree_two_folding, apply_twin_removal, apply_twin_folding, apply_domination_reduction, apply_crown_reduction, apply_all_reductions

# Define the test case class
class TestIsolatedVertexReduction(unittest.TestCase):
    def setUp(self):
        # Create a simple graph with isolated vertices
        self.G = nx.Graph()
        self.G.add_edges_from([(1, 2), (2, 3)])
        self.G.add_node(4)  # Isolated vertex
        self.G.add_node(5)  # Another isolated vertex

    def test_isolated_vertex_reduction(self):
        # Use the function from the current context
        G_reduced, changed, removed = apply_isolated_vertex_reduction(self.G.copy())

        self.assertTrue(changed, "Graph should have changed due to isolated vertex removal.")
        self.assertIn(4, removed, "Isolated vertex 4 should have been removed.")
        self.assertIn(5, removed, "Isolated vertex 5 should have been removed.")
        self.assertNotIn(4, G_reduced.nodes, "Isolated vertex 4 should not be in the reduced graph.")
        self.assertNotIn(5, G_reduced.nodes, "Isolated vertex 5 should not be in the reduced graph.")


# Define the test case class
class TestDegreeTwoFolding(unittest.TestCase):
    def setUp(self):
        # Create a simple graph with a degree-2 foldable node
        self.G = nx.Graph()
        self.G.add_edges_from([(1, 2), (1, 3)])
        self.G.add_edge(2, 4)
        self.G.add_edge(3, 5)

    def test_degree_two_folding(self):
        # Use the function from the current context
        G_folded, changed, folds = apply_degree_two_folding(self.G.copy())

        self.assertTrue(changed, "Graph should have changed due to folding.")
        self.assertEqual(len(folds), 1, "One fold should have occurred.")

        v, u, w = folds[0]
        self.assertIn(f"fold_{v}", G_folded.nodes, "Folded node should exist.")
        folded_neighbors = set(G_folded.neighbors(f"fold_{v}"))
        self.assertEqual(folded_neighbors, {4, 5}, "Folded node should connect to external neighbors of u and w.")

        for node in (v, u, w):
            self.assertNotIn(node, G_folded.nodes, f"Node {node} should have been removed.")


class TestTwinRemoval(unittest.TestCase):
    def setUp(self):
        # Create a graph with twin nodes
        self.G = nx.Graph()
        self.G.add_edges_from([(1, 4), (2, 4), (3, 4)])
        self.G.add_edges_from([(1, 5), (2, 5), (3, 5)])
        self.G.add_edges_from([(1, 6), (2, 6), (3, 6)])
        self.G.add_edges_from([(1, 2)])

    def test_twin_removal(self):
        G_reduced, changed, removed = apply_twin_removal(self.G.copy())
        print("Reductions applied (twin_removal):", [name for name, _ in removed])
        self.assertTrue(changed, "Graph should have changed due to twin removal.")
        self.assertLess(len(G_reduced.nodes), 6, "Graph should have fewer than 6 nodes after twin removal.")
        self.assertNotIn(4, G_reduced.nodes, "Node 4 should have been removed as a twin.")
        self.assertNotIn(5, G_reduced.nodes, "Node 5 should have been removed as a twin.")
        self.assertNotIn(6, G_reduced.nodes, "Node 6 should have been removed as a twin.")


class TestTwinFolding(unittest.TestCase):
    def setUp(self):
        # Create a graph with twin nodes
        self.G = nx.Graph()
        self.G.add_edges_from([(1, 4), (1, 5), (1, 6),
                               (2, 4), (2, 5), (2, 6)])

    def test_twin_folding(self):
        G_reduced, changed, folds = apply_twin_folding(self.G.copy())
        print("Reductions applied (twin_folding):", [name for name, _ in folds])
        self.assertTrue(changed, "Graph should have changed due to twin folding.")
        self.assertTrue(any("twin_folding" in name.lower() for name, _ in folds), "Twin folding should have been applied.")


class TestDominationReduction(unittest.TestCase):
    def setUp(self):
        # Create a graph for domination reduction
        self.G = nx.Graph()
        self.G.add_edges_from([(1, 3), (2, 3), (1, 2)])

    def test_domination(self):
        G_reduced, changed, removed = apply_domination_reduction(self.G.copy())
        print("Reductions applied (domination):", [name for name, _ in removed])
        self.assertTrue(changed, "Graph should have changed due to domination reduction.")
        self.assertLess(len(G_reduced.nodes), 3, "Graph should have fewer than 3 nodes after domination reduction.")


class TestCrownReduction(unittest.TestCase):
    def setUp(self):
        # Create a graph for crown reduction
        self.G = nx.Graph()
        self.G.add_edges_from([(1, 4), (2, 4), (3, 4), (1, 5), (2, 5), (3, 5)])

    def test_crown_reduction(self):
        G_reduced, changed, removed = apply_crown_reduction(self.G.copy())
        print("Reductions applied (crown_reduction):", [name for name, _ in removed])
        self.assertTrue(changed, "Graph should have changed due to crown reduction.")
        self.assertTrue(any("crown" in name.lower() for name, _ in removed), "Crown reduction should have been applied.")


class TestAllReductions(unittest.TestCase):
    def setUp(self):
        # Create a graph for testing all reductions
        self.G = nx.Graph()
        self.G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5)])
        self.G.add_node(6)  # Isolated vertex

    def test_all_reductions(self):
        G_reduced, trace = apply_all_reductions(self.G.copy(), verbose=False, timing=False)
        print("Reductions applied:", [name for name, _ in trace])
        self.assertTrue(any("isolated" in name.lower() for name, _ in trace), "Isolated vertex reduction should have been applied.")
        self.assertTrue(any("degree_two_folding" in name.lower() for name, _ in trace), "Degree two folding should have been applied.")
        self.assertTrue(any("twin_removal" in name.lower() for name, _ in trace), "Twin removal should have been applied.")
        self.assertTrue(any("twin_folding" in name.lower() for name, _ in trace), "Twin folding should have been applied.")
        self.assertTrue(any("domination" in name.lower() for name, _ in trace), "Domination reduction should have been applied.")
        self.assertTrue(any("crown" in name.lower() for name, _ in trace), "Crown reduction should have been applied.")


# Ensure the reductions module is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)


# Define the test suite
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestIsolatedVertexReduction))
    suite.addTest(unittest.makeSuite(TestDegreeTwoFolding))
    suite.addTest(unittest.makeSuite(TestTwinRemoval))
    suite.addTest(unittest.makeSuite(TestTwinFolding))
    suite.addTest(unittest.makeSuite(TestDominationReduction))
    suite.addTest(unittest.makeSuite(TestCrownReduction))
    suite.addTest(unittest.makeSuite(TestAllReductions))
    return suite


# Create a test runner
def run_tests():
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())


# If this script is run directly, execute the tests
if __name__ == '__main__':
    run_tests()




# old code
"""
import unittest
import networkx as nx
from reductions import apply_all_reductions

class TestReductions(unittest.TestCase):

    def test_isolated_vertex(self):
        G = nx.Graph()
        G.add_nodes_from([1, 2, 3])
        G.add_edge(1, 2)
        G_reduced, trace = apply_all_reductions(G.copy(), verbose=False, timing=False)
        print("Reductions applied (isolated_vertex):", [name for name, _ in trace])
        self.assertFalse(3 in G_reduced.nodes)
        self.assertTrue(any("isolated" in name.lower() for name, _ in trace))

###
    def test_degree_two_folding(self):
        G = nx.Graph()
        G.add_edges_from([(1, 2), (2, 3)])
        G_reduced, trace = apply_all_reductions(G.copy(), verbose=False, timing=False)
        print("Reductions applied (degree_two_folding):", [name for name, _ in trace])
        # Prüfe, dass 2, 1 und 3 entfernt wurden und ein Fold-Knoten existiert
        fold_nodes = [n for n in G_reduced.nodes if str(n).startswith("fold_")]
        self.assertTrue(len(fold_nodes) == 1)
        self.assertNotIn(2, G_reduced.nodes)
        self.assertNotIn(1, G_reduced.nodes)
        self.assertNotIn(3, G_reduced.nodes)
###

def test_degree_two_folding(self):
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3)])
    G_reduced, trace = apply_all_reductions(G.copy(), verbose=False, timing=False)
    print("Reductions applied (degree_two_folding):", [name for name, _ in trace])
    print("Nodes after reduction:", G_reduced.nodes)

    # fold_nodes = [n for n in G_reduced.nodes if str(n).startswith("fold_")]
    # print("Fold nodes:", fold_nodes)
    # self.assertTrue(len(fold_nodes) == 1)
    # self.assertNotIn(2, G_reduced.nodes)
    # self.assertNotIn(1, G_reduced.nodes)
    # self.assertNotIn(3, G_reduced.nodes)

    def test_twin_removal(self):
        G = nx.Graph()
        G.add_edges_from([(1, 4), (2, 4), (3, 4)])
        G.add_edges_from([(1, 5), (2, 5), (3, 5)])
        G.add_edges_from([(1, 6), (2, 6), (3, 6)])
        G.add_edges_from([(1, 2)])
        G_reduced, trace = apply_all_reductions(G.copy(), verbose=False, timing=False)
        print("Reductions applied (twin_removal):", [name for name, _ in trace])
        self.assertLess(len(G_reduced.nodes), 6)

    def test_domination(self):
        G = nx.Graph()
        G.add_edges_from([(1, 3), (2, 3), (1, 2)])
        G_reduced, trace = apply_all_reductions(G.copy(), verbose=False, timing=False)
        print("Reductions applied (domination):", [name for name, _ in trace])
        self.assertLess(len(G_reduced.nodes), 3)

    def test_twin_folding(self):
        G = nx.Graph()
        G.add_edges_from([(1, 4), (1, 5), (1, 6),
                          (2, 4), (2, 5), (2, 6)])
        G_reduced, trace = apply_all_reductions(G.copy(), verbose=False, timing=False)
        print("Reductions applied (twin_folding):", [name for name, _ in trace])
        twin_folded = any("twin_folding" in name.lower() for name, _ in trace)
        self.assertTrue(twin_folded)

    def test_crown_reduction(self):
        G = nx.Graph()
        G.add_edges_from([(1, 4), (2, 4), (3, 4), (1, 5), (2, 5), (3, 5)])
        G_reduced, trace = apply_all_reductions(G.copy(), verbose=False, timing=False)
        print("Reductions applied (crown_reduction):", [name for name, _ in trace])
        crown_reduced = any("crown" in name.lower() for name, _ in trace)
        self.assertTrue(crown_reduced)

if __name__ == '__main__':
    unittest.main()

"""