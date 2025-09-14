import unittest
import sys
import os
import networkx as nx
import matplotlib


# Ask once at the beginning whether plots should be shown interactively
SHOW_PLOTS = False
try:
    answer = input("Do you want to see plots interactively? (y/n): ").strip().lower()
    if answer in ("y", "yes"):
        SHOW_PLOTS = True
except EOFError:
    # This happens if input() isn't available (like in some CI systems)
    SHOW_PLOTS = False

# Choose backend BEFORE importing pyplot
def _select_backend(show: bool) -> str:
    if show:
        # Try interactive backends in order
        for cand in ("MacOSX", "TkAgg", "Qt5Agg", "QtAgg"):
            try:
                matplotlib.use(cand, force=True)
                return cand
            except Exception:
                pass
    # Fallback: non-interactive
    matplotlib.use("Agg", force=True)
    return "Agg"

BACKEND = _select_backend(SHOW_PLOTS)
print(f"[matplotlib backend] {BACKEND}")

import matplotlib.pyplot as plt


# import all necessary reduction functions from the reductions module
from reductions import (
    apply_isolated_vertex_reduction,
    apply_degree_two_folding,
    apply_twin_folding_or_removal,
    apply_domination_reduction,
    apply_crown_reduction,
    apply_all_reductions
)

# ---------- Visualization helpers ----------

def _ensure_plots_dir():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reductions_visual_plots")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def draw_graph(G, title, pos=None, highlight_removed=None, highlight_added=None, fig_key=""):
    """
    Draw a graph with optional highlighting:
      - highlight_removed: nodes to draw in red (used in BEFORE plot)
      - highlight_added: nodes to draw in green (used in AFTER plot)
    """
    plt.figure(figsize=(7.5, 5.0))
    if pos is None:
        pos = nx.spring_layout(G, seed=42)

    # Base nodes (not highlighted)
    base_nodes = list(G.nodes())
    base_colors = []
    for n in base_nodes:
        if highlight_removed and n in highlight_removed:
            base_colors.append("red")
        elif highlight_added and n in highlight_added:
            base_colors.append("green")
        else:
            base_colors.append("lightblue")

    nx.draw_networkx(
        G,
        pos=pos,
        with_labels=True,
        node_color=base_colors,
        edge_color="gray",
        font_size=9
    )
    plt.title(title)
    plt.tight_layout()

    # Show if interactive, always save to file
    out_dir = _ensure_plots_dir()
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in title)
    fname = os.path.join(out_dir, f"{safe}{('_' + fig_key) if fig_key else ''}.png")
    plt.savefig(fname, dpi=140)

    if SHOW_PLOTS and BACKEND.lower() != "agg":
        plt.show()

    plt.close()
    print(f"[saved plot] {fname}")

def visualize_before_after(G_before, G_after, label):
    """
    Create consistent layout across before/after by laying out the union graph.
    Highlight nodes removed (red) in BEFORE, and nodes added (green) in AFTER.
    """
    union = nx.compose(G_before, G_after)
    pos = nx.spring_layout(union, seed=42)

    removed = set(G_before.nodes()) - set(G_after.nodes())
    added = set(G_after.nodes()) - set(G_before.nodes())

    draw_graph(G_before, f"{label} — BEFORE", pos=pos, highlight_removed=removed, fig_key="before")
    draw_graph(G_after, f"{label} — AFTER", pos=pos, highlight_added=added, fig_key="after")


# ---------- Tests ----------

# Define the test case class
class TestIsolatedVertexReduction(unittest.TestCase):
    def setUp(self):
        # Create a simple graph with isolated vertices
        self.G = nx.Graph()
        self.G.add_edges_from([(1, 2), (2, 3)])
        self.G.add_node(4)  # Isolated vertex
        self.G.add_node(5)  # Another isolated vertex

    def test_isolated_vertex_reduction(self):
        G_reduced, changed, removed, VCC_addition = apply_isolated_vertex_reduction(self.G.copy())
        print("Reductions applied (isolated vertex):", removed)

        # Visualize
        visualize_before_after(self.G, G_reduced, "Isolated Vertex Reduction")

        self.assertTrue(changed, "Graph should have changed due to isolated vertex removal.")
        self.assertIn(4, removed, "Isolated vertex 4 should have been removed.")
        self.assertIn(5, removed, "Isolated vertex 5 should have been removed.")
        self.assertNotIn(4, G_reduced.nodes, "Isolated vertex 4 should not be in the reduced graph.")
        self.assertNotIn(5, G_reduced.nodes, "Isolated vertex 5 should not be in the reduced graph.")


class TestDegreeTwoFolding(unittest.TestCase):
    def setUp(self):
        # Create a simple graph with a degree-2 foldable node
        self.G = nx.Graph()
        self.G.add_edges_from([(1, 2), (1, 3)])
        self.G.add_edge(2, 4)
        self.G.add_edge(2, 5)
        self.G.add_edge(3, 6)
        self.G.add_edge(3, 7)

    def test_degree_two_folding(self):
        G_folded, changed, folds, VCC_addition = apply_degree_two_folding(self.G.copy())
        print("Reductions applied (degree two folding):", folds)

        # Visualize
        visualize_before_after(self.G, G_folded, "Degree-2 Folding")

        self.assertTrue(changed, "Graph should have changed due to folding.")
        self.assertEqual(len(folds), 1, "One fold should have occurred.")

        v, u, w = folds[0]
        self.assertIn(f"fold_{v}", G_folded.nodes, "Folded node should exist.")
        folded_neighbors = set(G_folded.neighbors(f"fold_{v}"))
        self.assertEqual(folded_neighbors, {4, 5, 6, 7}, "Folded node should connect to external neighbors of u and w.")

        for node in (v, u, w):
            self.assertNotIn(node, G_folded.nodes, f"Node {node} should have been removed.")


class TestTwinRemoval(unittest.TestCase):
    def setUp(self):
        # Create a graph with twin nodes
        self.G = nx.Graph()
        self.G.add_edges_from([(1, 4), (2, 4), (3, 4)])
        self.G.add_edges_from([(1, 5), (2, 5), (3, 5)])
        self.G.add_edge(1, 2)
        self.G.add_edges_from([(1, 6), (2, 7)])

    def test_twin_removal(self):
        G_reduced, changed, removed, VCC_addition = apply_twin_folding_or_removal(self.G.copy())
        print("Reductions applied (twin removal):", removed)

        # Visualize
        visualize_before_after(self.G, G_reduced, "Twin Removal")

        self.assertTrue(changed, "Graph should have changed due to twin removal.")
        self.assertLess(len(G_reduced.nodes), 5, "Graph should have fewer than 6 nodes after twin removal.")
        self.assertNotIn(4, G_reduced.nodes, "Node 4 should have been removed as a twin.")
        self.assertNotIn(5, G_reduced.nodes, "Node 5 should have been removed as a twin.")


class TestTwinFolding(unittest.TestCase):
    def setUp(self):
        # Create a graph with twin nodes
        self.G = nx.Graph()
        self.G.add_edges_from([(1, 4), (1, 5), (1, 6),
                               (2, 4), (2, 5), (2, 6)])
        self.G.add_edges_from([(4, 7), (5, 8)])
        # self.G.add_edges_from([(7, 8)]) # this should lead to no reduction because of the crossing independence

    def test_twin_folding(self):
        G_reduced, changed, folds, VCC_addition = apply_twin_folding_or_removal(self.G.copy())
#        print("Reductions applied (twin_folding):", [name for name, _ in folds])
        print("Folded nodes:", folds)
        print(f"edges after folding: {G_reduced.edges}")
        self.assertTrue(changed, "Graph should have changed due to twin folding.")
#        self.assertTrue(any("twin_folding" in name.lower() for name, _ in folds), "Twin folding should have been applied.")
        self.assertTrue(len(folds) > 0, "Twin folding should have been applied.")


class TestDominationReduction(unittest.TestCase):
    def setUp(self):
        # Create a graph for domination reduction
        self.G = nx.Graph()
        self.G.add_edges_from([(1, 3), (2, 3), (1, 2)])

    def test_domination(self):
        G_reduced, changed, removed, VCC_addition = apply_domination_reduction(self.G.copy())
        print("Reductions applied (domination):", [name for name, _ in removed])

        # Visualize
        visualize_before_after(self.G, G_reduced, "Domination Reduction")

        self.assertTrue(changed, "Graph should have changed due to domination reduction.")
        self.assertLess(len(G_reduced.nodes), 3, "Graph should have fewer than 3 nodes after domination reduction.")


class TestCrownReduction(unittest.TestCase):
    def setUp(self):
        # Create a graph for crown reduction
        self.G = nx.Graph()
        self.G.add_edges_from([
            (70, 80), (71, 80), (72, 80), (73, 80), (74, 80),
            (70, 81), (71, 81), (72, 81), (73, 81), (74, 81),
            (70, 82), (71, 82), (72, 82), (73, 82), (74, 82),
            (70, 83), (71, 83), (72, 83), (73, 83), (74, 83)
        ])
        self.G.add_node(1)
        self.G.add_node(2)

    def test_crown_reduction(self):
        print("Initial graph nodes:", self.G.nodes)
        print("Initial graph edges:", self.G.edges)
        G_reduced, changed, removed_or_sets, VCC_addition = apply_crown_reduction(self.G.copy())
        print(f"removed / crown_sets: {removed_or_sets}")
        print(f"clique cover addition: {VCC_addition}")
        print(f"remaining edges: {G_reduced.edges}")

        # Visualize
        visualize_before_after(self.G, G_reduced, "Crown Reduction 2")

        self.assertTrue(changed, "Graph should have changed due to crown reduction.")
        # accept either representation; ensure some change happened
        self.assertTrue(len(set(self.G.nodes()) - set(G_reduced.nodes())) > 0,
                        "Crown reduction should have removed some nodes.")


# Ensure the reductions module is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Define the test suite
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestIsolatedVertexReduction))
    suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestDegreeTwoFolding))
    suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestTwinRemoval))
    suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestTwinFolding))
    suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestDominationReduction))
    suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestCrownReduction))
    return suite

# Create a test runner
def run_tests():
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())

# If this script is run directly, execute the tests
if __name__ == '__main__':
    unittest.main(verbosity=2)
