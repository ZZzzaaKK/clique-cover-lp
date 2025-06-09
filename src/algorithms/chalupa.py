"""
Implementation of Chalupa's heuristic algorithm for clique coloring.

References:
[4] David Chalupa. On the effectiveness of the genetic algorithm for the
    clique coloring problem. Communications in Mathematics and Computer
    Science 1(1), 2023.

[8] David Chalupa. A genetic algorithm with neighborhood search for the
    generalized graph coloring problem. Information Sciences, 602:
    91-108, 2022.
"""

import itertools
import pickle
import numpy as np
from helpers import jump, random_permutation, uniformly_random

class ChalupaHeuristic:
    def __init__(self, path):
        with open(path, "rb") as f:
            G = pickle.load(f)
            self.V = list(G.nodes())
            self.E = list(G.edges())
            self.n = len(self.V)
            self.node_labels = np.zeros(self.n)
            self.upper_bound = None
            self.lower_bound = None
            self.best_clique_covering = None
            self.best_independent_set = None

    def run(self):
        """
        Main algorithm that implements Chalupa's heuristic.

        Returns a tuple containing:
        - Lower bound (maximum independent set size)
        - Upper bound (number of cliques in best covering)
        - Best clique covering found
        - Best independent set found
        """
        # Step 1: Find upper bound using Iterated Greedy heuristic for clique covering
        print("Finding upper bound using Iterated Greedy heuristic...")
        self.best_clique_covering = self.iterated_greedy_clique_covering()
        self.upper_bound = len(self.best_clique_covering) if self.best_clique_covering else float('inf')

        # Step 2: Find lower bound using Randomized Local Search for maximum independent set
        print("Finding lower bound using Randomized Local Search...")
        self.lower_bound, self.best_independent_set = self.find_maximum_independent_set()

        # Return results
        return {
            'lower_bound': self.lower_bound,
            'upper_bound': self.upper_bound,
            'clique_covering': self.best_clique_covering,
            'independent_set': self.best_independent_set,
            'bounds_interval': f"[{self.lower_bound}, {self.upper_bound}]"
        }

    def get_neighbors(self, vertex):
        """Compute neighbors from edge list - O(|E|) time"""
        neighbors = []
        for u, v in self.E:
            if u == vertex:
                neighbors.append(v)
            elif v == vertex:
                neighbors.append(u)
        return neighbors

    def count_neighbors_of_vertex_with_label(self, vertex, label):
        neighbors = self.get_neighbors(vertex)
        count = sum(1 for neighbor in neighbors if self.node_labels[neighbor] == label)
        return count

    def find_equal(self, vertex, sizes):
        """
        Find the lowest-labeled clique that vertex can join, using First Fit strategy.

        Algorithm from paper:
        1. Scan neighbors of vertex
        2. For each labeled neighbor, decrement sizes[neighbor_label]
        3. If sizes[c] reaches 0, then c is a candidate (vertex connects to all vertices in clique c)
        4. Choose lowest candidate c
        5. Restore original sizes values

        Args:
            vertex: The vertex to assign to a clique
            sizes: Array tracking size of each clique

        Returns:
            Label (index) of the clique to assign vertex to
        """
        neighbors = self.get_neighbors(vertex)

        # Store original values and track which ones we modified
        original_sizes = {}
        modified_labels = set()

        # Step 1-2: Scan neighbors and decrement sizes for labeled neighbors
        for neighbor in neighbors:
            neighbor_label = int(self.node_labels[neighbor])

            # Only process neighbors that have been assigned to a clique
            if neighbor_label >= 0:  # Assuming -1 or 0 means unassigned
                if neighbor_label not in original_sizes:
                    original_sizes[neighbor_label] = sizes[neighbor_label]

                sizes[neighbor_label] -= 1
                modified_labels.add(neighbor_label)

        # Step 3-4: Find candidates (where sizes[c] == 0) and choose lowest
        candidates = []
        for label in modified_labels:
            if sizes[label] == 0:
                candidates.append(label)

        # Choose the lowest candidate (First Fit strategy)
        if candidates:
            chosen_label = min(candidates)
        else:
            # No existing clique can accommodate this vertex, create new one
            chosen_label = len([s for s in sizes if s > 0])  # Next available label

        # Step 5: Restore original sizes values
        for label in modified_labels:
            sizes[label] = original_sizes[label]

        return chosen_label

    def find_greedy_clique_covering(self, permutation):
        """
        Transforms a permutation of vertices into a clique covering.

        Args:
            permutation: A permutation ordering to use on the list of vertices
        Returns:
            A greedy partitioning of the graph into cliques
        """
        number_of_vertices = len(self.V)
        sizes = np.zeros(number_of_vertices, dtype=int)
        cliques = []

        # Initialize node labels to -1 (unassigned)
        self.node_labels = np.full(self.n, -1, dtype=int)

        for i, current_vertex in enumerate(permutation):
            # Find which clique this vertex should join
            label = self.find_equal(current_vertex, sizes)

            # Extend cliques list if needed
            while len(cliques) <= label:
                cliques.append([])

            # Add vertex to the appropriate clique
            cliques[label].append(current_vertex)
            sizes[label] += 1
            self.node_labels[current_vertex] = label

        return cliques

    def iterated_greedy_clique_covering(self):
        permutation = random_permutation(self.V)
        iteration = 0
        stopping_criterion = iteration >= 1000

        while not stopping_criterion:
            cliques = self.find_greedy_clique_covering(permutation)
            if self.upper is not None and self.upper == len(cliques):
                return cliques
            # TODO: This permutation probably shouldn't be completely random. But how to do it exactly?
            # There's more on this in the chalupa paper: On the Efficiency of an Order-based Representation in the
            # Clique Covering Problem (2012)
            permutation = random_permutation(list(itertools.chain.from_iterable(cliques)))
            iteration += 1

    def greedy_independent_set(self, permutation):
        """
        Maps a permutation of vertices to an independent set using a greedy algorithm.

        Algorithm from paper:
        1. Start with empty independent set
        2. Initialize a(v) = 1 for all vertices (all can be added initially)
        3. Process vertices in permutation order:
           - If a(v) = 1, add v to independent set
           - Set a(w) = 0 for all neighbors w of v
        4. Continue until all vertices processed

        Args:
            permutation: Ordering of vertices to process

        Returns:
            List of vertices forming an independent set
        """
        # Initialize Boolean function a(v) - True means vertex can be added
        can_add = {v: True for v in self.V}

        # Start with empty independent set
        independent_set = []

        # Process vertices in permutation order
        for vertex in permutation:
            # If vertex can be added (not adjacent to any vertex in current set)
            if can_add[vertex]:
                # Add vertex to independent set
                independent_set.append(vertex)

                # Set a(w) = 0 for all neighbors w of vertex
                neighbors = self.get_neighbors(vertex)
                for neighbor in neighbors:
                    can_add[neighbor] = False

        return independent_set

    def find_maximum_independent_set_size(self):
        permutation = random_permutation(self.V)
        lower_bound = 1
        lower_bound_permutation = permutation
        iteration = 0
        stopping_criterion = iteration >= 1000

        while not stopping_criterion:
            k = len(self.greedy_independent_set(permutation))
            if k >= lower_bound:
                lower_bound = k
                lower_bound_permutation = permutation
            j = uniformly_random(1, self.V)
            permutation = jump(j, 0, lower_bound_permutation)
            iteration += 1

        return lower_bound
