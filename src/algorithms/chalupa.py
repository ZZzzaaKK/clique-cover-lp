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

import pickle
import networkx as nx
import random
import numpy as np
from typing import List, Set, Dict
from helpers import jump, random_permutation, uniformly_random

class ChalupaHeuristic:
    def load_graph(self, path):
        with open(path, "rb") as f:
            G = pickle.load(f)
            self.V = list(G.nodes())
            self.E = list(G.edges())
            self.n = len(self.V)
            self.upper = None

    def estimate_upper_bound(self):
        self.current_best_solution = self.iterated_greedy()
        self.upper = self.current_best_solution

    def estimate_lower_bound(self):
        self.lower = self.find_maximum_independent_set_size()

    def find_greedy_clique_covering(self, permutation):
        """
        Transforms a permutation of vertices into a clique covering.

        Args:
            permutation: A permutation ordering to use on the list of vertices
        Returns:
            A greedy partitioning of the graph into cliques
        """
        number_of_vertices = len(self.V)

        for c in range(number_of_vertices):


    def iterated_greedy(self):
        permutation = random_permutation(self.V)
        iteration = 0
        stopping_criterion = iteration >= 1000

        while not stopping_criterion:
            cliques = self.find_greedy_clique_covering()
            if self.upper is not None and self.upper == len(cliques):
                return cliques
            permutation = random_permutation(cliques.flatten())
            iteration += 1

    def greedy_independent_set(self, permutation):

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
