""" Das Originalproblem ist:
        Finde eine Partition der Knoten in möglichst wenige Cliquen (Vertex Clique Cover).
  Dieser Code:
        Nutzt eine Menge maximaler Cliquen (potenziell überlappend) und versucht, diese mit Farben zu überdecken — nicht notwendigerweise disjunkt!
 evtl schwierig weil:
    Eine Clique Cover muss aus disjunkten Cliquen bestehen.
    hier werden alle maximalen Cliquen verwendet, die sich überlappen dürfen, und es wird auf Constraints gesetzt, dass überlappende nicht gleiche Farbe bekommen
    → das führt nicht zwingend zu einer Cover-Partition.
"""
import networkx as nx
import pulp
from typing import List, Set, Optional

class ILPCliqueCover:
    """
    ILP formulation for the vertex clique coloring problem.
    """

    def __init__(self, G: nx.Graph, max_colors: Optional[int] = None,
                time_limit: int = 300, verbose: bool = False):
        """
        Initialize the ILP solver.

        Args:
            G: NetworkX graph to color
            max_colors: Maximum number of colors to use (default: number of nodes)
            time_limit: Time limit in seconds (default: 5 minutes)
            verbose: Whether to print verbose output
        """
        self.G = G
        self.num_nodes = G.number_of_nodes()
        self.max_colors = max_colors if max_colors is not None else self.num_nodes
        self.time_limit = time_limit
        self.verbose = verbose

        # Find all maximal cliques in the graph
        self.maximal_cliques = list(nx.find_cliques(G))
        self.num_cliques = len(self.maximal_cliques)

        if self.verbose:
            print(f"Graph has {self.num_nodes} nodes, {G.number_of_edges()} edges, and {self.num_cliques} maximal cliques.")

    def solve(self) -> List[Set[int]]:
        """
        Solve the vertex clique coloring problem using ILP.

        Returns:
            List of cliques (each represented as a set of nodes)
        """
        # Initialize problem
        prob = pulp.LpProblem("Vertex_Clique_Coloring", pulp.LpMinimize)

        # Create variables
        # x[i][c] = 1 if clique i is assigned color c, 0 otherwise
        x = {}
        for i in range(self.num_cliques):
            for c in range(self.max_colors):
                x[i, c] = pulp.LpVariable(f"x_{i}_{c}", cat=pulp.LpBinary)

        # y[c] = 1 if color c is used, 0 otherwise
        y = {}
        for c in range(self.max_colors):
            y[c] = pulp.LpVariable(f"y_{c}", cat=pulp.LpBinary)

        # Objective: Minimize the number of colors used
        prob += pulp.lpSum(y[c] for c in range(self.max_colors))

        # Constraint: Each clique must be assigned exactly one color
        for i in range(self.num_cliques):
            prob += pulp.lpSum(x[i, c] for c in range(self.max_colors)) == 1

        # Constraint: If a color is used, y[c] must be 1
        for c in range(self.max_colors):
            for i in range(self.num_cliques):
                prob += x[i, c] <= y[c]

        # Constraint: Overlapping cliques must have different colors
        for i in range(self.num_cliques):
            for j in range(i+1, self.num_cliques):
                # Check if cliques i and j overlap
                if not set(self.maximal_cliques[i]).isdisjoint(set(self.maximal_cliques[j])):
                    for c in range(self.max_colors):
                        prob += x[i, c] + x[j, c] <= 1

        # Set time limit
        if self.time_limit > 0:
            prob.solve(pulp.PULP_CBC_CMD(timeLimit=self.time_limit, msg=self.verbose))
        else:
            prob.solve(pulp.PULP_CBC_CMD(msg=self.verbose))

        if self.verbose:
            print(f"Status: {pulp.LpStatus[prob.status]}")

        # Check if a solution was found
        if prob.status != pulp.LpStatusOptimal:
            print("Warning: ILP solver did not find an optimal solution.")
            if prob.status == pulp.LpStatusInfeasible:
                print("The problem is infeasible. Try increasing max_colors.")
                return []

        # Extract the solution
        used_colors = 0
        for c in range(self.max_colors):
            if pulp.value(y[c]) == 1:
                used_colors += 1

        if self.verbose:
            print(f"Optimal solution uses {used_colors} colors.")

        # Group cliques by color
        color_to_cliques = {}
        for c in range(self.max_colors):
            if pulp.value(y[c]) == 1:
                color_to_cliques[c] = []
                for i in range(self.num_cliques):
                    if pulp.value(x[i, c]) == 1:
                        color_to_cliques[c].append(set(self.maximal_cliques[i]))

        # Convert to list of cliques
        result = []
        for c in range(self.max_colors):
            if c in color_to_cliques:
                # If there are multiple cliques with the same color, merge them
                if len(color_to_cliques[c]) > 1:
                    merged = set()
                    for clique in color_to_cliques[c]:
                        merged.update(clique)
                    result.append(merged)
                elif len(color_to_cliques[c]) == 1:
                    result.append(color_to_cliques[c][0])

        return result
