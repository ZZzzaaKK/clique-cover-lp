from utils import txt_to_networkx
import networkx as nx
from algorithms.chalupa import ChalupaHeuristic
from algorithms.ilp_solver import solve_ilp_clique_cover
from reductions.reductions import apply_all_reductions

def reduced_ilp_wrapper(txt_filepath):
    """
    Wrapper for reduction followed by ILP
        1. Estimate upper bound k on clique cover number θ(G)
        2. Run data reduction
        3. Run ILP on reduced graph
    """
    try:
        print(f"{txt_filepath}")
        G = txt_to_networkx(txt_filepath)
        # TODO: How to make use of the upper bound?
        # chalupa = ChalupaHeuristic(nx.complement(G))
        # best_clique_covering = chalupa.iterated_greedy_clique_covering()
        # upper_bound = len(best_clique_covering) if best_clique_covering else float('inf')
        G_reduced, trace, VCC_addition_total = apply_all_reductions(G)
        result = solve_ilp_clique_cover(G_reduced)
        if 'error' in result:
            print(f"ILP failed on {txt_filepath}: {result['error']}")
            return None
        return result['chromatic_number']
    except Exception as e:
        print(f"ILP failed on {txt_filepath}: {e}")
        return None

def interactive_reduced_ilp_wrapper(txt_filepath):
    """
    Wrapper for interactive reduction followed by ILP
        1. Estimate upper bound k on clique cover number θ(G)
        2. Run data reduction
        3. Repeat until no further reductions are possible
        4. Run ILP on reduced graph
    """
    try:
        print(f"{txt_filepath}")
        G = txt_to_networkx(txt_filepath)
        upper_bound = float('inf')
        # just any value lower than infinity
        current_upper_bound = 0
        while current_upper_bound < upper_bound:
            chalupa = ChalupaHeuristic(nx.complement(G))
            best_clique_covering = chalupa.iterated_greedy_clique_covering()
            upper_bound = current_upper_bound
            current_upper_bound = len(best_clique_covering) if best_clique_covering else float('inf')
            G, trace, VCC_addition_total = apply_all_reductions(G)
        result = solve_ilp_clique_cover(G)
        if 'error' in result:
            print(f"ILP failed on {txt_filepath}: {result['error']}")
            return None
        return result['chromatic_number']
    except Exception as e:
        print(f"ILP failed on {txt_filepath}: {e}")
        return None

def chalupa_wrapper(txt_filepath):
    """Wrapper for Chalupa algorithm"""
    try:
        print(f"{txt_filepath}")
        G = txt_to_networkx(txt_filepath)
        chalupa = ChalupaHeuristic(nx.complement(G))
        result = chalupa.run()
        return result['upper_bound']
    except Exception as e:
        print(f"Chalupa failed on {txt_filepath}: {e}")
        return None

def ilp_wrapper(txt_filepath):
    """Wrapper for ILP solver"""
    try:
        print(f"{txt_filepath}")
        G = txt_to_networkx(txt_filepath)
        result = solve_ilp_clique_cover(G)
        if 'error' in result:
            print(f"ILP failed on {txt_filepath}: {result['error']}")
            return None
        return result['chromatic_number']
    except Exception as e:
        print(f"ILP failed on {txt_filepath}: {e}")
        return None
