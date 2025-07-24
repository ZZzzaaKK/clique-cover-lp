from utils import txt_to_networkx
import networkx as nx
from algorithms.chalupa import ChalupaHeuristic
from algorithms.ilp_solver import solve_ilp_clique_cover

def chalupa_wrapper(txt_filepath):
    """Wrapper for Chalupa algorithm"""
    try:
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
        G = txt_to_networkx(txt_filepath)
        result = solve_ilp_clique_cover(G)
        if 'error' in result:
            print(f"ILP failed on {txt_filepath}: {result['error']}")
            return None
        return result['chromatic_number']
    except Exception as e:
        print(f"ILP failed on {txt_filepath}: {e}")
        return None
