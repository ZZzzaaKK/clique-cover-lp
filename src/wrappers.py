from utils import txt_to_networkx
import networkx as nx
from algorithms.chalupa import ChalupaHeuristic
from algorithms.ilp_solver import solve_ilp_clique_cover
from reductions.reductions import apply_all_reductions

def reduced_ilp_wrapper(txt_filepath, problem_type="vertex_clique_cover", time_limit=60):
    """
    Wrapper for reduction followed by ILP
        1. Estimate upper bound k on clique cover number θ(G)
        2. Run data reduction
        3. Run ILP on reduced graph

    Args:
        problem_type: "vertex_clique_cover" or "chromatic_number"
    """
    try:
        print(f"{txt_filepath}")
        G = txt_to_networkx(txt_filepath)

        if problem_type == "chromatic_number":
            # For chromatic number, work with the original graph
            G_complement_reduced, trace, chromatic_number_addition = apply_all_reductions(nx.complement(G))
            result = solve_ilp_clique_cover(nx.complement(G_complement_reduced), time_limit=time_limit)
            if 'error' in result:
                print(f"ILP failed on {txt_filepath}: {result['error']}")
                return None, False
            return int(result['chromatic_number']) + chromatic_number_addition, result['optimal']
        else:
            # For vertex clique cover, work with the complement
            G_reduced, trace, vcc_addition = apply_all_reductions(G)
            result = solve_ilp_clique_cover(nx.complement(G_reduced), time_limit=time_limit)
            if 'error' in result:
                print(f"ILP failed on {txt_filepath}: {result['error']}")
                return None, False
            return int(result['chromatic_number']) + vcc_addition, result['optimal']
    except Exception as e:
        print(f"ILP failed on {txt_filepath}: {e}")
        return None, False

def interactive_reduced_ilp_wrapper(txt_filepath, problem_type="vertex_clique_cover", time_limit=60):
    """
    Wrapper for interactive reduction followed by ILP
        1. Estimate upper bound k on clique cover number θ(G)
        2. Run data reduction
        3. Repeat until no further reductions are possible
        4. Run ILP on reduced graph

    Args:
        problem_type: "vertex_clique_cover" or "chromatic_number"
    """
    try:
        print(f"{txt_filepath}")
        G = txt_to_networkx(txt_filepath)
        upper_bound = float('inf')
        # just any value lower than infinity
        current_upper_bound = 0
        total_vcc_addition = 0
        while current_upper_bound < upper_bound:
            chalupa = ChalupaHeuristic(nx.complement(G))
            best_clique_covering = chalupa.iterated_greedy_clique_covering()
            upper_bound = current_upper_bound
            current_upper_bound = len(best_clique_covering) if best_clique_covering else float('inf')
            G, trace, vcc_addition = apply_all_reductions(G)
            total_vcc_addition += vcc_addition

        if problem_type == "chromatic_number":
            # For chromatic number, work with the graph directly
            result = solve_ilp_clique_cover(G, time_limit=time_limit)
        else:
            # For vertex clique cover, work with the complement
            result = solve_ilp_clique_cover(nx.complement(G), time_limit=time_limit)

        if 'error' in result:
            print(f"ILP failed on {txt_filepath}: {result['error']}")
            return None, False
        return int(result['chromatic_number']) + total_vcc_addition, result['optimal']
    except Exception as e:
        print(f"ILP failed on {txt_filepath}: {e}")
        return None, False

def reduced_chalupa_wrapper(txt_filepath, problem_type="vertex_clique_cover"):
    """Wrapper for Reduced Chalupa algorithm

    Args:
        problem_type: "vertex_clique_cover" or "chromatic_number"
    """
    try:
        print(f"{txt_filepath}")
        G = txt_to_networkx(txt_filepath)

        if problem_type == "chromatic_number":
            G_complement_reduced, trace, chromatic_number_addition = apply_all_reductions(nx.complement(G))
            print(G_complement_reduced)
            chalupa = ChalupaHeuristic(G_complement_reduced)
            result = chalupa.run()
            return result['upper_bound'] + chromatic_number_addition, True
        else:
            G_reduced, trace, vcc_addition = apply_all_reductions(G)
            chalupa = ChalupaHeuristic(G_reduced)
            result = chalupa.run()
            return result['upper_bound'] + vcc_addition, True

    except Exception as e:
        print(f"Reduced Chalupa failed on {txt_filepath}: {e}")
        return None, False

def chalupa_wrapper(txt_filepath, problem_type="vertex_clique_cover"):
    """Wrapper for Chalupa algorithm

    Args:
        problem_type: "vertex_clique_cover" or "chromatic_number"
    """
    try:
        print(f"{txt_filepath}")
        G = txt_to_networkx(txt_filepath)

        if problem_type == "chromatic_number":
            # For vertex clique cover, run Chalupa on the complement
            chalupa = ChalupaHeuristic(nx.complement(G))
        else:
            # For chromatic number, run Chalupa on the original graph
            chalupa = ChalupaHeuristic(G)

        result = chalupa.run()
        return result['upper_bound'], True  # Chalupa is a heuristic, so we consider its result 'optimal' for its own execution
    except Exception as e:
        print(f"Chalupa failed on {txt_filepath}: {e}")
        return None, False

def ilp_wrapper(txt_filepath, problem_type="vertex_clique_cover", require_optimal=False, time_limit=60):
    """Wrapper for ILP solver

    Args:
        problem_type: "vertex_clique_cover" or "chromatic_number"
    """
    try:
        print(f"{txt_filepath}")
        G = txt_to_networkx(txt_filepath)

        if problem_type == "chromatic_number":
            # For chromatic number, work with the original graph
            result = solve_ilp_clique_cover(G, require_optimal=require_optimal, time_limit=time_limit)
        else:
            # For vertex clique cover, work with the complement
            result = solve_ilp_clique_cover(nx.complement(G), require_optimal=require_optimal, time_limit=time_limit)

        if 'error' in result:
            print(f"ILP failed on {txt_filepath}: {result['error']}")
            return None, False
        return result['chromatic_number'], result['optimal']
    except Exception as e:
        print(f"ILP failed on {txt_filepath}: {e}")
        return None, False
