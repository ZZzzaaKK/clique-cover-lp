"""
wrapperV2.py
WRAPPERS for running reduction, heuristic and ILP on clique-cover Instances...
THOUGHTS
--------------------- How to handle UB for best ILP Performance? ------------------------
Using the upper bound (UB) strictly is optional and depends on our goals:
- In the OG code (wrappers.py), `interactive_reduced_ilp_wrapper` uses UB from the Chalupa heuristic to guide reduction rounds, stopping when no improvement is made.
  This makes sense if we want to minimize problem size before ILP and avoid wasting effort on non-promising reductions.
- Strictly enforcing UB inside the ILP (e.g., as a constraint on the number of colors) could speed up solving by pruning solutions worse than the heuristic.
Here “hard equality” would mean forcing the ILP to *exactly* match the UB value (`sum(colors) == UB`). This is risky because if the heuristic UB is not optimal,
we might exclude the true optimal solution entirely.
- A safer approach imo is to use UB as a **hard inequality** (`sum(colors) ≤ UB`), which never eliminates the optimal solution but can still prune the search space.
- For simplified `reduced_ilp_wrapper`, we could pass UB into the ILP to potentially reduce runtime. This would involve adding an argument to the ILP solver
to accept a UB and incorporate it into the model.

In short: I would suggest to use UB as an upper limit (≤ UB) to contribute to better performance and avoid enforcing UB as an exact target (== UB) unless
we are certain it is optimal.

----------------------- Theory interlude --------------------------------
 χ(G) — Chromatic Number
    Definition: Die minimale Anzahl an Farben, die benötigt wird, um die Knoten von G so zu färben,
                dass keine zwei benachbarten Knoten dieselbe Farbe haben.

    Interpretation: Jede Farbe steht für eine unabhängige Menge (kein Kantenpaar innerhalb einer Farbe).
                    Färben = Partition von V(G) in minimale Anzahl unabhängiger Mengen.

    χ(G) = Größe der kleinsten Partition von V(G) in independent sets.

 θ(G) — Clique Cover Number
    Definition: Die minimale Anzahl an Cliquen, deren Vereinigung alle Knoten von G überdeckt.

    Interpretation: Jede Clique kann isoliert werden und deckt einen Teil der Knoten ab, so dass am Ende alle Knoten abgedeckt sind.
    Clique Cover = Partition von V(G) in minimale Anzahl vollständiger Teilgraphen.

    θ(G) = Größe der kleinsten Partition von V(G) in cliques.

 why important?
    - Cliquen in G ↔ unabhängige Mengen im Komplementgraphen Ḡ
    - Eine Clique in G ist ein independent set in Ḡ.
    - Eine independent set in G ist eine Clique in Ḡ.
    - im OG Code ist, so wie ich das verstanden habe, folgendes passiert: ILP hat χ(G) berechnet (Färbung von G) → und das war somit nicht die gesuchte Zahl für Clique Cover.
    - Bsp:
        - G: Dreieck (3 Knoten, alle verbunden): χ(G) = 3 (jeder Knoten andere Farbe), θ(G) = 1 (das Dreieck selbst ist eine Clique, deckt alle Knoten),
        - Ḡ: 3 isolierte Knoten: χ(Ḡ) = 1 → stimmt mit θ(G) überein.


    - ILP sollte jetzt mit dem folgenden Code Ḡ färben, wonach dann gilt: θ(G) = χ(Ḡ)
        (Clique Cover von G = Färbung des Komplements Ḡ).

------------------------------ θ(G) vs χ(Ḡ) ----------------------------------

What’s different now:
    - all wrappers operate on the complement graph Ḡ (consistent Clique Cover via θ(G)=χ(Ḡ)). should lead to correct results for Cloque Cover.?
    - reduced_ilp_wrapper applies reductions on Ḡ before ILP interactive_reduced_ilp_wrapper fixes the loop logic: keeps improving UB on Ḡ until it stops decreasing,
        then ILP on reduced Ḡ
    - interactive_reduced_ilp_wrapper fixes the loop logic: keeps improving UB on Ḡ until it stops decreasing, then ILP on reduced Ḡ
    - Added a chalupa_wrapper that also works on Ḡ (UB for θ(G))
    - ilp_wrapper(..., use_warmstart: bool)` toggles whether a Chalupa-based warmstart
        is computed and passed into the ILP solver. Default is False (fair ILP baseline).
    - Interoperability:
        - script uses `solve_ilp_clique_cover` from `algorithms.ilp_solver` which computes θ(G)
            by coloring the complement graph Ḡ (χ(Ḡ) = θ(G)).
        - If `use_warmstart=True` and Chalupa is available, a warmstart assignment is built.

Dependencies:
- utils.txt_to_networkx (reads graph from .txt)
- algorithms.ilp_solver.solve_ilp_clique_cover
- algorithms.chalupa.ChalupaHeuristic (optional; only if use_warmstart=True)
"""
import networkx as nx
from typing import Optional, Dict, Any, List
import os
import time

# Imports robust machen
try:
    from src.utils import txt_to_networkx
except ImportError:
    from utils import txt_to_networkx

try:
    from src.algorithms.ilp_solver import solve_ilp_clique_cover
except ImportError:
    from algorithms.ilp_solver import solve_ilp_clique_cover

try:
    from src.algorithms.chalupa import ChalupaHeuristic
except ImportError:
    from algorithms.chalupa import ChalupaHeuristic

try:
    from src.reductions.reductions import apply_all_reductions
except ImportError:
    from reductions.reductions import apply_all_reductions

# --- Config flags (optional, via ENV überschreibbar) ---
INTERACTIVE_MAX_ROUNDS = int(os.getenv("INTERACTIVE_MAX_ROUNDS", "10"))
USE_WARMSTART_DEFAULT = bool(int(os.getenv("USE_WARMSTART", "0")))

def _compact_int_labels(G: nx.Graph) -> nx.Graph:
    """
    Brings node labels safely to 0...n-1 (stable sorting)
    Important after reductions to maintain consistent array/ILP indexing
    """
    if G is None or G.number_of_nodes() == 0:
        return G
        # enumerate in deterministic insertion order
    mapping = {old: i for i, old in enumerate(list(G.nodes()))}
    return nx.relabel_nodes(G, mapping, copy=True)



def _is_valid_clique_cover(G: nx.Graph, color_of: dict) -> bool:
    """
    Validates whether the color classes in G form cliques (valid clique cover).
    color_of: dict[node] -> color_id
    """
    if not color_of:
        return False
    if set(color_of.keys()) != set(G.nodes()):
        return False

    # Group nodes by colors
    by_color = {}
    for v, c in color_of.items():
        by_color.setdefault(c, []).append(v)

    # Each color class must form a clique in G
    for color_id, nodes in by_color.items():
        for i, u in enumerate(nodes):
            for v in nodes[i + 1:]:
                if not G.has_edge(u, v):
                    return False
    return True


def _validate_result(G: nx.Graph, result: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
    """
    Validates the ILP result and adds validation information.
    """
    validated_result = result.copy()

    validated_result['validation'] = {
        'is_valid': False,
        'error': None,
        'num_cliques': None
    }

    try:
        if not isinstance(result, dict) or 'assignment' not in result or result['assignment'] is None:
            validated_result['validation']['error'] = "No assignment found in result"
            return validated_result

        assignment = result['assignment']
        if not assignment:
            validated_result['validation']['error'] = "Empty assignment"
            return validated_result

        # Convert assignment to color_of format for validation
        color_of = {}
        for color_id, nodes in assignment.items():
            for node in nodes:
                color_of[node] = color_id

        # Validate the clique cover property
        is_valid = _is_valid_clique_cover(G, color_of)
        validated_result['validation']['is_valid'] = is_valid
        validated_result['validation']['num_cliques'] = len(assignment)

        if not is_valid:
            validated_result['validation']['error'] = "Assignment does not form a valid clique cover"

        if verbose and is_valid:
            print(f"✓ Valid clique cover with {len(assignment)} cliques")
        elif verbose and not is_valid:
            print(f"✗ Invalid clique cover: {validated_result['validation']['error']}")

    except Exception as e:
        validated_result['validation']['error'] = f"Validation failed: {str(e)}"
        if verbose:
            print(f"✗ Validation error: {str(e)}")

    return validated_result


def chalupa_wrapper(txt_filepath: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Upper bound for θ(G) via Chalupa heuristic.
    FIXED: Now correctly applies Chalupa's clique covering to G directly,
    not to the complement.
    """
    try:
        if verbose:
            print(f"Running Chalupa on: {txt_filepath}")

        t0 = time.time()
        G = txt_to_networkx(txt_filepath)
        G = _compact_int_labels(G)

        if ChalupaHeuristic is None:
            if verbose:
                print("✗ ChalupaHeuristic not available")
            return {'theta': None, 'time': None, 'status': 'failed'}

        chalupa = ChalupaHeuristic(G)

        if hasattr(chalupa, 'iterated_greedy_clique_covering'):
            covering = chalupa.iterated_greedy_clique_covering()
            theta = len(covering) if covering is not None else None
        elif hasattr(chalupa, 'run'):
            res = chalupa.run()
            theta = res.get('upper_bound') if isinstance(res, dict) else int(res)
        else:
            theta = None

        elapsed = time.time() - t0

        return {
            'theta': theta,
            'time': elapsed,
            'status': 'heuristic' if theta else 'failed',
            'gap': None,  # Heuristic has no gap
            'kernel_nodes': G.number_of_nodes(),
            'kernel_edges': G.number_of_edges(),
            'rounds': None,
            'k_start': None,
            'k_final': None
        }

    except Exception as e:
        if verbose:
            print(f"✗ Chalupa failed on {txt_filepath}: {e}")
        return {'theta': None, 'time': None, 'status': 'failed'}


def _chalupa_warmstart(G: nx.Graph) -> Optional[Dict[Any, int]]:

   # Generate warmstart for ILP from Chalupa's clique covering of G.
   # FIXED: Now correctly works on G directly.

   # Returns: Dictionary mapping node -> clique_id

    if ChalupaHeuristic is None:
        return None
    try:
        # Work directly on G for clique covering
        heuristic = ChalupaHeuristic(G)
        cover = heuristic.iterated_greedy_clique_covering()

        if cover is None:
            return None

        # Create node -> clique_id mapping
        col = {}
        for c_idx, clique in enumerate(cover):
            for v in clique:
                col[v] = c_idx
        return col
    except Exception:
        return None


def _chalupa_warmstart_for_coloring(Gc: nx.Graph) -> Optional[Dict[Any, int]]:
    """
    Generate warmstart for graph coloring on Gc.
    This finds independent sets in Gc (which correspond to cliques in G).

    Args:
        Gc: The graph to color (typically complement of original)
    Returns: Dictionary mapping node -> color
    """
    if ChalupaHeuristic is None:
        return None
    try:
        # For graph coloring, we need to find independent sets
        # Chalupa's algorithm when applied to Gc will find cliques in Gc
        # which are independent sets in G
        heuristic = ChalupaHeuristic(Gc)

        # This will find cliques in Gc (= independent sets in G)
        cover = heuristic.iterated_greedy_clique_covering()

        if cover is None:
            return None

        col = {}
        for c_idx, indep_set in enumerate(cover):
            for v in indep_set:
                col[v] = c_idx
        return col
    except Exception:
        return None

""" oder so: 
def _chalupa_warmstart_for_coloring(Gc: nx.Graph) -> Optional[Dict[Any, int]]:
    
    #Warmstart für das Coloring auf Gc: benutze eine Clique-Cover auf G = complement(Gc),
    #denn Cliquen in G entsprechen Unabhängigkeitsmengen in Gc (Farben).
    
    if ChalupaHeuristic is None:
        return None
    try:
        G = nx.complement(Gc)
        heuristic = ChalupaHeuristic(G)
        cover = heuristic.iterated_greedy_clique_covering()
        if cover is None:
            return None
        col = {}
        for c_idx, clique in enumerate(cover):
            for v in clique:
                col[v] = c_idx  # diese "Cliquen in G" sind Farben (= IS) in Gc
        return col
    except Exception:
        return None
"""

def ilp_wrapper(txt_filepath: str, use_warmstart: bool = False, validate: bool = True,
                verbose: bool = False, **kwargs):
    """
    Standard ILP wrapper with validation.
    Computes θ(G) by solving the clique covering problem on G.

    Note: The ILP solver internally handles the conversion to coloring
    problem on the complement if needed.
    """
    if verbose:
        print(f"Running ILP on: {txt_filepath}")

    G = txt_to_networkx(txt_filepath)
    G_original = G.copy()
    G = _compact_int_labels(G)

    # Track warmstart info
    warm = _chalupa_warmstart(G) if use_warmstart else None
    start_objective = None

    if warm:
        start_objective = len(set(warm.values()))  # Number of colors in warmstart
        if verbose:
            print(f"Warmstart generated with {start_objective} colors")

    t0 = time.time()
    res = solve_ilp_clique_cover(G, is_already_complement=False, warmstart=warm, **kwargs)
    elapsed = time.time() - t0

    # Normalize result
    if not isinstance(res, dict):
        res = {'theta': res}

    # Add all expected metadata
    res['time'] = res.get('time', elapsed)
    res['status'] = res.get('status', 'optimal')
    res['gap'] = res.get('gap', 0.0)

    # Warmstart specific
    if use_warmstart:
        res['used_warmstart'] = warm is not None
        res['start_objective'] = start_objective
        if start_objective and res.get('theta'):
            res['improvement_over_start'] = start_objective - res['theta']

    # These are not applicable for standard ILP but needed for CSV
    res['kernel_nodes'] = G.number_of_nodes()
    res['kernel_edges'] = G.number_of_edges()
    res['rounds'] = None  # No rounds in standard ILP
    res['k_start'] = None
    res['k_final'] = None

    if validate and isinstance(res, dict):
        res = _validate_result(G_original, res, verbose=verbose)

    return res


def reduced_ilp_wrapper(txt_filepath: str, use_warmstart: bool = False, validate: bool = True,
                        verbose: bool = False, **kwargs) -> Dict[str, Any]:
    """
    Apply reductions on the complement graph Ḡ, then solve χ(Ḡ_red) via ILP
    to obtain θ(G) (since θ(G) = χ(Ḡ)).

    Key insight: Reductions are more effective on the complement for many graphs.
    """
    if not os.path.exists(txt_filepath):
        raise FileNotFoundError(f'Input file not found: {txt_filepath}')

    if verbose:
        print(f"Running Reduced ILP on: {txt_filepath}")

    t0_total = time.time()

    G = txt_to_networkx(txt_filepath)
    G_original = G.copy()
    G = _compact_int_labels(G)

    # Initial sizes
    original_nodes = G.number_of_nodes()
    original_edges = G.number_of_edges()

    # Build complement for reductions
    Gc = nx.complement(G)
    Gc = _compact_int_labels(Gc)

    if verbose:
        print(f"Original: {original_nodes} nodes, {original_edges} edges")
        print(f"Complement: {Gc.number_of_nodes()} nodes, {Gc.number_of_edges()} edges")

    # Apply reductions
    Gc_red, _meta, VCC_addition = apply_all_reductions(Gc, verbose=verbose, timing=verbose)
    Gc_red = _compact_int_labels(Gc_red)

    reduced_nodes = Gc_red.number_of_nodes()
    reduced_edges = Gc_red.number_of_edges()

    if verbose:
        print(f"After reductions: {reduced_nodes} nodes, {reduced_edges} edges")

    # Warmstart
    warm = _chalupa_warmstart_for_coloring(Gc_red) if use_warmstart else None
    start_objective = len(set(warm.values())) if warm else None

    # Solve
    t0_solve = time.time()
    res = solve_ilp_clique_cover(Gc_red, is_already_complement=True, warmstart=warm, **kwargs)
    solve_time = time.time() - t0_solve
    total_time = time.time() - t0_total

    # Normalize result
    if not isinstance(res, dict):
        res = {'theta': res}

    # Complete metadata
    res['time'] = total_time
    res['status'] = res.get('status', 'optimal')
    res['gap'] = res.get('gap', 0.0)
    res['kernel_nodes'] = reduced_nodes
    res['kernel_edges'] = reduced_edges

    # Track reduction info as pseudo-rounds
    res['rounds'] = 1  # Single reduction pass
    res['k_start'] = original_nodes  # Use nodes as proxy for k
    res['k_final'] = res.get('theta')
    res['kernel_nodes_rounds'] = [original_nodes, reduced_nodes]
    res['kernel_edges_rounds'] = [original_edges, reduced_edges]

    if use_warmstart:
        res['used_warmstart'] = warm is not None
        res['start_objective'] = start_objective
        if start_objective and res.get('theta'):
            res['improvement_over_start'] = start_objective - res['theta']

    res['round_logs'] = [{
        'round': 1,
        'reduction_removed_nodes': original_nodes - reduced_nodes,
        'kernel_nodes': reduced_nodes,
        'kernel_edges': reduced_edges
    }]

    return res

def interactive_reduced_ilp_wrapper(
        txt_filepath: str,
        use_warmstart: bool = False,
        max_rounds: int = 10,
        validate: bool = True,
        verbose: bool = False,
        **kwargs
) -> Dict[str, Any]:
    """
    Interactive reduction: iteratively improve an UB on χ(Gc) and apply reductions
    on the complement graph Gc, then solve the final kernel exactly via ILP.

    Returns a dict with θ(G)=χ(Gc) plus rich metadata expected by evaluation scripts.
    """
    if not os.path.exists(txt_filepath):
        raise FileNotFoundError(f'Input file not found: {txt_filepath}')

    if verbose:
        print(f"Running Interactive Reduced ILP on: {txt_filepath}")

    t0_total = time.time()

    # Load and prepare
    G = txt_to_networkx(txt_filepath)
    G_original = G.copy()
    G = _compact_int_labels(G)

    # Work on complement for coloring
    Gc_curr = nx.complement(G)
    Gc_curr = _compact_int_labels(Gc_curr)

    # --- Bookkeeping for metadata / plots ---
    rounds = 0
    prev_ub = float('inf')
    k_rounds: List[Optional[int]] = []           # UB per round (acts as "k" surrogate)
    kn_rounds: List[int] = [Gc_curr.number_of_nodes()]  # kernel nodes after each stage (start included)
    ke_rounds: List[int] = [Gc_curr.number_of_edges()]  # kernel edges after each stage (start included)
    round_logs: List[Dict[str, Any]] = []

    if verbose:
        print(f"Starting interactive reduction (max {max_rounds} rounds)")
        print(f"Initial complement size: |V|={Gc_curr.number_of_nodes()}, |E|={Gc_curr.number_of_edges()}")

    while rounds < max_rounds:
        # --- Heuristic UB on the *current* problem instance ---
        if ChalupaHeuristic is None:
            if verbose:
                print("✗ ChalupaHeuristic not available → stop interactive loop")
            break


        # UB auf χ(Gc_curr) via θ(G) mit G = complement(Gc_curr)
        # Für die Schleife brauchsts Upper Bound auf χ(Gc). Der kommt hier aus einer Clique-Cover auf G = complement(Gc) (weil θ(G) = χ(Gc))
        G_curr = nx.complement(Gc_curr)
        chal = ChalupaHeuristic(G_curr)
        if hasattr(chal, 'iterated_greedy_clique_covering'):
            cover = chal.iterated_greedy_clique_covering()
            ub = len(cover) if cover is not None else float('inf')
        elif hasattr(chal, 'run'):
            _r = chal.run()
            # falls run() ein Dict mit 'upper_bound' liefert (θ(G)):
            ub = _r.get('upper_bound', float('inf')) if isinstance(_r, dict) else float('inf')
        else:
            ub = float('inf')

        # Using Chalupa on Gc_curr (as in your current setup)
        """
        chal = ChalupaHeuristic(Gc_curr)
        if hasattr(chal, 'iterated_greedy_clique_covering'):
            cover = chal.iterated_greedy_clique_covering()
            ub = len(cover) if cover is not None else float('inf')
        elif hasattr(chal, 'run'):
            _r = chal.run()
            ub = _r.get('upper_bound', float('inf')) if isinstance(_r, dict) else float('inf')
        else:
            ub = float('inf')
        """
        k_rounds.append(None if (ub == float('inf')) else int(ub))

        if verbose:
            print(f"Round {rounds + 1}: UB={ub} (prev={prev_ub})")

        # Stop if no improvement
        if ub >= prev_ub:
            if verbose:
                print("✓ No UB improvement → stop")
            break
        prev_ub = ub

        # --- Apply reductions on complement ---
        nodes_before = Gc_curr.number_of_nodes()
        edges_before = Gc_curr.number_of_edges()

        Gc_next, _meta, VCC_addition = apply_all_reductions(Gc_curr, verbose=False, timing=False)
        Gc_next = _compact_int_labels(Gc_next)

        nodes_after = Gc_next.number_of_nodes()
        edges_after = Gc_next.number_of_edges()

        round_logs.append({
            'round': rounds + 1,
            'ub': None if (ub == float('inf')) else int(ub),
            'kernel_nodes_before': nodes_before,
            'kernel_edges_before': edges_before,
            'kernel_nodes': nodes_after,
            'kernel_edges': edges_after,
            'reduction_removed_nodes': nodes_before - nodes_after
        })

        if verbose:
            print(f"   Reduction: |V| {nodes_before} → {nodes_after}  |E| {edges_before} → {edges_after}")

        # If no structural change, stop
        if nodes_after == nodes_before and edges_after == edges_before:
            if verbose:
                print("   No further reductions possible → stop")
            Gc_curr = Gc_next  # keep it anyway
            kn_rounds.append(nodes_after)
            ke_rounds.append(edges_after)
            rounds += 1
            break

        # advance
        Gc_curr = Gc_next
        kn_rounds.append(nodes_after)
        ke_rounds.append(edges_after)
        rounds += 1

    if verbose:
        print(f"Finished after {rounds} round(s). Final kernel: |V|={Gc_curr.number_of_nodes()}, |E|={Gc_curr.number_of_edges()}")

    # --- Warmstart on the final kernel (optional) ---
    warm = _chalupa_warmstart_for_coloring(Gc_curr) if use_warmstart else None
    if verbose and use_warmstart:
        print(f"✓ Warmstart {'generated' if warm else 'not available'}")

    # --- Exact solve on the reduced complement instance ---
    t0_solve = time.time()
    res = solve_ilp_clique_cover(Gc_curr, is_already_complement=True, warmstart=warm, **kwargs)
    t_solve = time.time() - t0_solve

    # --- Normalize / enrich result dict ---
    res = dict(res) if isinstance(res, dict) else {'theta': res}
    res.setdefault('status', 'optimal')
    # ensure a total time is present (solver may already set 'time')
    res.setdefault('time', t_solve + (time.time() - t0_total - t_solve))

    # Validation note (true assignment check would need back-mapping)
    if validate and isinstance(res, dict):
        res['validation'] = {
            'note': 'Interactive reduction on complement graph; validation on reduced kernels only.',
            'final_reduced_nodes': Gc_curr.number_of_nodes(),
            'original_nodes': G_original.number_of_nodes(),
            'rounds': rounds,
            'reduction_efficiency': f"{G_original.number_of_nodes() - Gc_curr.number_of_nodes()} nodes removed in {rounds} rounds"
        }

    # --- Metadata expected by evaluation scripts ---
    res['rounds'] = rounds
    res['k_start'] = (k_rounds[0] if k_rounds and k_rounds[0] is not None else None)
    res['k_final'] = (k_rounds[-1] if k_rounds and k_rounds[-1] is not None else None)
    res['k_rounds'] = (k_rounds if k_rounds else None)

    res['kernel_nodes'] = Gc_curr.number_of_nodes()
    res['kernel_edges'] = Gc_curr.number_of_edges()
    res['kernel_nodes_rounds'] = (kn_rounds if kn_rounds else None)
    res['kernel_edges_rounds'] = (ke_rounds if ke_rounds else None)
    res['round_logs'] = round_logs if round_logs else None

    return res




def batch_ilp(file_list: List[str], use_warmstart: bool = False, validate: bool = True,
              verbose: bool = False, **kwargs) -> List[Dict[str, Any]]:
    """Batch processing of multiple graph files with validation."""
    results = []
    for i, path in enumerate(file_list):
        if verbose:
            print(f"\n--- Processing {i + 1}/{len(file_list)}: {path} ---")
        try:
            res = ilp_wrapper(path, use_warmstart=use_warmstart, validate=validate,
                              verbose=verbose, **kwargs)
            results.append({'file': path, **res})
            if verbose and isinstance(res, dict) and 'validation' in res:
                valid = res['validation'].get('is_valid', False)
                print(f"{'✓' if valid else '✗'} Result: θ(G) = {res.get('theta', 'N/A')}")
        except Exception as e:
            results.append({'file': path, 'error': str(e)})
            if verbose:
                print(f"✗ Error: {str(e)}")
    return results


def debug_clique_cover(txt_filepath: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Debug function that runs multiple approaches and compares results.
    Helps identify issues with different methods.
    """
    print(f"\n=== DEBUGGING CLIQUE COVER for {txt_filepath} ===")

    results = {}

    # 1. Chalupa Upper Bound
    print("\n1. Chalupa Upper Bound (Clique Covering):")
    chalupa_ub = chalupa_wrapper(txt_filepath, verbose=verbose)
    results['chalupa_ub'] = chalupa_ub

    # 2. Standard ILP
    print("\n2. Standard ILP:")
    ilp_res = ilp_wrapper(txt_filepath, use_warmstart=False, validate=True, verbose=verbose)
    results['ilp_standard'] = ilp_res

    # 3. ILP with Warmstart
    print("\n3. ILP with Warmstart:")
    ilp_warm_res = ilp_wrapper(txt_filepath, use_warmstart=True, validate=True, verbose=verbose)
    results['ilp_warmstart'] = ilp_warm_res

    # 4. Reduced ILP
    print("\n4. Reduced ILP:")
    red_ilp_res = reduced_ilp_wrapper(txt_filepath, use_warmstart=False, validate=True, verbose=verbose)
    results['reduced_ilp'] = red_ilp_res

    # Summary
    print(f"\n=== SUMMARY for {txt_filepath} ===")
    if chalupa_ub:
        print(f"Chalupa UB: {chalupa_ub}")

    for method, res in results.items():
        if isinstance(res, dict) and 'theta' in res:
            theta = res['theta']
            valid = res.get('validation', {}).get('is_valid', 'N/A')
            time_taken = res.get('time', 'N/A')
            print(f"{method}: θ(G) = {theta}, valid = {valid}, time = {time_taken}s")

    # Sanity check: Chalupa should always be >= ILP
    if chalupa_ub and isinstance(results.get('ilp_standard'), dict):
        ilp_theta = results['ilp_standard'].get('theta')
        if ilp_theta and chalupa_ub < ilp_theta:
            print("Chalupa should always provide an upper bound (≥ optimal)")

    return results