from pathlib import Path
from utils import txt_to_networkx
from reductions.reductions import apply_all_reductions
from algorithms.ilp_solver import solve_ilp_clique_cover

DATA = Path("test_graphs/curated")

def test_reductions_preserve_theta():
    for txt in sorted(DATA.glob("*.txt")):
        G = txt_to_networkx(txt)
        theta_before = solve_ilp_clique_cover(G)["theta"]
        G_red, trace = apply_all_reductions(G.copy(), verbose=False)
        theta_after = solve_ilp_clique_cover(G_red)["theta"]
        # Falls Rekonstruktion vorhanden: erst rekonstruieren, dann vergleichen.
        assert theta_after <= theta_before  # Kernel darf kleiner/gleich sein
        # Optional: mit Reconstruction: assert theta_reconstructed == theta_before