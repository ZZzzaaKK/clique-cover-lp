from pathlib import Path
from utils import get_value, txt_to_networkx
from algorithms.ilp_solver import solve_ilp_clique_cover

DATA = Path("test_graphs/curated")  # oder dein Zielordner

def test_ilp_matches_gold():
    for txt in sorted(DATA.glob("*.txt")):
        gold = int(get_value(txt, "Clique Cover Number θ(G)"))
        G = txt_to_networkx(txt)
        res = solve_ilp_clique_cover(G, verbose=False, mip_gap=0.0)
        assert res["theta"] == gold
        assert res["chi_complement"] == gold
