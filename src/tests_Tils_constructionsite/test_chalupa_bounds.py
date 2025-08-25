from pathlib import Path
from utils import get_value, txt_to_networkx
from algorithms.chalupa import ChalupaHeuristic

DATA = Path("test_graphs/curated")

def test_chalupa_bounds():
    for txt in sorted(DATA.glob("*.txt")):
        gold = int(get_value(txt, "Clique Cover Number θ(G)"))
        G = txt_to_networkx(txt)
        ch = ChalupaHeuristic(G)  # oder auf Ḡ, je nach Implementierung in deinem Repo
        bounds = ch.run()
        assert bounds["lower_bound"] <= gold <= bounds["upper_bound"]
