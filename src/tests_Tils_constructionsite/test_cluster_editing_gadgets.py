from pathlib import Path
from utils import get_value, txt_to_networkx
from algorithms.cluster_editing_solver import ClusterEditingSolver

DATA = Path("test_graphs/curated")

def _read_kopt(txt_path):
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Cluster Editing optimum k*"):
            return int(line.split(":")[1])
    return None

def test_gadgets_kopt():
    for txt in sorted(DATA.glob("*cluster_editing*txt")):
        kopt = _read_kopt(txt)
        if kopt is None:
            continue
        G = txt_to_networkx(txt)
        solver = ClusterEditingSolver()
        sol = solver.solve(G)  # je nach API
        assert sol["cost"] == kopt
