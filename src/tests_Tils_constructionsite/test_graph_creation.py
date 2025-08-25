# This script createss a compact testpack of small graphs with *known* ground truths
# for the Clique Cover / Coloring (and simple Cluster Editing gadgets).
#
# Files will be written to: /mnt/data/cc_testpack_2025-08-20
# - Each graph is saved in the same adjacency-list .txt style that the project uses.
# - included: a ground-truth line: "Clique Cover Number θ(G): K"
#   (computed EXACTLY as χ(Ḡ) via backtracking coloring)
# - For some instances we also add "Cluster Editing optimum k*: <int>" when it is guaranteed.
# - A manifest CSV summarizes everything.
#
import os
import itertools
import math
import json
from pathlib import Path
import networkx as nx
import pandas as pd

OUT_DIR = Path("/mnt/data/cc_testpack_2025-08-20")
OUT_DIR.mkdir(exist_ok=True, parents=True)

def write_graph_txt(G: nx.Graph, path: Path, theta: int, extra_lines=None):
    """Write adjacency format + ground truth into a .txt file."""
    lines = []
    # Ensure integer labels 1..n for compatibility with your parser
    mapping = {old: i+1 for i, old in enumerate(sorted(G.nodes()))}
    Gm = nx.relabel_nodes(G, mapping, copy=True)
    for v in sorted(Gm.nodes()):
        neigh = sorted(set(Gm.neighbors(v)))
        if neigh:
            lines.append(f"{v}: " + " ".join(map(str, neigh)))
        else:
            lines.append(f"{v}:")
    lines.append(f"Clique Cover Number θ(G): {theta}")
    if extra_lines:
        lines.extend(extra_lines)
    path.write_text("\n".join(lines), encoding="utf-8")


def exact_chromatic_number(G: nx.Graph) -> int:
    """Exact χ(G) via simple backtracking (ok for n<=14~16 in our tiny cases)."""
    n = G.number_of_nodes()
    order = sorted(G.nodes(), key=lambda u: G.degree(u), reverse=True)  # heuristic: high-degree first
    best = n
    colors = {}

    def valid(u, c):
        return all((v not in colors or colors[v] != c) for v in G.neighbors(u))

    def dfs(i, used):
        nonlocal best
        if i == n:
            best = min(best, used)
            return
        u = order[i]
        # Try existing colors
        for c in range(used):
            if valid(u, c):
                colors[u] = c
                if used < best:
                    dfs(i+1, used)
                del colors[u]
        # Try new color if it won't exceed current best
        if used + 1 < best:
            colors[u] = used
            dfs(i+1, used+1)
            del colors[u]

    dfs(0, 0)
    return best


def theta_via_complement(G: nx.Graph) -> int:
    H = nx.complement(G)
    return exact_chromatic_number(H)


def disjoint_cliques(sizes):
    """Build disjoint cliques with sizes list."""
    G = nx.Graph()
    cur = 0
    for s in sizes:
        nodes = list(range(cur, cur + s))
        cur += s
        for i in range(s):
            for j in range(i+1, s):
                G.add_edge(nodes[i], nodes[j])
    return G


def planted_cluster_editing_gadget(clique_sizes, add_cross_edges=(), remove_intra_edges=()):
    """
    Start from disjoint cliques, then add cross edges and/or remove some intra edges.
    For these gadgets, the *optimal* cluster editing cost is guaranteed to be exactly:
        k* = len(add_cross_edges) + len(remove_intra_edges)
    because each flip is independent and doesn't interact with others.
    """
    G = disjoint_cliques(clique_sizes)
    # Map local indices to global ids
    cliques = []
    cur = 0
    for s in clique_sizes:
        cliques.append(list(range(cur, cur+s)))
        cur += s

    # Add cross-edges (between different cliques)
    for (ci, ui), (cj, vj) in add_cross_edges:
        u = cliques[ci][ui]
        v = cliques[cj][vj]
        if G.has_edge(u, v):
            raise ValueError("Cross-edge already present; pick a non-edge.")
        G.add_edge(u, v)

    # Remove intra-clique edges
    for (ci, ui, vj) in remove_intra_edges:
        u = cliques[ci][ui]
        v = cliques[ci][vj]
        if not G.has_edge(u, v):
            raise ValueError("Intra edge missing; pick an existing intra edge.")
        G.remove_edge(u, v)

    k_opt = len(add_cross_edges) + len(remove_intra_edges)
    return G, k_opt


rows = []

PACKS = []

# 1) Empty graphs & complete graphs (edge cases)
for n in range(1, 7):
    G_empty = nx.empty_graph(n)
    theta = theta_via_complement(G_empty)  # = n
    fname = OUT_DIR / f"A_empty_n{n}.txt"
    write_graph_txt(G_empty, fname, theta)
    rows.append(dict(name=fname.name, n=n, m=G_empty.number_of_edges(), theta=theta, family="empty"))

    G_complete = nx.complete_graph(n)
    theta = theta_via_complement(G_complete)  # = 1
    fname = OUT_DIR / f"B_complete_K{n}.txt"
    write_graph_txt(G_complete, fname, theta)
    rows.append(dict(name=fname.name, n=n, m=G_complete.number_of_edges(), theta=theta, family="complete"))

# 2) Disjoint cliques (planted structure)
for sizes in [(2,2), (3,3), (3,4), (2,3,2), (4,4,2)]:
    G = disjoint_cliques(sizes)
    theta = theta_via_complement(G)  # equals len(sizes)
    fname = OUT_DIR / f"C_disjoint_cliques_{'-'.join(map(str,sizes))}.txt"
    write_graph_txt(G, fname, theta, extra_lines=[f"Planted cliques: {list(sizes)}"])
    rows.append(dict(name=fname.name, n=G.number_of_nodes(), m=G.number_of_edges(),
                     theta=theta, family="disjoint_cliques", planted=list(sizes)))

# 3) Complement families with known χ(H)
# H bipartite => χ(H)=2 => θ(G)=2
for a,b in [(3,4), (5,6)]:
    H = nx.complete_graph(a+b)
    # build K_{a,b} as bipartite, then G = complement(H_bipartite)
    from networkx.algorithms.bipartite.generators import complete_bipartite_graph
    H = complete_bipartite_graph(a, b)
    G = nx.complement(H)
    theta = 2
    fname = OUT_DIR / f"D_comp_of_K{a},{b}_theta2.txt"
    write_graph_txt(G, fname, theta, extra_lines=[f"Constructed as complement of K_{{{a},{b}}} with χ=2"])
    rows.append(dict(name=fname.name, n=G.number_of_nodes(), m=G.number_of_edges(),
                     theta=theta, family="complement_of_bipartite", note="θ=2"))

# H = odd cycles: χ(H)=3 => θ(G)=3.
for k in [5,7,9]:
    H = nx.cycle_graph(k)
    G = nx.complement(H)
    theta = 3
    fname = OUT_DIR / f"E_comp_of_C{k}_theta3.txt"
    write_graph_txt(G, fname, theta, extra_lines=[f"Constructed as complement of C_{k} (odd) with χ=3"])
    rows.append(dict(name=fname.name, n=G.number_of_nodes(), m=G.number_of_edges(),
                     theta=theta, family="complement_of_odd_cycle", note="θ=3"))

# 4) Paths & small tricky shapes (computed exactly)
for n in [4,5,6,7]:
    G = nx.path_graph(n)
    theta = theta_via_complement(G)
    fname = OUT_DIR / f"F_path_P{n}.txt"
    write_graph_txt(G, fname, theta, extra_lines=[f"Computed θ via χ(Ḡ) exact"])
    rows.append(dict(name=fname.name, n=n, m=G.number_of_edges(), theta=theta, family="path"))

# 5) Twin patterns (for reduction testing)
# True twins: nodes connected to each other and sharing identical neighbors
# Build K3 with a true-twin duplicate of one node => still clique cover θ=1
G = nx.Graph()
G.add_edges_from([(1,2),(2,3),(1,3)])  # K3
G.add_node(4)
G.add_edges_from([(4,1),(4,2),(4,3)])  # node 4 connected to all others -> K4
theta = theta_via_complement(G)  # 1
fname = OUT_DIR / "G_true_twins_K4.txt"
write_graph_txt(G, fname, theta, extra_lines=["Contains true twins (useful for twin-reduction checks)"])
rows.append(dict(name=fname.name, n=G.number_of_nodes(), m=G.number_of_edges(), theta=theta, family="twins_true"))

# False twins: identical neighbor sets but no edge between them
# Build a square with two opposite nodes as false twins (both connect only to same two neighbors)
G = nx.Graph()
G.add_edges_from([(1,3),(2,3),(1,4),(2,4)])  # 1,2 both connect to 3 and 4, but 1-2 not connected
theta = theta_via_complement(G)
fname = OUT_DIR / "H_false_twins_square.txt"
write_graph_txt(G, fname, theta, extra_lines=["Contains false twins (useful for twin-reduction checks)"])
rows.append(dict(name=fname.name, n=G.number_of_nodes(), m=G.number_of_edges(), theta=theta, family="twins_false"))

# 6) Cluster Editing gadgets with guaranteed optimum k*
# Start from 3 cliques of sizes 4,4,3; add 3 cross-edges far apart; remove 2 intra edges
G, kopt = planted_cluster_editing_gadget(
    clique_sizes=[4,4,3],
    add_cross_edges=[( (0,0), (1,0) ), ( (0,1), (2,0) ), ( (1,2), (2,1) )],
    remove_intra_edges=[(0,2,3), (1,1,3)]
)
theta = theta_via_complement(G)
fname = OUT_DIR / "I_cluster_editing_gadget_k5.txt"
write_graph_txt(G, fname, theta, extra_lines=[f"Cluster Editing optimum k*: {kopt} (independent flips)"])
rows.append(dict(name=fname.name, n=G.number_of_nodes(), m=G.number_of_edges(), theta=theta,
                 family="cluster_editing_gadget", cluster_kopt=kopt))

# Pure cross-edge noise only
G, kopt = planted_cluster_editing_gadget(
    clique_sizes=[5,5],
    add_cross_edges=[((0,0),(1,0)), ((0,2),(1,3)), ((0,4),(1,1))],
    remove_intra_edges=[]
)
theta = theta_via_complement(G)
fname = OUT_DIR / "J_cluster_editing_cross_only_k3.txt"
write_graph_txt(G, fname, theta, extra_lines=[f"Cluster Editing optimum k*: {kopt} (delete those cross-edges)"])
rows.append(dict(name=fname.name, n=G.number_of_nodes(), m=G.number_of_edges(), theta=theta,
                 family="cluster_editing_gadget", cluster_kopt=kopt))

# Pure missing intra-edges only
G, kopt = planted_cluster_editing_gadget(
    clique_sizes=[6],
    add_cross_edges=[],
    remove_intra_edges=[(0,0,1),(0,2,3),(0,4,5)]
)
theta = theta_via_complement(G)
fname = OUT_DIR / "K_cluster_editing_missing_intra_k3.txt"
write_graph_txt(G, fname, theta, extra_lines=[f"Cluster Editing optimum k*: {kopt} (add those intra-edges)"])
rows.append(dict(name=fname.name, n=G.number_of_nodes(), m=G.number_of_edges(), theta=theta,
                 family="cluster_editing_gadget", cluster_kopt=kopt))

# 7) Mixed edge cases: isolated vertices added to a clique
G = nx.complete_graph(4)
G.add_nodes_from([4,5,6])  # isolated (labels will be remapped on write)
theta = theta_via_complement(G)
fname = OUT_DIR / "L_clique_plus_isolates.txt"
write_graph_txt(G, fname, theta, extra_lines=["Contains isolated vertices (should be handled safely)"])
rows.append(dict(name=fname.name, n=G.number_of_nodes(), m=G.number_of_edges(), theta=theta, family="mixed_isolates"))

# 8) Small random sprinkled graphs (n<=10) with exact θ
rng = nx.utils.create_random_state(42)
for idx, n in enumerate([7,8,9,10]):
    G = nx.gnp_random_graph(n, 0.25, seed=rng)
    theta = theta_via_complement(G)
    fname = OUT_DIR / f"M_random_small_{n}_{idx}.txt"
    write_graph_txt(G, fname, theta, extra_lines=["Exact θ via χ(Ḡ) backtracking"])
    rows.append(dict(name=fname.name, n=G.number_of_nodes(), m=G.number_of_edges(), theta=theta, family="random_small"))

# Manifest
df = pd.DataFrame(rows).sort_values(["family","n","m","name"])
manifest_path = OUT_DIR / "MANIFEST.csv"
df.to_csv(manifest_path, index=False)

# Also zip the pack for convenience
import zipfile
zip_path = "/mnt/data/cc_testpack_2025-08-20.zip"
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for p in OUT_DIR.glob("*.txt"):
        zf.write(p, arcname=p.name)
    zf.write(manifest_path, arcname=manifest_path.name)

import caas_jupyter_tools
caas_jupyter_tools.display_dataframe_to_user("Clique-Cover Testpack Manifest", df)

zip_path
