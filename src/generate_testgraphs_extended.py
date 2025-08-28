"""
Graph Generator für etwas komplexere Graphen
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import networkx as nx

# ---------------------------
# Globale Parameter (anpassbar)
# ---------------------------
OUT_DIR = Path("test_graphs/generated/perturbed_bigger_graphs")

# Ziel-Komplexität ~ halb so groß wie zuvor
TARGET_N_RANGE = (5,80)      # gewünschter Knotenbereich
MAX_EDGES = 100             # harte Obergrenze für |E|
SEED = 33
RNG = np.random.default_rng(SEED)

# Perturbations wie in der Vorlage (mit absichtlichem Tippfehler 'pertubation' im Dateinamen)
PERTURBATION_LEVELS = [0, 10, 20, 30, 40, 60]   # in Prozent
EDGE_ADD_RATIO = 0.25                       # add_prob = remove_prob * EDGE_ADD_RATIO

# ---------------------------
# Hilfsfunktionen
# ---------------------------

def _ensure_simple_graph(G: nx.Graph) -> nx.Graph:
    """Sorgt dafür, dass der Graph einfache Kanten hat (ohnehin Standard bei nx.Graph)."""
    # NetworkX Graph ist bereits einfach, aber wir stellen sicher, dass self-loops weg sind
    G.remove_edges_from(nx.selfloop_edges(G))
    return G

def _limit_edges_uniform(G: nx.Graph, max_edges: int) -> nx.Graph:
    """Reduziert die Anzahl Kanten auf max_edges, indem zufällig Kanten entfernt werden."""
    m = G.number_of_edges()
    if m <= max_edges:
        return G
    edges = list(G.edges())
    RNG.shuffle(edges)
    keep = set(edges[:max_edges])
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    H.add_edges_from(keep)
    return H

def _apply_perturbation(G: nx.Graph, remove_prob: float, add_prob: float) -> nx.Graph:
    """
    Entfernt jede existierende Kante mit remove_prob und fügt Nicht-Kanten mit add_prob hinzu.
    Begrenzung via MAX_EDGES erfolgt danach.
    """
    H = nx.Graph()
    H.add_nodes_from(G.nodes())

    # Entfernen
    for u, v in G.edges():
        if RNG.random() >= remove_prob:
            H.add_edge(u, v)

    # Hinzufügen (sparsam – wir sampeln nur einen kleinen Kandidatenpool)
    n = H.number_of_nodes()
    if add_prob > 0 and n > 1:
        # Kandidaten-Anzahl begrenzen (linear in n) um Aufwand klein zu halten
        num_candidates = min(5 * n, (n * (n - 1)) // 2)  # heuristisch
        candidates = set()
        while len(candidates) < num_candidates:
            u = int(RNG.integers(0, n))
            v = int(RNG.integers(0, n))
            if u == v:
                continue
            a, b = (u, v) if u < v else (v, u)
            if not H.has_edge(a, b):
                candidates.add((a, b))

        for (u, v) in candidates:
            if RNG.random() < add_prob:
                H.add_edge(u, v)

    H = _ensure_simple_graph(H)
    H = _limit_edges_uniform(H, MAX_EDGES)
    return H

def _write_graph_txt(G: nx.Graph, filename: Path):
    """Schreibt den Graphen im gewünschten .txt-Format."""
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w") as f:
        vertices = sorted(G.nodes())
        deg = dict(G.degree())
        for v in vertices:
            nbrs = sorted(G.neighbors(v))
            line = f"{v}: {' '.join(map(str, nbrs))}" if nbrs else f"{v}:"
            f.write(line + "\n")

        f.write("\n")
        # Attribute
        connected = "Yes" if nx.is_connected(G) else "No"
        n = G.number_of_nodes()
        m = G.number_of_edges()
        avg_deg = 2 * m / n if n > 0 else 0.0
        density = nx.density(G) if n > 1 else 0.0
        components = nx.number_connected_components(G)

        f.write(f"Connected: {connected}\n")
        f.write(f"Number of Vertices: {n}\n")
        f.write(f"Number of Edges: {m}\n")
        f.write(f"Average Degree: {avg_deg:.3f}\n")
        f.write(f"Density: {density:.3f}\n")
        f.write(f"Number of Components: {components}\n")
        f.write(f"Maximum Degree: {max(deg.values()) if deg else 0}\n")
        f.write(f"Minimum Degree: {min(deg.values()) if deg else 0}\n")

# ---------------------------
# Generatortypen
# ---------------------------

def generate_uniform_clique_blocks(total_nodes: int, num_cliques: int) -> Tuple[nx.Graph, Dict[int, int]]:
    """
    Erzeugt disjunkte Cliquen ähnlicher Größe. Nodes sind 0..total_nodes-1.
    """
    G = nx.Graph()
    G.add_nodes_from(range(total_nodes))
    sizes = np.full(num_cliques, total_nodes // num_cliques, dtype=int)
    sizes[: (total_nodes % num_cliques)] += 1

    node = 0
    community = {}
    for c, s in enumerate(sizes):
        block = list(range(node, node + s))
        node += s
        # Clique-Kanten
        for i in range(len(block)):
            u = block[i]
            community[u] = c
            for j in range(i + 1, len(block)):
                v = block[j]
                G.add_edge(u, v)
    return _ensure_simple_graph(G), community

def generate_skewed_clique_blocks(total_nodes: int, num_cliques: int,
                                  min_size: int, max_size: int) -> Tuple[nx.Graph, Dict[int, int]]:
    """
    Erzeugt disjunkte Cliquen variabler Größe in [min_size, max_size] bis total_nodes erreicht ist.
    """
    G = nx.Graph()
    G.add_nodes_from(range(total_nodes))
    community = {}

    sizes: List[int] = []
    remaining = total_nodes
    for _ in range(num_cliques - 1):
        s = int(RNG.integers(min_size, max_size + 1))
        s = max(1, min(s, remaining))
        sizes.append(s)
        remaining -= s
        if remaining <= 0:
            break
    if remaining > 0:
        sizes.append(remaining)

    node = 0
    for c, s in enumerate(sizes):
        block = list(range(node, node + s))
        node += s
        for i in range(len(block)):
            u = block[i]
            community[u] = c
            for j in range(i + 1, len(block)):
                v = block[j]
                G.add_edge(u, v)
    return _ensure_simple_graph(G), community

def generate_sbm_like(total_nodes: int, blocks: int, p_in: float, p_out: float) -> Tuple[nx.Graph, Dict[int, int]]:
    """
    Einfache Blockstruktur (SBM-artig) mit gleich großen Blöcken.
    """
    sizes = [total_nodes // blocks] * blocks
    sizes[0] += total_nodes - sum(sizes)  # Rest in Block 0

    G = nx.Graph()
    community = {}
    start = 0
    block_nodes: List[List[int]] = []
    for b, s in enumerate(sizes):
        nodes = list(range(start, start + s))
        start += s
        block_nodes.append(nodes)
        for u in nodes:
            community[u] = b

    # Intra
    for nodes in block_nodes:
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if RNG.random() < p_in:
                    G.add_edge(nodes[i], nodes[j])

    # Inter
    for a in range(blocks):
        for b in range(a + 1, blocks):
            for u in block_nodes[a]:
                for v in block_nodes[b]:
                    if RNG.random() < p_out:
                        G.add_edge(u, v)

    G.add_nodes_from(range(total_nodes))
    return _limit_edges_uniform(_ensure_simple_graph(G), MAX_EDGES), community

# ---------------------------
# Hauptsuite
# ---------------------------

def generate_suite():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- 1) Uniforme Clique-Blöcke ----------
    # Kleinere, gut lösbare Größen
    uniform_setups = [
        # (num_cliques, total_nodes)
        (8, RNG.integers(TARGET_N_RANGE[0], TARGET_N_RANGE[1] + 1)),
        (10, RNG.integers(TARGET_N_RANGE[0], TARGET_N_RANGE[1] + 1)),
    ]

    for k, n in uniform_setups:
        G0, _ = generate_uniform_clique_blocks(int(n), int(k))
        for p in PERTURBATION_LEVELS:
            rem = p / 100.0
            add = rem * EDGE_ADD_RATIO
            Gp = _apply_perturbation(G0, remove_prob=rem, add_prob=add)
            # Benennung analog Vorlage (mit 'pertubation')
            name = f"uniform_n{k}_s{int(n//k)}_pertubation{p:02d}.txt"
            _write_graph_txt(Gp, OUT_DIR / name)

    # ---------- 2) Skewed Clique-Blöcke ----------
    skewed_setups = [
        # (num_cliques, min_size, max_size, total_nodes)
        (5, 8, 25, int(RNG.integers(TARGET_N_RANGE[0], TARGET_N_RANGE[1] + 1))),
        (6, 5, 20, int(RNG.integers(TARGET_N_RANGE[0], TARGET_N_RANGE[1] + 1))),
    ]

    for k, mn, mx, n in skewed_setups:
        G0, _ = generate_skewed_clique_blocks(n, k, mn, mx)
        for p in PERTURBATION_LEVELS:
            rem = p / 100.0
            add = rem * EDGE_ADD_RATIO
            Gp = _apply_perturbation(G0, remove_prob=rem, add_prob=add)
            # gewollt falsche Reihenfolge/Schreibweise nach deinem Beispiel:
            # skewed_cliques5_min8_max5_pertubation30.txt
            # (dein Beispiel hatte min>max — wir übernehmen nur das Schema & Tippfehler im Namen)
            name = f"skewed_cliques{k}_min{mn}_max{mx}_pertubation{p:02d}.txt"
            _write_graph_txt(Gp, OUT_DIR / name)

    # ---------- 3) SBM-artig (sehr simple) ----------
    sbm_setups = [
        # (blocks, p_in, p_out, total_nodes)
        (5, 0.15, 0.01, int(RNG.integers(TARGET_N_RANGE[0], TARGET_N_RANGE[1] + 1))),
        (6, 0.10, 0.02, int(RNG.integers(TARGET_N_RANGE[0], TARGET_N_RANGE[1] + 1))),
    ]
    for b, pin, pout, n in sbm_setups:
        G0, _ = generate_sbm_like(n, b, pin, pout)
        for p in PERTURBATION_LEVELS:
            rem = p / 100.0
            add = rem * EDGE_ADD_RATIO
            Gp = _apply_perturbation(G0, remove_prob=rem, add_prob=add)
            name = f"sbm_blocks{b}_pin{int(pin*100)}_pout{int(pout*100)}_n{n}_pertubation{p:02d}.txt"
            _write_graph_txt(Gp, OUT_DIR / name)

    # ---------- 4) Edge Cases ----------
    # (a) Sehr dünn: ER mit ~O(n) Kanten
    n = int(RNG.integers(TARGET_N_RANGE[0], TARGET_N_RANGE[1] + 1))
    p_edge = min(4.0 / n, 0.02)  # sehr dünn
    G_thin = nx.fast_gnp_random_graph(n, p_edge, seed=int(RNG.integers(1_000_000)))
    G_thin = _limit_edges_uniform(_ensure_simple_graph(G_thin), MAX_EDGES)
    _write_graph_txt(G_thin, OUT_DIR / f"edgecase_very_thin_n{n}_m{G_thin.number_of_edges()}.txt")

    # (b) Sehr dicht: nahe an vollständig, dann begrenzen
    n = int(RNG.integers(TARGET_N_RANGE[0], TARGET_N_RANGE[1] + 1))
    p_edge = 0.9  # sehr dicht
    G_dense = nx.fast_gnp_random_graph(n, p_edge, seed=int(RNG.integers(1_000_000)))
    G_dense = _limit_edges_uniform(_ensure_simple_graph(G_dense), MAX_EDGES)
    _write_graph_txt(G_dense, OUT_DIR / f"edgecase_very_dense_n{n}_m{G_dense.number_of_edges()}.txt")

    # (c) Ketten von kleinen Cliquen mit dünnen Brücken
    k = 12
    size = max(10, int((TARGET_N_RANGE[0] // k)))
    nodes = k * size
    G_chain = nx.Graph()
    community = {}
    start = 0
    for i in range(k):
        block = list(range(start, start + size))
        start += size
        for u in block:
            community[u] = i
        # mache die Clique
        for a in range(len(block)):
            for b in range(a + 1, len(block)):
                G_chain.add_edge(block[a], block[b])
        # Brücke zur nächsten Clique
        if i < k - 1:
            u = block[-1]
            v = start  # first node of next block
            G_chain.add_edge(u, v)
    if nodes < TARGET_N_RANGE[0]:
        # hänge ein paar isolierte Knoten an (bleiben leicht)
        extra = TARGET_N_RANGE[0] - nodes
        G_chain.add_nodes_from(range(nodes, nodes + extra))
    G_chain = _limit_edges_uniform(_ensure_simple_graph(G_chain), MAX_EDGES)
    _write_graph_txt(G_chain, OUT_DIR / f"edgecase_chain_of_cliques_k{k}_size{size}_n{G_chain.number_of_nodes()}_m{G_chain.number_of_edges()}.txt")

    print(f"Fertig. Dateien liegen in: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    generate_suite()
