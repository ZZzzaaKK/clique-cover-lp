import pickle

path = "data/uniform_n3_s5_r10_perturbed.pkl"

with open(path, "rb") as f:
    G = pickle.load(f)
    V = list(G.nodes())
    E = list(G.edges())
    n = len(V)
    upper = None

    print(f"Graph: ${G}")
    print(f"Vertices: ${V}")
    print(f"Edges: ${E}")
