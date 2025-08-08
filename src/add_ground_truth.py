from wrappers import ilp_wrapper
from pathlib import Path
from utils import get_value
import sys

def add_ground_truth_if_missing(directory):
    """Add ground truth only to files that don't already have it"""
    path = Path(graph_dir)

    for txt_file in path.glob("**/*.txt"):
        # Check if ground truth already exists
        existing_ground_truth = get_value(txt_file, "Chromatic Number")

        if existing_ground_truth is None:
            print(f"Computing ground truth for {txt_file.name}...")
            chromatic_number = ilp_wrapper(str(txt_file))

            if chromatic_number is not None:
                # Append to file
                with open(txt_file, 'a') as f:
                    f.write("# Calculated by ILP\n")
                    f.write(f"Chromatic Number: {chromatic_number}\n")
                print(f"  Added: Chromatic Number: {chromatic_number}")
            else:
                print(f"  Failed to compute for {txt_file.name}")
        else:
            print(f"Ground truth already exists for {txt_file.name}: {existing_ground_truth}")

if __name__ == "__main__":
    graph_dir = sys.argv[1] if len(sys.argv) > 1 else "test_graphs/generated/perturbed"
    add_ground_truth_if_missing(graph_dir)


"""
'''
EDIT suggestion: 
Add ground truth (Clique Cover Number θ(G)) to graph instance files.

Why this matters:
- We solve Clique Cover by coloring the **complement graph** (θ(G) = χ(Ḡ)).
- Writing the line as **"Clique Cover Number θ(G): …"** avoids the previous
  ambiguity (which imo incorrectly suggested χ(G)).
- The note **"Calculated by ILP on complement graph"** documents the method,
  making downstream evaluation (WP1/WP2) unambiguous and reproducible.
'''
from pathlib import Path
import sys

from wrappers import ilp_wrapper
from utils import get_value

def add_ground_truth_if_missing(directory):
    '''Append exact Clique Cover Number θ(G) to each .txt graph file if missing.

    Notes
    -----
    - `ilp_wrapper` computes θ(G) by solving χ(Ḡ) (ILP on the complement graph).
    - We check for the **new label** to avoid duplicate or legacy entries.
    '''
    path = Path(directory)
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {directory}")

    for txt_file in path.glob("**/*.txt"):
        existing_theta = get_value(txt_file, "Clique Cover Number θ(G)")
        if existing_theta is not None:
            print(f"Ground truth already exists for {txt_file.name}: {existing_theta}")
            continue

        print(f"Computing ground truth for {txt_file.name}…")
        theta = ilp_wrapper(str(txt_file))
        if theta is None:
            print(f"  Failed to compute θ(G) for {txt_file.name}")
            continue

        with open(txt_file, 'a', encoding='utf-8') as f:
            f.write("# Calculated by ILP on complement graph\n")
            f.write(f"Clique Cover Number θ(G): {theta}\n")
        print(f"  Added: Clique Cover Number θ(G): {theta}")

if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "test_graphs/generated/perturbed"
    add_ground_truth_if_missing(target_dir)

"""