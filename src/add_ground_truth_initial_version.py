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

