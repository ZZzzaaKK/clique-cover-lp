from wrappers import ilp_wrapper, reduced_ilp_wrapper
from pathlib import Path
from utils import get_value
import sys

def add_ground_truth_if_missing(directory):
    """Add chromatic number only to files that don't already have it"""
    path = Path(graph_dir)

    for txt_file in path.glob("**/*.txt"):
        # Check if chromatic number already exists
        existing_ground_truth = get_value(txt_file, "Chromatic Number")

        if existing_ground_truth is None:
            print(f"Computing chromatic number for {txt_file.name}...")
            chromatic_number = reduced_ilp_wrapper(str(txt_file), "chromatic_number", time_limit=300)[0]

            if chromatic_number is not None:
                # Append to file
                with open(txt_file, 'a') as f:
                    f.write("\n# Calculated by ILP\n")
                    f.write(f"Chromatic Number: {chromatic_number}\n")
                print(f"  Added: Chromatic Number: {chromatic_number}")
            else:
                print(f"  Failed to compute for {txt_file.name}")
        else:
            print(f"Chromatic Number already exists for {txt_file.name}: {existing_ground_truth}")

if __name__ == "__main__":
    graph_dir = sys.argv[1] if len(sys.argv) > 1 else "test_graphs/generated/perturbed"
    add_ground_truth_if_missing(graph_dir)
