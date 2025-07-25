from wrappers import ilp_wrapper
from pathlib import Path
import time
import sys

graph_dir = "test_cases/generated/perturbed"
# Can also specify a different path if you like
if len(sys.argv) > 1:
    graph_dir = sys.argv[1]

for file in Path(graph_dir).iterdir():
    print(f"Currently processing file {file.name}")
    with open(file, 'a') as f:
        start_time = time.time()
        predicted = ilp_wrapper(file)
        end_time = time.time()

        f.write("\n")
        f.write(f"# Calculated by ILP\n")
        f.write(f"ILP Time to solve: {end_time - start_time}\n")
        f.write(f"Chromatic Number: {predicted}")
