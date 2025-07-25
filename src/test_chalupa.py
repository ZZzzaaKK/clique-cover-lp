from test_curated import TestRunner, save_summary
from wrappers import chalupa_wrapper
import sys

graph_dir = "test_cases/generated/perturbed"
# Can also specify a different path if you like
if len(sys.argv) > 1:
    graph_dir = sys.argv[1]

runner = TestRunner(graph_dir)

print("Testing Chalupa Algorithm:")
chalupa_results = runner.run_tests(chalupa_wrapper, "Chromatic Number")
save_summary(chalupa_results, "chalupa_on_ilp_tests")
