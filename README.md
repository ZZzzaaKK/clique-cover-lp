# Getting Started

Install [uv](https://github.com/astral-sh/uv). Then add all the dependencies by running `uv sync` and activate the virtual environment with `source .venv/bin/activate`.

# Test Cases

## Simulator

Run `python src/generate_test_graphs.py` to generate test cases of differing distributions and save them to `test_graphs/generated`. It's possible, but not necessary, to specify custom parameters for the graphs that should be generated here. They are documented in the main function of the script.

## Testing

For convenience, you can run the script `run_tests.sh`. It first calculates vertex clique cover numbers for all graphs in a path that can be specified (default: `test_graphs/generated/perturbed`) via the `src/add_vertex_clique_cover_number.py` script, or chromatic numbers via `src/add_chromatic_number.py` script if you pass the --chromatic-number flag. It then runs through the tests you specify as command line arguments, or all available tests if none were specified. Adding ground truths relies on the Gurobi solver, so you'll need a license for larger graphs. You can also deviate from default path by running with a positional argument like this: `run_tests.sh test_graphs/curated`.

Example Usage:
```
  ./run_tests.sh --ilp --reduced-ilp --chalupa --reduced-chalupa --cluster-editing --reduced-cluster-editing --chromatic-number test_graphs/generated/perturbed
```
This will run all the specified algorithms, which are made available through the `src/wrappers.py` script, on the specified test graph directory and save the results in `results/raw/`. These results are human-readable and already contain interesting information.

## Analysis

To compare results of different algorithms, you can run `python src/comparison.py <results-file1> <results-file2> ...` with two or more of these results files. This will output analysis plots in the `results/analyses` directory.

## Curation

Curated test cases were found at [houseofgraphs.org](houseofgraphs.org). Each `.txt` file contains the graph structure as well as their invariants. Feel free to add more test cases (choose Invariant values as file format)!

## File Structure

Files in `src/algorithms` pertain to algorithms, `chalupa.py` and `helpers.py` for the Chalupa heuristic, `cluster_editing.py` for the cluster editing problem, `ilp_solver.py` for the vertex clique cover (VCC) problem. The `src/reductions` directory contains code for reductions pertaining to the VCC problem. The cluster editing reductions are currently contained in the `src/algorithms/cluster_editing.py` file.

The files in the top-level `src/` directory relate to the workflow infrastructure. The files `src/test_cluster_editing.py` and `src/test_reductions.py` were used during development to verify the cluster editing and VCC reduction algorithms. `src/wp5_rfam_to_graph.py` is meant for converting an rfam file into a weighted graph, but is currently still a work-in-progress. `src/algorithms/helpers.py` contains some convenience functions for test infrastructure.



# Current State of Progress

- [x] WP0 Simulator
  - [x] Generate test cases for different distributions
  - [x] Introduce perturbations
  - [x] Choose reasonable parameters for task completions
- [ ] WP1 Exact vs Heuristic
  - [x] Chalupa
    - [x] How to actually use lower bound? -> Currently just output in results file
  - [x] ILP
  - [x] Compare algorithms
- [x] WP2 Kernelizations for vertex clique cover problem
  - [x] Implement reductions
  - [ ] Double-check for difference between reduced-ilp and interactive-reduced-ilp
- [ ] WP3 Kernelizations for cluster editing problem
- [ ] WP4 Comparison of vertex clique cover and cluster editing solutions
- [ ] Bonus
- [ ] WP5 Real Data

# Next Steps

- [ ] Check cluster editing implementation
  - [x] ILP
    - [x] What's the goal? -> produce modifications to create cluster graph with minimum cost
    - [x] What's the triangle inequality? -> way to check if vertices are connected in a clique
    - [ ] What are 2-partition inequalities?
  - [x] Reduction based on edge cuts -> produces consistent results, results in small, but sometimes noticeable speedup
  - [ ] Should output number of clusters obtained to compare with vertex clique cover number
- [x] Rethink Chalupa, where to use upper/lower bound
- [ ] Double-check interactive scheme
- [ ] Weighting in Real Data

# Work Program

## WP0

Write a simulator for test cases by starting from a disjoint union of
cliques. Then introduce perturbations by removing edges from within
the cliques and adding edges between them. Different distributions of
cliques sizes should be generated, from uniform (all the same size) to skew
distributions with few large and many small ones.

## WP1 Exact versus heuristic solutions:

(1.a) Implement Chalupa’s heuristic algorithm (see [4, 8]).

(1.b) Implement the ILP for vertex clique coloring and compare the quality
of solutions. What are practical limits on size and “perturbation
strength” for the ILP solution?

(1.c) Compare the Chalupa to the ILP solutions. How often does the
heuristic produce the exact solutions for larger instances. How well
performs the heuristic on average?

## WP2 Kernelizations for the vertex clique cover problem

(2.a) Implement the different reduction algorithms and test them thoroughly.

(2.b) Use the following workflow to get exact solutions: (i) Estimate an
upper bound k on the clique cover number θ(G). (ii) Run the kernelization/data reduction.
(iii) Run the exact solution.
Does that significantly increase the size/perturbation level of the instances that can be processed?

(2.c) Extend your workflow to an interactive scheme: After data reduction/kernelization run Chalupa again. If you now get a smaller estimate for k, rerun the pipeline. Does that improve running time or
problem size before feeding the kernel into the ILP?

## WP3 Kernelizations for the Cluster editing problem.

(3.a) Implement the kernelizations for cluster editing and test them thoroughly.

(3.b) Quantify the improvements achieved by the kernelizations on the
cluster editing problem for your instances.

## WP4 Comparison of Vertex Clique Cover and Cluster Editing solutions.

The two problems are conceptually similar. How good are the solutions of the
cluster editing compared to vertex clique cover. To this end compare the
number of clusters C(G) obtained by cluster editing with the vertex clique
cover number θ(G).

## Bonus

Can you think of a good heuristic to get a better solution for one
problem from an exact solution of the other?

## WP5 Real data.

Once everything is up and running, apply the workflows to
real data from shift-alignment predictions on Rfam RNA families. This
data will by supplied around May 1st by Maria Waldl.

## References

[1] N. Bansal, A. Blum, and S. Chawla. Correlation clustering. Mach. Learn.,
56:89–113, 2004.
[2] S. Böcker and J. Baumbach. Cluster editing. In P. Bonizzoni, V. Brattka, and B. Löwe, editors, The Nature of Computation. Logic, Algorithms,
Applications, pages 33–44. Springer, Berlin, Heidelberg, 2013.
[3] Y. Cao and J. Chen. Cluster editing: Kernelization based on edge cuts.
Algorithmica, 64:152–169, 2012.
[4] D. Chalupa. Construction of near-optimal vertex clique covering for real-
world networks. Computing and Informatics, 34(6):1397–1417, 2016.
[5] G. Fritzsch, M. Schlegel, and P. F. Stadler. Alignments of mitochondrial
genome arrangements: Applications to metazoan phylogeny. J. Theor. Biol.,
240:511–520, 2006.
[6] R. Shamir, R. Sharan, and D. Tsur. Cluster graph modification problems.
Discrete Applied Mathematics, 144:173–182, 2004.
[7] A. Stoltzfus, J. M. Logsdon Jr, J. D. Palmer, and W. F. Doolittle. Intron
“sliding” and the diversity of intron positions. Proc Natl Acad Sci U S A,
94:10739–10744, 1997.
[8] D. Strash and L. Thompson. Effective data reduction for the vertex clique
cover problem. In 2022 Proceedings of the Symposium on Algorithm Engineering and Experiments (ALENEX), pages 41–53, 2022.
[9] M. Waldl, S. Will, M. Wolfinger, I. L. Hofacker, and P. F. Stadler. Bi-
alignments as models of incongruent evolution of RNA sequence and secondary structure. In P. Cazzaniga, D. Besozzi, I. Merelli, and L. Manzoni, editors, Computational Intelligence Methods for Bioinformatics and
Biostatistics, 16th International Meeting, CIBB’19, volume 12313 of Lect.
Notes Comp. Sci., pages 159–170, Cham, CH, 2020. Springer Nature
