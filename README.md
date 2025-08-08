# Getting Started

Install [uv](https://github.com/astral-sh/uv). Then add all the dependencies and activate the virtual environment with `source .venv/bin/activate`.

# Test Cases

## Simulator

Run `python src/generate_test_graphs.py` to generate test cases of differing distributions and save them to `test_graphs/generated`.

## Testing

For convenience, you can run the script `run_tests.sh`. It first calculates chromatic numbers for all generated perturbed graphs via the `src/add_ground_truth.py` script, and then runs through the tests you specify in the script. Adding ground truths relies on the Gurobi solver, so you'll need a license for larger graphs. You can also deviate from default path by running with a positional argument like this: `run_tests.sh test_graphs/curated`.

## Curation

Curated test cases were found at [houseofgraphs.org](houseofgraphs.org). Each `.txt` file contains the graph structure as well as their invariants. Feel free to add more test cases (choose Invariant values as file format)!

# Current State of Progress

- [ ] WP0 Simulator
  - [x] Generate test cases for different distributions
  - [x] Introduce perturbations
  - [ ] Choose reasonable parameters for task completions
- [ ] WP1 Exact vs Heuristic
  - [x] Chalupa
  - [x] ILP
  - [ ] Compare algorithms and write down answers
- [ ] WP2 Kernelizations for vertex clique cover problem
- [ ] WP3 Kernelizations for cluster editing problem
- [ ] WP4 Comparison of vertex clique cover and cluster editing solutions
- [ ] Bonus
- [ ] WP5 Real Data

# Next Steps

- Test ILP individually (in test_curated.py)
- Use ILP solutions as ground truth for chalupa in generated examples (in test_generated.py)

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
