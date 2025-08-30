---
title: "Projekt-Dokumentation
---

## Überblick
Diese Dokumentation beschreibt Module, Interoperabilität und Ausführungsreihenfolge des Repos und ergänzt die mathematischen Grundlagen mit **nummerierten Gleichungen**, **Querverweisen**, sowie einem **Code‑Abgleich** der Böcker‑Kernelregeln und der Chalupa‑Heuristik.

## wichtigste Skripte → Zweck → API → WP (Überblick)

| Datei | Zweck | Wichtigste Klassen/Funktionen | WP |
|---|---|---|---|
| src/algorithms/chalupa.py | Chalupa-Heuristik (Vertex Clique Cover) | ChalupaHeuristic | WP1 |
| src/algorithms/ilp_solver.py | ILP für Clique Cover (χ(Ḡ)) | solve_ilp_clique_cover, solve_ilp_direct_on_complement | WP1 |
| src/algorithms/cluster_editing_kernelization.py | CE-Kernelization (Böcker R1–R5, Critical Cliques) | AdvancedClusterEditingInstance, AdvancedKernelization | WP3 |
| src/algorithms/cluster_editing_ilp.py | CE-ILP mit Schnitten | solve_cluster_editing_ilp, _find_2partition_cuts | WP3 |
| src/algorithms/cluster_editing_solver.py | CE-Orchestrierung (Kernel + ILP) | ClusterEditingSolver | WP3 |
| src/wrappers.py | Wrapper (Heuristik/ILP/Reduced/Interactive) | reduced_ilp_wrapper, interactive_reduced_ilp_wrapper, chalupa_wrapper, ilp_wrapper | WP1 |
| src/wp3_evaluation.py | WP3-Experimente (inkl. Complexity-Fit) | WP3EnhancedEvaluator | WP3 |
| src/WP4_comparison_VCC_CE.py | WP4: Vergleich VCC vs CE | ComparisonFramework | WP4 |
| src/generate_test_graphs.py | WP0: Testgraphen erzeugen | generate_test_suite | WP0 |

## Skripte


| Datei | Zweck | Wichtigste Klassen/Funktionen | WP |
|---|---|---|---|
| src/WP2BC.py | WP2b and WP2c Analysis from Existing Evaluation Results | WP2bcResultsAnalyzer | WP2 |
| src/WP4_comparison_VCC_CE.py | WP4: Comparison of Vertex Clique Cover and Cluster Editing Solutions | ClusteringResult, ComparisonResult, SolverAdapter, ComparisonFramework | WP4 |
| src/WP5_constructionsite.py | WP5: Real Data Analysis on Rfam RNA Families | RNAClusteringResult, WP5RfamAnalysis | WP5 |
| src/__init__.py | Hilfs-/Auswertungsskript | - | shared |
| src/add_ground_truth.py | EDIT suggestion: | _append_theta_line, add_ground_truth_if_missing, _cli | shared |
| src/add_ground_truth_initial_version.py | Hilfs-/Auswertungsskript | add_ground_truth_if_missing | shared |
| src/algorithms/__init__.py | Hilfs-/Auswertungsskript | - | shared |
| src/algorithms/chalupa.py | Implementation of Chalupa's heuristic algorithm for clique coloring. | ChalupaHeuristic | WP1 |
| src/algorithms/cluster_editing_ilp.py | Cluster Editing: Kernelization/ILP/Orchestrierung | solve_cluster_editing_ilp, _solve_with_cutting_planes, _find_2partition_cuts, validate_clustering, calculate_clustering_cost | WP3 |
| src/algorithms/cluster_editing_kernelization.py | Cluster Editing: Kernelization/ILP/Orchestrierung | ReductionRule, UnionFind, KernelizationCache, RuleEffectiveness | WP3 |
| src/algorithms/cluster_editing_solver.py | Cluster Editing: Kernelization/ILP/Orchestrierung | ClusterEditingSolver | WP3 |
| src/algorithms/helpers.py | Hilfs-/Auswertungsskript | random_permutation, uniformly_random, jump | shared |
| src/algorithms/ilp_solver.py | Integer Linear Programming (ILP) formulation for the vertex clique coloring problem. | _parse_warmstart, solve_ilp_clique_cover, solve_ilp_direct_on_complement | WP1 |
| src/comparison_chalupa_ilp_evaluation_all_wrappers.py | comparison_chalupa_ilp_evaluation_all_wrappers.py | WP1cEvaluator, ExtendedWP1cEvaluator | WP1 |
| src/generate_test_graphs.py | Generate and save test cases for clique covering experiments. | visualize_graph, visualize_solution_comparison, save_test_case_as_txt, generate_test_suite | WP0 |
| src/generate_testgraphs_extended.py | Graph Generator für etwas komplexere Graphen | _ensure_simple_graph, _limit_edges_uniform, _apply_perturbation, _write_graph_txt, generate_uniform_clique_blocks, generate_skewed_clique_blocks | WP0 |
| src/reductions/__init__.py | VCC-Reduktionen/Kernelization | - | WP2 |
| src/reductions/branch_and_reduce.py | VCC-Reduktionen/Kernelization | branch_and_reduce | WP2 |
| src/reductions/lower_bound_linear.py | VCC-Reduktionen/Kernelization | compute_lower_bound | WP2 |
| src/reductions/reductions.py | VCC-Reduktionen/Kernelization | is_isolated_vertex, apply_isolated_vertex_reduction, apply_degree_two_folding, apply_twin_removal, apply_twin_folding, apply_domination_reduction | WP2 |
| src/reductions/test_reductions.py | VCC-Reduktionen/Kernelization | TestIsolatedVertexReduction, TestDegreeTwoFolding, TestTwinRemoval, TestTwinFolding | WP2 |
| src/reductions/utils.py | VCC-Reduktionen/Kernelization | reset_stats, log_reduction, timed_step, print_final_stats | WP2 |
| src/simulator.py | Graph generation and perturbation for clique covering experiments. | GraphConfig, GraphGenerator | WP0 |
| src/test.py | Hilfs-/Auswertungsskript | TestRunner | shared |
| src/tests_Tils_constructionsite/hog_to_txt.py | Hilfs-/Auswertungsskript | - | shared |
| src/tests_Tils_constructionsite/test_chalupa_bounds.py | Chalupa-Heuristik (Vertex Clique Cover) | test_chalupa_bounds | shared |
| src/tests_Tils_constructionsite/test_cluster_editing_gadgets.py | Cluster Editing: Kernelization/ILP/Orchestrierung | _read_kopt, test_gadgets_kopt | shared |
| src/tests_Tils_constructionsite/test_graph_creation.py | Hilfs-/Auswertungsskript | write_graph_txt, exact_chromatic_number, theta_via_complement, disjoint_cliques, planted_cluster_editing_gadget | shared |
| src/tests_Tils_constructionsite/test_ilp_against_gold.py | Hilfs-/Auswertungsskript | test_ilp_matches_gold | shared |
| src/tests_Tils_constructionsite/test_reductions_safety.py | Hilfs-/Auswertungsskript | test_reductions_preserve_theta | shared |
| src/utils.py | Hilfs-/Auswertungsskript | get_value, txt_to_networkx | shared |
| src/utils_metrics.py | utils_metrics.py | set_global_seeds, safe_ratio, rel_change, clean_for_plot, ensure_finite_array, nanmean | shared |
| src/wp3_evaluation.py | Enhanced WP3 Evaluation with Statistical Testing and VCC Comparison | BenchmarkResult, WP3EnhancedEvaluator | WP3 |
| src/wrapperV2.py | wrapperV2.py | _compact_int_labels, _is_valid_clique_cover, _validate_result, chalupa_wrapper, _chalupa_warmstart, _chalupa_warmstart_for_coloring | WP1 |
| src/wrappers.py | Wrapper (Heuristik/ILP/Reduced/Interactive) | reduced_ilp_wrapper, interactive_reduced_ilp_wrapper, chalupa_wrapper, ilp_wrapper | WP1 |


## Hierarchie / Abhängigkeiten

```text
src/
├─ generate_test_graphs.py
├─ generate_testgraphs_extended.py
├─ simulator.py
│
├─ algorithms/
│  ├─ chalupa.py
│  ├─ ilp_solver.py
│  ├─ cluster_editing_kernelization.py
│  ├─ cluster_editing_ilp.py
│  ├─ cluster_editing_solver.py
│  └─ helpers.py
│
├─ reductions/
│  ├─ reductions.py
│  ├─ lower_bound_linear.py
│  ├─ branch_and_reduce.py
│  └─ utils.py
│
├─ wrappers.py / wrapperV2.py
│
├─ comparison_chalupa_ilp_evaluation_all_wrappers.py
├─ wp3_evaluation.py
├─ WP4_comparison_VCC_CE.py
├─ WP5_constructionsite.py
├─ WP2BC.py
│
├─ add_ground_truth.py / add_ground_truth_initial_version.py (@Philipp, kann die alte Version weg, benötigst du die noch für Skripte von dir, die aktuell laufen?)
├─ utils.py / utils_metrics.py
└─ test.py
```

## Walkthrough (per IDE)
1. **WP0** – Testgraphen erzeugen (`src/generate_test_graphs.py`, `src/generate_testgraphs_extended.py`)
2. **WP0** - per ILP Ground Truth berechnen + an die Testgraphen-Dateien anhängen (`src/add_ground_truth.py`)
3. **WP1** – VCC: Heuristik/ILP (`src/comparison_chalupa_ilp_evaluation_all_wrappers.py --extended`) 
4. **WP2** – VCC-Reduktionen + Evaluation (`src/reductions/test_reductions.py`, `WP2BC.py`)
5. **WP3** – CE Experimente (`src/wp3_evaluation.py`) 
6. **WP4** – Vergleich VCC vs CE (`src/WP4_comparison_VCC_CE.py --verbose`) # unfinished
7. **WP5** – Rfam Analyse (Baustelle) (`src/WP5_constructionsite.py`) #unfinished

**Hinweise**
- Reproduzierbarkeit: `utils_metrics.set_global_seeds()` 
- Zeitlimits/Gaps für ILP/CE-ILP anpassen
- Pfade: immer aus der Projektwurzel starten

## Details zu den wichtigsten Skripten

### src/add_ground_truth.py
- Kurzbeschreibung: Rekursives Annotieren von Graph-Instanzen mit Ground Truth θ(G), exakt berechnet per ILP auf Ḡ (Kaltstart).
- Integration: Aufruf als Modul: python -m src.add_ground_truth <dir> (Optional: --overwrite via argparse ergänzen.)
- Schlüsselzeile: Clique Cover Number θ(G): K (Calculated by ILP on complement graph Ḡ). Dies unterscheidet das Sktip von `src/algorithms/add_ground_truth_initial`, wo Ground Truth über G ausgewiesen wird
- zugehöriges Arbeitspaket: WP1/WP2

### src/algorithms/chalupa.py
- Kurzbeschreibung: Implementation of Chalupa's heuristic algorithm for clique coloring.  References: [4] David Chalupa. On the effectiveness of the genetic algorithm for the     clique coloring problem. Communications in Mathematics and Computer     Science 1(1), 2023.  [8] David Chalupa. A genetic algorithm with neighborhood search for the     generalized graph coloring problem. Information Sciences, 602:     91-108, 2022.
- Wichtige Klassen: ChalupaHeuristic
- Zugehöriges Arbeitspaket: WP1

### src/algorithms/ilp_solver.py
- Kurzbeschreibung: Integer Linear Programming (ILP) formulation for the vertex clique coloring problem.  kurzer Hinweis zur Integration:  - Aufruf: res = solve_ilp_clique_cover(G, warmstart=heur_cover_dict, time_limit=600, mip_gap=0.0) - Ergebnis: res['theta'] ist θ(G); identisch zu res['chi_complement']. - Für add_ground_truth.py: Zeile Clique Cover Number θ(G): {res['theta']} (Calculated by ILP on complement graph Ḡ). - Für Warmstart: akzeptiert {node: color} oder {color: [nodes...]} oder Liste in G.nodes()-Reihenfolge.
- Wichtige Funktionen: _parse_warmstart, solve_ilp_clique_cover, solve_ilp_direct_on_complement
- Zugehöriges Arbeitspaket: WP1

### src/reductions/reductions.py
- Wichtige Funktionen: is_isolated_vertex, apply_isolated_vertex_reduction, apply_degree_two_folding, apply_twin_removal, apply_twin_folding, apply_domination_reduction, maximal_independent_set_from_matching, apply_crown_reduction, apply_all_reductions
- Zugehöriges Arbeitspaket: WP2

### src/reductions/lower_bound_linear.py
- Wichtige Funktionen: compute_lower_bound
- Zugehöriges Arbeitspaket: WP2

### src/reductions/branch_and_reduce.py
- Wichtige Funktionen: branch_and_reduce
- Zugehöriges Arbeitspaket: WP2

### src/algorithms/cluster_editing_kernelization.py
- Wichtige Klassen: ReductionRule, UnionFind, KernelizationCache, RuleEffectiveness, AdvancedClusterEditingInstance, AdvancedKernelization
- Wichtige Funktionen: load_instance_from_txt
- Zugehöriges Arbeitspaket: WP3

### src/algorithms/cluster_editing_ilp.py
- Wichtige Funktionen: solve_cluster_editing_ilp, _solve_with_cutting_planes, _find_2partition_cuts, validate_clustering, calculate_clustering_cost
- Zugehöriges Arbeitspaket: WP3

### src/algorithms/cluster_editing_solver.py
- Wichtige Klassen: ClusterEditingSolver
- Wichtige Funktionen: benchmark_solver, main
- Zugehöriges Arbeitspaket: WP3

### src/wp3_evaluation.py
- Kurzbeschreibung: Enhanced WP3 Evaluation with Statistical Testing and VCC Comparison
- Wichtige Klassen: BenchmarkResult, WP3EnhancedEvaluator
- Wichtige Funktionen: main
- Zugehöriges Arbeitspaket: WP3

### src/comparison_chalupa_ilp_evaluation_all_wrappers.py
- Kurzbeschreibung: comparison_chalupa_ilp_evaluation_all_wrappers.py WP1.c Evaluation ++WP2: Comprehensive comparison of all solver variants Extends the original comparison to include reduced and interactive reduced ILP methods
- Wichtige Klassen: WP1cEvaluator, ExtendedWP1cEvaluator
- Wichtige Funktionen: _paired_speedup_stats, _normalize_wrapper_result, _merge_into_result, _compat_add_quality_columns, main
- Zugehöriges Arbeitspaket: WP1

### src/wrapperV2.py
- Kurzbeschreibung: wrapperV2.py WRAPPERS for running reduction, heuristic and ILP on clique-cover Instances... THOUGHTS --------------------- How to handle UB for best ILP Performance? ------------------------ Using the upper bound (UB) strictly is optional and depends on our goals: - In the OG code (wrappers.py), `interactive_reduced_ilp_wrapper` uses UB from the Chalupa heuristic to guide reduction rounds, stopping when no improvement is made.   This makes sense if we want to minimize problem size before ILP and avoid wasting effort on non-promising reductions. - Strictly enforcing UB inside the ILP (e.g., as a constraint on the number of colors) could speed up solving by pruning solutions worse than the heuristic. Here "hard equality" would mean forcing the ILP to *exactly* match the UB value (`sum(colors) == UB`). This is risky because if the heuristic UB is not optimal, we might exclude the true optimal solution entirely. - A safer approach imo is to use UB as a **hard inequality** (`sum(colors) ≤ UB`), which never eliminates the optimal solution but can still prune the search space. - For simplified `reduced_ilp_wrapper`, we could pass UB into the ILP to potentially reduce runtime. This would involve adding an argument to the ILP solver to accept a UB and incorporate it into the model.  In short: I would suggest to use UB as an upper limit (≤ UB) to contribute to better performance and avoid enforcing UB as an exact target (== UB) unless we are certain it is optimal.  ----------------------- Theory interlude --------------------------------  χ(G) - Chromatic Number     Definition: Die minimale Anzahl an Farben, die benötigt wird, um die Knoten von G so zu färben,                 dass keine zwei benachbarten Knoten dieselbe Farbe haben.      Interpretation: Jede Farbe steht für eine unabhängige Menge (kein Kantenpaar innerhalb einer Farbe).                     Färben = Partition von V(G) in minimale Anzahl unabhängiger Mengen.      χ(G) = Größe der kleinsten Partition von V(G) in independent sets.   θ(G) - Clique Cover Number     Definition: Die minimale Anzahl an Cliquen, deren Vereinigung alle Knoten von G überdeckt.      Interpretation: Jede Clique kann isoliert werden und deckt einen Teil der Knoten ab, so dass am Ende alle Knoten abgedeckt sind.     Clique Cover = Partition von V(G) in minimale Anzahl vollständiger Teilgraphen.      θ(G) = Größe der kleinsten Partition von V(G) in cliques.   why important?     - Cliquen in G ↔ unabhängige Mengen im Komplementgraphen Ḡ     - Eine Clique in G ist ein independent set in Ḡ.     - Eine independent set in G ist eine Clique in Ḡ.     - im OG Code ist, so wie ich das verstanden habe, folgendes passiert: ILP hat χ(G) berechnet (Färbung von G) → und das war somit nicht die gesuchte Zahl für Clique Cover.     - Bsp:         - G: Dreieck (3 Knoten, alle verbunden): χ(G) = 3 (jeder Knoten andere Farbe), θ(G) = 1 (das Dreieck selbst ist eine Clique, deckt alle Knoten),         - Ḡ: 3 isolierte Knoten: χ(Ḡ) = 1 → stimmt mit θ(G) überein.       - ILP sollte jetzt mit dem folgenden Code Ḡ färben, wonach dann gilt: θ(G) = χ(Ḡ)         (Clique Cover von G = Färbung des Komplements Ḡ).  ------------------------------ θ(G) vs χ(Ḡ) ----------------------------------  Whats different now:     - all wrappers operate on the complement graph Ḡ (consistent Clique Cover via θ(G)=χ(Ḡ)). should lead to correct results for Cloque Cover.?     - reduced_ilp_wrapper applies reductions on Ḡ before ILP interactive_reduced_ilp_wrapper fixes the loop logic: keeps improving UB on Ḡ until it stops decreasing,         then ILP on reduced Ḡ     - interactive_reduced_ilp_wrapper fixes the loop logic: keeps improving UB on Ḡ until it stops decreasing, then ILP on reduced Ḡ     - Added a chalupa_wrapper that also works on Ḡ (UB for θ(G))     - ilp_wrapper(..., use_warmstart: bool)` toggles whether a Chalupa-based warmstart         is computed and passed into the ILP solver. Default is False (fair ILP baseline).     - Interoperability:         - script uses `solve_ilp_clique_cover` from `algorithms.ilp_solver` which computes θ(G)             by coloring the complement graph Ḡ (χ(Ḡ) = θ(G)).         - If `use_warmstart=True` and Chalupa is available, a warmstart assignment is built.  Dependencies: - utils.txt_to_networkx (reads graph from .txt) - algorithms.ilp_solver.solve_ilp_clique_cover - algorithms.chalupa.ChalupaHeuristic (optional; only if use_warmstart=True)
- Wichtige Funktionen: _compact_int_labels, _is_valid_clique_cover, _validate_result, chalupa_wrapper, _chalupa_warmstart, _chalupa_warmstart_for_coloring, ilp_wrapper, reduced_ilp_wrapper, interactive_reduced_ilp_wrapper, batch_ilp, debug_clique_cover
- Zugehöriges Arbeitspaket: WP1

### src/wrappers.py
- Wichtige Funktionen: reduced_ilp_wrapper, interactive_reduced_ilp_wrapper, chalupa_wrapper, ilp_wrapper
- Zugehöriges Arbeitspaket: WP1

### src/simulator.py
- Kurzbeschreibung: Graph generation and perturbation for clique covering experiments.
- Wichtige Klassen: GraphConfig, GraphGenerator
- Zugehöriges Arbeitspaket: WP0

### src/generate_test_graphs.py
- Kurzbeschreibung: Generate and save test cases for clique covering experiments.
- Wichtige Funktionen: visualize_graph, visualize_solution_comparison, save_test_case_as_txt, generate_test_suite
- Zugehöriges Arbeitspaket: WP0

### src/utils.py
- Wichtige Funktionen: get_value, txt_to_networkx
- Zugehöriges Arbeitspaket: shared

### src/utils_metrics.py
- Kurzbeschreibung: utils_metrics.py Zentrale Helfer für Evaluations-Metriken, Plots (Datenhygiene), Reproduzierbarkeit und simple Heuristiken. Ziel: konsistente Auswertungen in WP1-WP5.
- Wichtige Funktionen: set_global_seeds, safe_ratio, rel_change, clean_for_plot, ensure_finite_array, nanmean, safe_idxmax, graph_density, should_kernelize, estimate_loglog_slope
- Zugehöriges Arbeitspaket: shared

### src/WP2BC.py
- Kurzbeschreibung: Implementiert eine Auswertung der Ergebnisse (csv-Dateien) von src/comparison_chalupa_ilp_evaluation_all_wrappers.py
- WP2b: Vergleich Kernelization vs. keine Kernelization.
- WP2c: Analyse von Verbesserung (Speedup, Kosten, Qualität) auf Basis bereits vorhandener Evaluationsläufe (CSV/JSON-Ergebnisse).
- Cave! Dieses Skript erzeugt keine neuen Solver-Läufe, sondern arbeitet ausschließlich mit bereits gespeicherten Resultaten und ist somit vom Vorhandensein der results aus src/comparison_chalupa_ilp_evaluation_all_wrappers.py abhängig


### src/WP4_comparison_VCC_CE.py
- Kurzbeschreibung: WP4: Comparison of Vertex Clique Cover and Cluster Editing Solutions  This module implements the comparison framework for WP4, comparing solutions from the Vertex Clique Cover (VCC) and Cluster Editing (CE) problems.  Main objectives: - Compare θ(G) from VCC with C(G) from CE - Analyze solution quality and structural differences - Implement cross-optimization heuristics (Bonus) - Generate comprehensive reports and visualizations  Usage:     python src/wp4_comparison.py [options]  Options:     --test-dir PATH      Directory with test graphs (default: test_graphs/generated)     --output-dir PATH    Output directory for results (default: results/wp4)     --quick              Run quick test with fewer instances     --verbose            Enable verbose output     --skip-visualizations   Skip generating plots (faster execution)
- Wichtige Klassen: ClusteringResult, ComparisonResult, SolverAdapter, ComparisonFramework, CrossOptimizationHeuristic, StatisticalAnalyzer, Visualizer, ReportGenerator
- Wichtige Funktionen: main, load_test_graphs
- Zugehöriges Arbeitspaket: WP4

## Mathematischer Kern des ganzen Projekts
**Symbole.** G=(V,E); Ḡ: Komplement; θ(G): Clique-Cover-Zahl; χ(G): chromatische Zahl; z_ij∈{0,1}; x[v,k], y[k] ∈{0,1}.

### VCC ≡ Färbung des Komplements
[<a id="eq-vcc-coloring"></]()a>
**(1)**  θ(G) = χ(Ḡ)

#### ILP-Modell A (Assignment/Coloring)
<a id="eq-ilp-assign"></a>
**(2a)**  ∑_k x[v,k] = 1 (∀v)  
**(2b)**  x[v,k] ≤ y[k]  
**(2c)**  x[u,k] + x[v,k] ≤ y[k] (∀{u,v}∈E(Ḡ))  
**(2d)**  min ∑_k y[k]

#### ILP‑Modell B (Set-Cover über Cliquen von G)
<a id="eq-ilp-cover"></a>
**(3a)**  ∑_{C: v∈C} z[C] ≥ 1 (∀v)  
**(3b)**  min ∑_C z[C]

### Schranken für VCC
<a id="eq-bounds"></a>
**(4)**  ω(G)=α(Ḡ) ≤ θ(G) ≤ UB

### Cluster Editing (gewichtet)
<a id="eq-ce-objective"></a>
**(5)**  min Σ_{i<j} [ w_del(i,j)·(1−z_ij)  für {i,j}∈E  +  w_ins(i,j)·z_ij  für {i,j}∉E ]

#### Transitivität
<a id="eq-triangle"></a>
**(6a)**  z_ij + z_jk − 1 ≤ z_ik;  **(6b)**  z_ij + z_ik − 1 ≤ z_jk;  **(6c)**  z_ik + z_jk − 1 ≤ z_ij


## Böcker-Kernelization (Cluster Editing) - Regeln & Code-Abgleich

**Notation.** Ungerichteter Graph \(G=(V,E)\) mit Gewichten \(s(u,v)\) für jedes Paar \(\{u,v\}\). 
\(s(u,v)>0\) für vorhandene Kante (Löschkosten), \(s(u,v)<0\) für Nicht-Kante (Einfügekosten). 
Für \(U\subseteq V\): \(s(v,U)=\sum_{u\in U}s(v,u)\). \(N(u)\) ist die Nachbarschaft von \(u\) in \(G\).

### Regel (CE-R1) – Heavy Non‑Edge  <a id="ce-r1"></a>
**(7)**  \(|s(u,v)| \;\ge\; \sum_{w\in N(u)} s(u,w) \;\;\Rightarrow\;\; uv \text{ auf }\textsf{forbidden}\).  
**Repo:** `cluster_editing_kernelization.apply_heavy_non_edges_batch()` prüft \(|s(u,v)| \ge s(u,N(u))\) für \(u<v\) und markiert \(uv\) als *forbidden* (setzt Gewicht auf FORBIDDEN und entfernt ggf. Kante).  
*Anmerkung:* Die Implementierung prüft standardmäßig nur die **u‑Seite** (für \(u<v\)). Es ist **korrigierend, aber weniger scharf** als die zweiseitige Variante („… oder analog mit \(v\)“). Optional kann man symmetrisieren (siehe Hinweis unten).

### Regel (CE-R2) – Heavy Edge (einseitig)  <a id="ce-r2"></a>
**(8)**  \(s(u,v) \;\ge\; \sum_{w\in V\setminus\{u,v\}} |s(u,w)| \;\;\Rightarrow\;\; \text{merge } u,v.\)  
**Repo:** `apply_heavy_edge_single_batch()` summiert \(\sum_{w\neq u,v}|s(u,w)|\) und mergt bei erfüllter Schranke.

### Regel (CE-R3) – Heavy Edge (beidseitig)  <a id="ce-r3"></a>
**(9)**  \(s(u,v) \;\ge\; \sum_{w\in N(u)\setminus\{v\}} s(u,w) \;+\; \sum_{w\in N(v)\setminus\{u\}} s(v,w) \;\;\Rightarrow\;\; \text{merge } u,v.\)  
**Repo:** `apply_heavy_edge_both_batch()` implementiert genau diese beidseitige Schranke.

### Regel (CE-R4) – Almost‑Clique (Min‑Cut)  <a id="ce-r4"></a>
Für \(C\subseteq V\) und den **Min‑Cut** \(k_C\) im von \(C\) induzierten Subgraphen gilt:  
**(10)**  \(k_C \;\ge\; \sum_{\substack{u,v\in C\\ s(u,v)\le 0}} |s(u,v)| \;+\; \sum_{\substack{u\in C,\ v\notin C\\ s(u,v)>0}} s(u,v) \;\;\Rightarrow\;\; \text{merge } C.\)  
**Repo:** `apply_almost_clique_mincut()` konstruiert Kandidaten \(C\) greedy und testet mit `nx.minimum_cut_value(...)` genau die o. g. Ungleichung.

### Regel (CE-R5) – Similar Neighborhood (DP)  <a id="ce-r5"></a>
Setze exklusive Nachbarschaften \(N_u=N(u)\setminus(N(v)\cup\{v\})\), \(N_v=N(v)\setminus(N(u)\cup\{u\})\), \(W=V\setminus(N_u\cup N_v\cup\{u,v\})\) und  
\(\Delta_u=s(u,N_u)-s(u,N_v)\), \(\Delta_v=s(v,N_v)-s(v,N_u)\). Dann gilt mit DP‑Schranke:  
**(11)**  \( s(u,v) \;\ge\; \displaystyle\max_{\substack{C_u,C_v\subseteq W\\ C_u\cap C_v=\varnothing}} \min\{\, s(v,C_v)-s(v,C_u)+\Delta_v,\;\; s(u,C_u)-s(u,C_v)+\Delta_u \,\} \;\Rightarrow\; \text{merge } u,v.\)  
**Repo:** `apply_similar_neighborhood_dp()` baut \(N_u,N_v,W,\Delta_u,\Delta_v\) und ruft `_compute_dp_bound(B, Δu, Δv)`; dort wird die DP genau wie in der Literatur implementiert (Aggregation von \((X,Y)\)‑Summen, danach \(\max\min\{\cdot\}\)).  
*Hinweis:* Bei **reellen** Gewichten ggf. Quantisierung/Rundung vorsehen (Paper-Empfehlung).

**Vorstufe („Regel 0“) - Critical Cliques.** `apply_critical_cliques_batch()` kontrahiert Knoten mit identischer *closed neighborhood*; Standard‑Kernelisierungsschritt.

> **Optionale Schärfung (CE‑R1):** statt nur \(u\) kann man symmetrisch testen -> #könnte man noch implementieren, ist mir spät aufgefallen
> `if abs(s(u,v)) >= min( s(u,N(u)), s(v,N(v)) ): forbid(u,v)`  
> Das erhöht die Reduktionskraft ohne Korrektheitsrisiko.



## Chalupa‑Heuristik (VCC) – Mathematischer Abgleich & Code‑Mapping  <a id="chalupa-verify"></a>

**Ziel.** VCC sucht die kleinste Anzahl Cliquen zur Überdeckung aller Knoten. Implementiert ist eine **Iterated‑Greedy (IG)**‑Konstruktion für eine obere Schranke \(\text{UB}\) und eine **Greedy/RLS**‑Suche für eine untere Schranke \(\text{LB}\) (max. unabhängige Menge).

### Formale Bausteine
- Äquivalenz: **(1)** \(\;\theta(G)=\chi(\bar G)\).  
- UB via *Greedy Clique Covering* aus einer Permutation \(\pi\):  
  **(12)**  Erzeuge Cliquen \(C_1,C_2,\dots\) iterativ; füge \(v\) in die **kleinste zulässige** vorhandene Clique (First‑Fit), sonst starte neue.  
  Zulässigkeitstest in \(G\): \(v\) ist mit **allen** Mitgliedern von \(C_j\) adjazent.
- LB via *Greedy MIS*:  
  **(13)**  Durchlaufe \(\pi\); wenn \(v\) zu bislang gewähltem \(I\) **keine** Kante hat, dann \(I\leftarrow I\cup\{v\}\).  
  Danach **RLS/IG‑Verbesserungen** durch Permutations‑Jumps/Neustarts.

### Repo‑Implementierung vs Theorie
- **Permutation → Greedy‑Cover (UB)**  
  `ChalupaHeuristic.find_greedy_clique_covering(π)` implementiert (12):  
  `find_equal(v, sizes)` setzt den *First‑Fit*‑Test um:  
  **(14)**  Scanne Nachbarn von \(v\); dekrementiere temporär Größen `sizes[c]` für jede Clique \(c\) mit Nachbar; **Kandidaten** sind genau die \(c\) mit `sizes[c]==0` ⇒ \(v\) ist mit allen in \(c\) verbunden. Wähle den **kleinsten Index** (First‑Fit).
- **Iterated Greedy**  
  `iterated_greedy_clique_covering()` erzeugt \(\pi\) zufällig, bildet Cover via (12)/(14), hält bestes Cover \(|\mathcal C|\) als \(\text{UB}\), und ändert \(\pi\) (Neustarts/Jumps), bis Abbruchkriterien greifen.
- **Greedy MIS + RLS (LB)**  
  `greedy_independent_set(π)` implementiert (13). `find_maximum_independent_set()` macht RLS‑Verbesserungen (`jump`, `uniformly_random`, neue Permutationen). Ergebnis ist \(\text{LB}=|I|\).
- **Gesamtausgabe**  
  `run()` gibt \([\text{LB},\text{UB}]\), bestes Cover und bestes \(I\) zurück. Tests sichern \( \text{LB}\le \theta(G)\le \text{UB}\).
- tbc, hab noch nicht alles gegengecheckt 

## Literatur (Auswahl)
- Chalupa, D. (2016). Construction of Near‑Optimal Vertex Clique Covering for Real‑World Networks.
- Böcker, S., Briesemeister, S., & Klau, G.W. (2011). Exact Algorithms for Cluster Editing.
- Grötschel, M., & Wakabayashi, Y. (1989). A Cutting Plane Algorithm for a Clustering Problem.
- Bansal, N., Blum, A., & Chawla, S. (2004). Correlation Clustering.
- Mutzel, P. (2022). Graph Coloring: ILP Formulations.
