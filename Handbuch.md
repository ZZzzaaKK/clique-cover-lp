# Handbuch: Nutzung und Verständnis des Softwarepakets *clique-cover-lp*

Dieses Handbuch beschreibt die Benutzung der bereitgestellten Skripte sowie die internen Pakete und Algorithmen. Ziel ist es, das Paket effizient auszuführen, Ergebnisse zu reproduzieren und die Datenflüsse zu verstehen. Die Beschreibung orientiert sich an der vorhandenen Projektstruktur und den Skripten.

---

## 1. Überblick

- **Zweck des Pakets**
  - Bestimmung der **Vertex Clique Cover Number** $\theta(G)$ mit
    - einer **Heuristik** (Chalupa) und
    - einem **ILP-Solver** (Gurobi).
  - **Cluster Editing (CE)** mit Kernelization-Regeln und ILP.
  - **Auswertung**: Parser für Ergebnisdateien & Plot-Skripte.
  - **Testdaten**: Generator für synthetische Graphen.
  - **Tests**: Unit-Tests für Reduktionen und CE-Minimal.

- **Zentrale Bibliotheken**
  - `networkx` (Graphstruktur und -operationen)
  - `gurobipy` (ILP mit Gurobi)
  - `numpy`, `pandas`
  - `matplotlib`, `seaborn` (Visualisierung)


## 2. Verzeichnisstruktur (typisch)

- `src/`
  - `algorithms/`
    - `chalupa.py` – Heuristik für Vertex Clique Cover.
    - `ilp_solver.py` – ILP-Formulierung für Clique Cover.
    - `cluster_editing.py` – CE-Kernelization (erweiterter Stack).
    - `cluster_editing_minimal.py` – CE-Minimal-Stack (didaktisch).
  - `reductions/`
    - `reductions.py` – Reduktionsregeln (VCC/VCH-Kontext).
    - `utils.py` – Zähler/Timing für Reduktionen.
  - `wrappers.py` – Einfache Aufrufe/Workflows für Heuristik, ILP, CE.
  - `simulator.py` – Generator synthetischer Graphen.
  - `generate_test_graphs.py` – CLI für Testgraphen & Visualisierung.
  - `utils.py` – Ein-/Auslese-Helfer (TXT → NetworkX).
  - `comparison.py` – Parser & Auswertung von Ergebnisprotokollen.
  - `add_vertex_clique_cover_number.py` – $\theta(G)$ als Attribut ergänzen.
  - `add_chromatic_number.py` – Chromatische Zahl ergänzen.
  - `test.py` – Test-Runner für verschiedene Verfahren.
  - `test_reductions.py` – Unit-Tests Reduktionen.
  - `test_cluster_editing.py` – Unit-Tests CE-Minimal.
- `test_graphs/` – Beispiel-/Generator-Daten.
- `results/` – Ergebnisdateien (Logs, CSV, Plots).

---

## 3. Datenformate

### 3.1 Eingabegraph (TXT)
- **Format**: Adjazenzliste zeilenweise
  - Beispiel: `u: v1 v2 v3`
- **Hinweise**:
  - Knoten-IDs sind Integer oder Strings; intern erfolgt meist ein $0$‑Index‑Relabeling.
  - Attribute (z. B. „Chromatic Number“ oder „Vertex Clique Cover Number“) können als eigene Zeilen im Anschluss aufgeführt sein.
  - Parser bricht das Adjazenzlesen ab, sobald ein nicht-adjazenter Abschnitt beginnt.

### 3.2 CE-Gewichte (für Cluster Editing)
- **Konvention** (Minimal-Stack):
  - **Positiv**: Löschungskosten einer vorhandenen Kante.
  - **Negativ**: Einfügekosten für eine nicht vorhandene Kante.
- **Ablage**: Je nach Skript als Mapping $(u, v) \mapsto \text{gewicht}$ oder in einer gewichteten Kantenliste.

### 3.3 Ergebnisprotokolle
- **Struktur**:
  - Blöcke pro Instanz, getrennt durch `------------------------------`.
  - Key‑Value‑Zeilen mit Laufzeiten, Kosten, Parametern, Ergebniswerten.

---

## 4. Typische Workflows

### 4.1 Vertex Clique Cover ($\theta(G)$)
- **Ziel**: Bestimme $\theta(G)$ heuris­tisch und/oder exakt via ILP.
- **Ablauf**:
  1. Graph im TXT-Format bereitstellen.
  2. Heuristik ausführen (Chalupa) für erste Schranken.
  3. ILP ausführen (ggf. mit Obergrenze aus Heuristik).
- **Ausführung** (Beispiel über `test.py`):
  ```bash
  # Heuristik
  python src/test.py --chalupa path/to/graphs
  # ILP
  python src/test.py --ilp path/to/graphs
  # Kombiniert (je nach Flags)
  python src/test.py --chalupa --ilp path/to/graphs
  ```
- **Output**:
  - Pro Instanz: $\theta(G)$, Laufzeit, Status, ggf. Färbung/Konfiguration.
  - Sammellogs in `results/`.

### 4.2 Cluster Editing (CE)
- **Ziel**: Transformation des Graphen in Clustergraph mittels minimaler Editierungen.
- **Varianten**:
  - **Erweiterter Stack** (`algorithms/cluster_editing.py`): Kernelization-Regeln (kritische Cliquen u. a.) + ILP.
  - **Minimal-Stack** (`algorithms/cluster_editing_minimal.py`): Schmale Lehrvariante (edge-cuts-Regel) + ILP.
- **Ausführung via Wrapper**:
  ```bash
  # Voller CE-Stack
  python src/test.py --cluster-editing path/to/graphs
  # Minimaler CE-Stack
  python src/test.py --reduced-cluster-editing path/to/graphs
  ```
- **Output**:
  - Anzahl Edits, Clusteranzahl, Kernelgröße, Laufzeiten, ggf. Modifikationsliste.

### 4.3 Testgraphen erzeugen & visualisieren
- **Generator**:
  ```bash
  python src/generate_test_graphs.py --out test_graphs/generated \
      --num-graphs 20 --type skewed --min-size 8 --max-size 12 \
      --p-in 0.9 --p-out 0.1 --noise 0.2 --seed 33
  ```
- **Visualisierung**:
  - Optionaler Modus im Skript (`--plot`), speichert PNGs/SVGs in Unterordnern.
- **Verwendung**:
  - Daten können direkt in `test.py`, `comparison.py` und Wrappern benutzt werden.

### 4.4 Ergebnisanalyse & Plots
- **Parser/Auswerter**:
  ```bash
  python src/comparison.py results/raw/generated/perturbed/*.txt
  ```
- **Funktionen**:
  - Aggregation über Logs, Kennzahlenvergleich, Plot-Erstellung (Seaborn/Matplotlib).
- **Ausgaben**:
  - CSV/PNG in `results/` (Verzeichnisse abhängig vom Skript).

### 4.5 Ground-Truth ergänzen
- **$\theta(G)$ (Clique Cover Number)**:
  ```bash
  python src/add_vertex_clique_cover_number.py path/to/graphs
  ```
- **Chromatische Zahl**:
  ```bash
  python src/add_chromatic_number.py path/to/graphs
  ```

---

## 5. Skript-Referenz (Benutzbarkeit & Ausführung)

### 5.1 `src/utils.py`
- **Funktionen**:
  - `txt_to_networkx(txt_filepath)`
    - Liest Adjazenzliste und erzeugt `nx.Graph`.
    - Achtet auf Trenner/Attribute; führt $0$‑Index‑Relabeling durch.
  - `get_value(txt_filepath, attribute_name)`
    - Liest einen Attributwert (z. B. „Vertex Clique Cover Number“) aus der Datei.
- **Benutzung**: Wird von den Wrappern und Analyse-Skripten verwendet, kann auch direkt importiert werden.

### 5.2 `src/algorithms/chalupa.py`
- **Zweck**: Heuristische Clique-Cover-Bestimmung.
- **Kernklasse**: `ChalupaHeuristic`
  - `run(G, **kwargs)` → liefert u. a. obere/untere Schranken.
- **Einsatz**:
  - Direkt via Wrapper (`wrappers.py`) oder eigene Skripte.
- **Input/Output**:
  - Input: `networkx.Graph`
  - Output: Dict/Struktur mit Schranken und Covering.

### 5.3 `src/algorithms/ilp_solver.py`
- **Zweck**: Exakte Clique-Cover-Bestimmung per ILP (Gurobi).
- **Funktion**: `solve_ilp_clique_cover(G, time_limit=600, require_optimal=False)`
  - Minimiert die Zahl der Farben (Cover).
  - Rückgabedaten enthalten u. a. `chromatic_number` (je nach Modellierung entspricht dies $\theta(G)$ auf dem Komplementgraphen), Färbung, Status.
- **Einsatz**:
  - Über Wrapper oder direkt im eigenen Code.
- **Hinweise**:
  - `time_limit` beachten; `require_optimal` steuert Verhalten bei `TIME_LIMIT`.

### 5.4 `src/algorithms/cluster_editing.py`
- **Zweck**: Kernelization-Regeln + ILP für CE.
- **Kernfunktion**: `kernelize(G, weights=None, k=None, max_iterations=100)`
  - Iteriert Regeln bis Fixpunkt/Limit.
  - Rückgabe umfasst reduzierten Graph, Gewichte, verbleibendes Budget/Parameter, Abbildungen und Modifikationen.
- **Einsatz**:
  - Über Wrapper (`--cluster-editing`).
- **Input/Output**:
  - Input: `nx.Graph`, optional Gewichte/Parameter.
  - Output: Reduzierte Instanz und Metadaten; ILP-Teil liefert finale Edit-Kosten/Konfiguration.

### 5.5 `src/algorithms/cluster_editing_minimal.py`
- **Zweck**: Minimaler CE-Stack (Edge-Cut-Regel) + ILP.
- **Kernfunktion**: `kernelize_edge_cuts(G, weights=None, k=None)`
  - Liefert reduzierten Graphen, Gewichte, Parameter und Modifikationsset.
- **Einsatz**:
  - Über Wrapper (`--reduced-cluster-editing`) oder für didaktische Zwecke.
- **Input/Output**:
  - Analog zum vollen CE-Stack, in reduzierter Regelmenge.

### 5.6 `src/reductions/reductions.py`
- **Zweck**: Generische Reduktionen (VCC/VCH-Kontext).
- **Funktionen**:
  - `apply_isolated_vertex_reduction(G)`, `apply_degree_two_folding(G)`, `apply_twin_folding_or_removal(G)`, `apply_domination_reduction(G)`, `apply_crown_reduction(G)`
  - `apply_all_reductions(G, ...)`
- **Einsatz**:
  - Direkt in eigenen Workflows oder via Wrapper, insbesondere vor ILP-Läufen.

### 5.7 `src/reductions/utils.py`
- **Zweck**: Zähler/Timing/Logging.
- **Funktionen**:
  - `reduction_stats`, Dekoratoren, `print_final_stats()`
- **Einsatz**: Optionale Statistik über Reduktionsaufrufe.

### 5.8 `src/wrappers.py`
- **Zweck**: Einheitliche Ausführung der Algorithmen.
- **Typische Funktionen/Flags**:
  - `chalupa_wrapper`, `reduced_chalupa_wrapper`
  - `ilp_wrapper`, `reduced_ilp_wrapper`, `interactive_reduced_ilp_wrapper`
  - `cluster_editing_wrapper`, `reduced_cluster_editing_wrapper`
  - `minimal_cluster_editing_wrapper`, `minimal_reduced_cluster_editing_wrapper`
- **Benutzung** (CLI über `test.py`, s. u.).
- **Output**:
  - Einträge/Zeilen für Ergebnisprotokolle mit Kennzahlen und Laufzeiten.

### 5.9 `src/simulator.py`
- **Zweck**: Erzeugung synthetischer Graphen.
- **Wichtige Inhalte**:
  - `GraphConfig` (Dataclass) – Konfiguration.
  - `GraphGenerator` – Erzeugt Cliquenstrukturen und fügt Rauschen/Perturbationen hinzu.
- **Benutzung**:
  - Direkt oder über `generate_test_graphs.py`.

### 5.10 `src/generate_test_graphs.py`
- **Zweck**: CLI zum Generieren von Testgraphen.
- **Aufruf**:
  ```bash
  python src/generate_test_graphs.py --out test_graphs/generated \
      --num-graphs 10 --type uniform --min-size 6 --max-size 9 --p-in 0.85 --p-out 0.1
  ```
- **Optionen**:
  - Anzahl, Größenbereiche, Verteilungstyp, inner/outer edge probability, Seed.
- **Ausgaben**:
  - TXT-Dateien und optional Plots.

### 5.11 `src/comparison.py`
- **Zweck**: Parser/Aggregation/Plots für Ergebnislogs.
- **Aufruf**:
  ```bash
  python src/comparison.py results/raw/*.txt
  ```
- **Funktion**:
  - Liest Blöcke je Instanz, extrahiert Kennzahlen, erstellt Statistiken/Plots.

### 5.12 `src/add_vertex_clique_cover_number.py`
- **Zweck**: Ergänzt $\theta(G)$ in Graphdateien.
- **Aufruf**:
  ```bash
  python src/add_vertex_clique_cover_number.py path/to/graph_dir
  ```
- **Ablauf**:
  - Liest Graph, berechnet $\theta(G)$ (ggf. via Komplementgraph/Wrapper), schreibt Attributzeile.

### 5.13 `src/add_chromatic_number.py`
- **Zweck**: Ergänzt chromatische Zahl in Graphdateien.
- **Aufruf**:
  ```bash
  python src/add_chromatic_number.py path/to/graph_dir
  ```

### 5.14 `src/test.py`
- **Zweck**: zentraler Test-Runner.
- **Optionen**:
  - `--chalupa`, `--reduced-chalupa`
  - `--ilp`, `--reduced-ilp`, `--interactive-reduced-ilp`
  - `--cluster-editing`, `--reduced-cluster-editing`
  - `--timeout TIMEOUT`
  - Optional: Pfad zu Daten (Default z. B. `test_graphs/generated/perturbed`)
- **Beispiel**:
  ```bash
  python src/test.py --chalupa --ilp --cluster-editing test_graphs/generated/perturbed
  ```

### 5.15 `src/test_reductions.py`
- **Zweck**: Unit-Tests für Reduktionen.
- **Aufruf**:
  ```bash
  python src/test_reductions.py
  ```

### 5.16 `src/test_cluster_editing.py`
- **Zweck**: Unit-Tests für CE-Minimal (Kernelization + ILP).
- **Aufruf**:
  ```bash
  python src/test_cluster_editing.py
  ```

---

## 6. Eingabe-/Ausgabe-Pfade und Konventionen

- **Eingabepfad**: Ordner mit TXT-Graphen, die von `utils.txt_to_networkx` verstanden werden.
- **Ausgabepfad**: `results/` mit Unterordnern pro Experiment/Variante.
- **Benennungen**:
  - Häufig: `*_perturbationXX.txt`, `uniform_n*_s*_r*.txt` u. ä.
  - Attribute/Metadaten im Footer der TXT-Dateien.

---

## 7. Beispiele der Handhabung

### 7.1 $\theta(G)$ für vorhandene Graphen bestimmen
```bash
python src/test.py --chalupa data/graphs
python src/test.py --ilp data/graphs --timeout 600
python src/comparison.py results/raw/*.txt
```

### 7.2 CE-Minimal auf Testgraphen
```bash
python src/test.py --reduced-cluster-editing test_graphs/generated
python src/comparison.py results/raw/generated/*.txt
```

### 7.3 Testgraphen erzeugen, dann evaluieren
```bash
python src/generate_test_graphs.py --out test_graphs/generated --num-graphs 50 \
  --type skewed --min-size 8 --max-size 12 --p-in 0.9 --p-out 0.1 --noise 0.2
python src/test.py --chalupa --ilp test_graphs/generated
python src/comparison.py results/raw/generated/*.txt
```

---

## 8. Mathematischer Kern des ganzen Projekts

**Symbole.** $G=(V,E)$; $\bar{G}$: Komplement; $\theta(G)$: Clique-Cover-Zahl; $\chi(G)$: chromatische Zahl; $z_{ij}\in\{0,1\}$; $x_{v,k}, y_k \in\{0,1\}$.

### VCC $\equiv$ Färbung des Komplements
**(1)**
$$
\theta(G) = \chi(\bar{G})
$$

#### ILP-Modell A (Assignment/Coloring)
**(2a)** $\displaystyle \sum_{k} x_{v,k} = 1 \quad (\forall v)$  
**(2b)** $x_{v,k} \le y_k$  
**(2c)** $x_{u,k} + x_{v,k} \le y_k \quad (\forall \{u,v\}\in E(\bar{G}))$  
**(2d)** $\min \displaystyle \sum_{k} y_k$

#### ILP‑Modell B (Set-Cover über Cliquen von $G$)
**(3a)** $\displaystyle \sum_{C:\ v\in C} z_C \ge 1 \quad (\forall v)$  
**(3b)** $\min \displaystyle \sum_{C} z_C$

### Schranken für VCC
**(4)** $\omega(G)=\alpha(\bar{G}) \le \theta(G) \le \text{UB}$

### Cluster Editing (gewichtet)
**(5)**
$$
\min \sum_{i<j} \Big( w_{\mathrm{del}}(i,j)\cdot(1-z_{ij}) \ \text{für } \{i,j\}\in E \;+\; w_{\mathrm{ins}}(i,j)\cdot z_{ij} \ \text{für } \{i,j\}\notin E \Big)
$$

#### Transitivität
**(6a)** $z_{ij} + z_{jk} - 1 \le z_{ik}$;  
**(6b)** $z_{ij} + z_{ik} - 1 \le z_{jk}$;  
**(6c)** $z_{ik} + z_{jk} - 1 \le z_{ij}$

## Böcker-Kernelization (Cluster Editing) – Regeln & Code-Abgleich

**Notation.** Ungerichteter Graph $G=(V,E)$ mit Gewichten $s(u,v)$ für jedes Paar $\{u,v\}$.  
$s(u,v)>0$ für vorhandene Kante (Löschkosten), $s(u,v)<0$ für Nicht-Kante (Einfügekosten).  
Für $U\subseteq V$: $s(v,U)=\sum_{u\in U}s(v,u)$. $N(u)$ ist die Nachbarschaft von $u$ in $G$.

### Regel (CE-R1) – Heavy Non‑Edge
**(7)**  $\lvert s(u,v)\rvert \ \ge\ \sum_{w\in N(u)} s(u,w) \ \Rightarrow\ uv \text{ ist \textsf{forbidden}}.$

### Regel (CE-R2) – Heavy Edge (einseitig)
**(8)**  $s(u,v) \ \ge\ \sum_{w\in V\setminus\{u,v\}} \lvert s(u,w)\rvert \ \Rightarrow\ \text{merge } u,v.$

### Regel (CE-R3) – Heavy Edge (beidseitig)
**(9)**  $s(u,v) \ \ge\ \sum_{w\in N(u)\setminus\{v\}} s(u,w) \;+\; \sum_{w\in N(v)\setminus\{u\}} s(v,w) \ \Rightarrow\ \text{merge } u,v.$

### Regel (CE-R4) – Almost‑Clique (Min‑Cut)
**(10)**
$$
k_C \ \ge\ \sum_{\substack{u,v\in C\\ s(u,v)\le 0}} \lvert s(u,v)\rvert \;+\; \sum_{\substack{u\in C,\ v\notin C\\ s(u,v)>0}} s(u,v) \ \Rightarrow\ \text{merge } C.
$$

### Regel (CE-R5) – Similar Neighborhood (DP)
Setze $N_u=N(u)\setminus(N(v)\cup\{v\})$, $N_v=N(v)\setminus(N(u)\cup\{u\})$, $W=V\setminus(N_u\cup N_v\cup\{u,v\})$ und  
$\Delta_u=s(u,N_u)-s(u,N_v)$, $\Delta_v=s(v,N_v)-s(v,N_u)$. Dann gilt mit DP‑Schranke:

**(11)**
$$
 s(u,v) \ \ge\ \max_{\substack{C_u,C_v\subseteq W\\ C_u\cap C_v=\varnothing}} \min\Big\{\, s(v,C_v)-s(v,C_u)+\Delta_v,\ \ s(u,C_u)-s(u,C_v)+\Delta_u \,\Big\}
 \ \Rightarrow\ \text{merge } u,v.
$$

**Vorstufe („Regel 0“) – Critical Cliques.** Kontrahiere Knoten mit identischer *closed neighborhood* (Standard‑Kernelisierungsschritt).

## Chalupa‑Heuristik (VCC)

**Ziel.** VCC sucht die kleinste Anzahl Cliquen zur Überdeckung aller Knoten. Implementiert ist eine **Iterated‑Greedy (IG)**‑Konstruktion für eine obere Schranke $\mathrm{UB}$ und eine **Greedy/RLS**‑Suche für eine untere Schranke $\mathrm{LB}$ (max. unabhängige Menge).

### Formale Bausteine
- Äquivalenz: **(1)** $\theta(G)=\chi(\bar{G})$.
- UB via *Greedy Clique Covering* aus einer Permutation $\pi$:
  **(12)**  Erzeuge Cliquen $C_1,C_2,\dots$ iterativ; füge $v$ in die **kleinste zulässige** vorhandene Clique (First‑Fit), sonst starte neue.  
  Zulässigkeitstest in $G$: $v$ ist mit **allen** Mitgliedern von $C_j$ adjazent.
- LB via *Greedy MIS*:
  **(13)**  Durchlaufe $\pi$; wenn $v$ zu bislang gewähltem $I$ **keine** Kante hat, dann $I\leftarrow I\cup\{v\}$.
  Danach **RLS/IG‑Verbesserungen** durch Permutations‑Jumps/Neustarts.

---

## 9. Zusammenfassung

- Das Softwarepaket ermöglicht sowohl heuristische als auch exakte Berechnungen von $\theta(G)$ sowie Cluster Editing mittels Kernelization und ILP.
- Wrapper und CLI-Skripte erlauben eine einheitliche Ausführung.
- Parser und Plot-Skripte unterstützen Auswertung und Visualisierung.
- Testgraph-Generator und Unit-Tests erleichtern Reproduktion und Validierung.

---
## Literatur
- Chalupa, D. (2016). Construction of Near‑Optimal Vertex Clique Covering for Real‑World Networks.
- Böcker, S., Briesemeister, S., & Klau, G.W. (2011). Exact Algorithms for Cluster Editing.
- Grötschel, M., & Wakabayashi, Y. (1989). A Cutting Plane Algorithm for a Clustering Problem.
- Bansal, N., Blum, A., & Chawla, S. (2004). Correlation Clustering.
- Mutzel, P. (2022). Graph Coloring: ILP Formulations.
