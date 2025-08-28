# src/algorithms/cluster_editing_kernelization.py
import logging
import time
import hashlib
import pickle
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Tuple, Set, List, Optional, FrozenSet, Any

import networkx as nx
import numpy as np

from src.utils import txt_to_networkx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Endliches "verboten"-Gewicht (statt -inf, damit ILP stabil bleibt)
FORBIDDEN_WEIGHT = -1e6


class ReductionRule(Enum):
    CRITICAL_CLIQUE = "critical_clique"
    HEAVY_NON_EDGE = "heavy_non_edge"
    HEAVY_EDGE_SINGLE = "heavy_edge_single"
    HEAVY_EDGE_BOTH = "heavy_edge_both"
    ALMOST_CLIQUE = "almost_clique"
    SIMILAR_NEIGHBORHOOD = "similar_neighborhood"


class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}
        self.size = {}

    def make_set(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            self.size[x] = 1

    def find(self, x):
        if x not in self.parent:
            self.make_set(x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
            self.size[ry] += self.size[rx]
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
            self.size[rx] += self.size[ry]
        else:
            self.parent[ry] = rx
            self.size[rx] += self.size[ry]
            self.rank[rx] += 1
        return True

    def get_components(self) -> List[Set]:
        comps = defaultdict(set)
        for x in self.parent:
            comps[self.find(x)].add(x)
        return list(comps.values())


@dataclass
class KernelizationCache:
    _cache: Dict[str, Any] = field(default_factory=dict)
    _hits: int = field(default=0)
    _misses: int = field(default=0)
    _max_size: int = field(default=10000)

    def _make_key(self, *args) -> str:
        return hashlib.md5(pickle.dumps(args)).hexdigest()

    def get(self, *args):
        key = self._make_key(*args)
        if key in self._cache:
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def set(self, value, *args):
        if len(self._cache) >= self._max_size:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        key = self._make_key(*args)
        self._cache[key] = value

    def get_stats(self) -> Dict[str, Any]:
        total = self._hits + self._misses
        return {
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': self._hits / total if total > 0 else 0,
            'size': len(self._cache)
        }


@dataclass
class RuleEffectiveness:
    rule_stats: Dict[ReductionRule, Dict[str, float]] = field(default_factory=dict)

    def __post_init__(self):
        for rule in ReductionRule:
            self.rule_stats[rule] = {
                'applications': 0,
                'reductions': 0,
                'time_spent': 0.0,
                'effectiveness': 1.0
            }

    def update(self, rule: ReductionRule, applied: bool, reduction: int, time_spent: float):
        stats = self.rule_stats[rule]
        if applied:
            stats['applications'] += 1
        stats['reductions'] += reduction
        stats['time_spent'] += time_spent
        if stats['time_spent'] > 0:
            stats['effectiveness'] = stats['reductions'] / stats['time_spent']

    def get_rule_order(self) -> List[ReductionRule]:
        return sorted(self.rule_stats.keys(),
                      key=lambda r: self.rule_stats[r]['effectiveness'],
                      reverse=True)


class AdvancedClusterEditingInstance:
    def __init__(self, graph: nx.Graph, weights: Optional[Dict] = None, k: Optional[float] = None):
        self.graph = graph
        self.weights = weights or {}
        self.k = k

        self.union_find = UnionFind()
        self.cache = KernelizationCache()

        self._use_sparse = graph.number_of_nodes() > 1000
        if self._use_sparse:
            self._init_sparse_representation()

        self._delta_queue = deque(maxlen=100)
        self._checkpoint_interval = 10
        self._operations_since_checkpoint = 0

        self._neighbor_sums = {}
        self._weight_index = defaultdict(list)
        self._init_structures()

        # Mapping: aktueller Knoten (Supernode) -> wirft Menge Originalknoten
        self.supernode_members: Dict[int, Set[int]] = {v: {v} for v in graph.nodes()}

    def _init_sparse_representation(self):
        from scipy.sparse import lil_matrix
        n = self.graph.number_of_nodes()
        self._node_list = list(self.graph.nodes())
        self._node_to_idx = {node: i for i, node in enumerate(self._node_list)}
        self._sparse_weights = lil_matrix((n, n))
        for (u, v), w in self.weights.items():
            if u in self._node_to_idx and v in self._node_to_idx:
                i, j = self._node_to_idx[u], self._node_to_idx[v]
                self._sparse_weights[i, j] = w
                self._sparse_weights[j, i] = w

    def _init_structures(self):
        for node in self.graph.nodes():
            self.union_find.make_set(node)
        for u in self.graph.nodes():
            self._neighbor_sums[u] = sum(
                self.get_weight(u, v) for v in self.graph.neighbors(u)
            )
        for edge, weight in self.weights.items():
            bucket = int(weight / 10)
            self._weight_index[bucket].append(edge)

    def get_weight(self, u: int, v: int) -> float:
        if u == v:
            return 0.0
        cached = self.cache.get('weight', u, v)
        if cached is not None:
            return cached
        if self._use_sparse and hasattr(self, '_sparse_weights'):
            if u in self._node_to_idx and v in self._node_to_idx:
                i, j = self._node_to_idx[u], self._node_to_idx[v]
                weight = float(self._sparse_weights[i, j])
                self.cache.set(weight, 'weight', u, v)
                return weight
        weight = float(self.weights.get((min(u, v), max(u, v)), 0.0))
        self.cache.set(weight, 'weight', u, v)
        return weight

    def create_checkpoint(self):
        self._checkpoint = {
            'graph': self.graph.copy(),
            'weights': self.weights.copy(),
            'neighbor_sums': self._neighbor_sums.copy()
        }
        self._operations_since_checkpoint = 0

    def rollback_to_checkpoint(self):
        if hasattr(self, '_checkpoint'):
            self.graph = self._checkpoint['graph']
            self.weights = self._checkpoint['weights']
            self._neighbor_sums = self._checkpoint['neighbor_sums']
            self.cache._cache.clear()

    def get_neighbor_sum(self, u: int) -> float:
        return float(self._neighbor_sums.get(u, 0.0))


class AdvancedKernelization:
    def __init__(self, instance: AdvancedClusterEditingInstance,
                 use_parallel: bool = True,
                 use_preprocessing: bool = True,
                 use_smart_ordering: bool = True):
        self.instance = instance
        self.use_parallel = use_parallel and instance.graph.number_of_nodes() > 100
        self.use_preprocessing = use_preprocessing
        self.use_smart_ordering = use_smart_ordering

        self.rule_effectiveness = RuleEffectiveness()
        self.stats = defaultdict(int)
        self.reduction_history = []

        self.executor = None  # Threadpool optional

        self.merge_queue: List[Tuple[int, int]] = []
        self.forbid_queue: List[Tuple[int, int]] = []

    def preprocess(self) -> bool:
        if not self.use_preprocessing:
            return False

        reduced = False

        components = list(nx.connected_components(self.instance.graph))
        for component in components:
            if len(component) <= 20:
                subgraph = self.instance.graph.subgraph(component)
                if self._is_clique(subgraph):
                    for node in list(component):
                        self.instance.graph.remove_node(node)
                    reduced = True

        if self.instance.weights:
            wvals = list(self.instance.weights.values())
            if len(wvals) >= 2:
                heavy_threshold = float(np.percentile(wvals, 95))
                light_threshold = float(np.percentile(wvals, 5))
            else:
                heavy_threshold = 1.0
                light_threshold = -1.0

            for (u, v), weight in self.instance.weights.items():
                if weight > heavy_threshold:
                    nu = set(self.instance.graph.neighbors(u))
                    nv = set(self.instance.graph.neighbors(v))
                    if len(nu | nv) > 0 and len(nu & nv) / float(len(nu | nv)) > 0.8:
                        self.merge_queue.append((u, v))
                        reduced = True

            for (u, v), weight in self.instance.weights.items():
                if weight < light_threshold:
                    self.forbid_queue.append((u, v))
                    reduced = True

        return reduced

    def _is_clique(self, subgraph) -> bool:
        n = subgraph.number_of_nodes()
        m = subgraph.number_of_edges()
        return m == n * (n - 1) // 2

    def apply_rules_smart_order(self) -> bool:
        if not self.use_smart_ordering:
            return self.apply_rules_standard_order()

        applied = False
        for rule in self.rule_effectiveness.get_rule_order():
            start = time.time()
            before = self.instance.graph.number_of_nodes()

            ok = False
            if rule == ReductionRule.CRITICAL_CLIQUE:
                ok = self.apply_critical_cliques_batch()
            elif rule == ReductionRule.HEAVY_NON_EDGE:
                ok = self.apply_heavy_non_edges_batch()
            elif rule == ReductionRule.HEAVY_EDGE_SINGLE:
                ok = self.apply_heavy_edge_single_batch()
            elif rule == ReductionRule.HEAVY_EDGE_BOTH:
                ok = self.apply_heavy_edge_both_batch()
            elif rule == ReductionRule.ALMOST_CLIQUE:
                ok = self.apply_almost_clique_advanced()
            elif rule == ReductionRule.SIMILAR_NEIGHBORHOOD:
                ok = self.apply_similar_neighborhood_advanced()

            spent = time.time() - start
            reduction = before - self.instance.graph.number_of_nodes()
            self.rule_effectiveness.update(rule, ok, reduction, spent)
            applied = applied or ok

        return applied

    def apply_rules_standard_order(self) -> bool:
        applied = False
        if self.apply_critical_cliques_batch():
            applied = True
        if self.apply_heavy_non_edges_batch():
            applied = True
        if self.apply_heavy_edge_single_batch():
            applied = True
        if self.apply_heavy_edge_both_batch():
            applied = True
        return applied

    def apply_critical_cliques_batch(self) -> bool:
        crits = self._find_critical_cliques()
        if not crits:
            return False
        for clique in crits:
            if len(clique) > 1:
                base = next(iter(clique))
                for v in clique:
                    if v != base:
                        self.instance.union_find.union(base, v)
                        self.merge_queue.append((base, v))
        self._apply_merge_batch()
        return True

    def _find_critical_cliques(self) -> List[Set[int]]:
        cached = self.instance.cache.get('critical_cliques', self.instance.graph)
        if cached is not None:
            return cached

        groups = defaultdict(set)
        for v in self.instance.graph.nodes():
            closed = frozenset(self.instance.graph.neighbors(v)) | {v}
            groups[closed].add(v)

        crits: List[Set[int]] = []
        for verts in groups.values():
            if len(verts) > 1:
                sub = self.instance.graph.subgraph(verts)
                exp_deg = len(verts) - 1
                if all(sub.degree(x) == exp_deg for x in verts):
                    crits.append(set(verts))

        self.instance.cache.set(crits, 'critical_cliques', self.instance.graph)
        return crits

    def apply_heavy_non_edges_batch(self) -> bool:
        for u in self.instance.graph.nodes():
            u_sum = float(self.instance._neighbor_sums.get(u, 0.0))
            for v in self.instance.graph.nodes():
                if u < v:
                    w = self.instance.get_weight(u, v)
                    if w < 0 and abs(w) >= u_sum:
                        self.forbid_queue.append((u, v))
        if self.forbid_queue:
            self._apply_forbid_batch()
            return True
        return False

    def apply_heavy_edge_single_batch(self) -> bool:
        for u, v in list(self.instance.graph.edges()):
            w = self.instance.get_weight(u, v)
            if w > 0:
                total = 0.0
                for wnb in self.instance.graph.nodes():
                    if wnb != u and wnb != v:
                        total += abs(self.instance.get_weight(u, wnb))
                if w >= total:
                    self.merge_queue.append((u, v))
        if self.merge_queue:
            self._apply_merge_batch()
            return True
        return False

    def apply_heavy_edge_both_batch(self) -> bool:
        for u, v in list(self.instance.graph.edges()):
            w = self.instance.get_weight(u, v)
            if w > 0:
                su = sum(self.instance.get_weight(u, wnb) for wnb in self.instance.graph.neighbors(u) if wnb != v)
                sv = sum(self.instance.get_weight(v, wnb) for wnb in self.instance.graph.neighbors(v) if wnb != u)
                if w >= su + sv:
                    self.merge_queue.append((u, v))
        if self.merge_queue:
            self._apply_merge_batch()
            return True
        return False

    def apply_almost_clique_advanced(self) -> bool:
        if self.instance.graph.number_of_nodes() < 10:
            return False
        try:
            lap = nx.laplacian_matrix(self.instance.graph).todense()
            eigvals, eigvecs = np.linalg.eigh(lap)
            fiedler = eigvecs[:, 1]
            med = float(np.median(fiedler))
            nodes = list(self.instance.graph.nodes())
            part1 = [nodes[i] for i, val in enumerate(fiedler) if val < med]
            part2 = [nodes[i] for i, val in enumerate(fiedler) if val >= med]

            for part in (part1, part2):
                if 3 <= len(part) <= 20 and self._check_almost_clique_fast(part):
                    base = part[0]
                    for v in part[1:]:
                        self.merge_queue.append((base, v))

            if self.merge_queue:
                self._apply_merge_batch()
                return True
        except Exception as e:
            logger.debug(f"Almost clique rule failed: {e}")
        return False

    def _check_almost_clique_fast(self, nodes: List[int]) -> bool:
        sub = self.instance.graph.subgraph(nodes)
        n = sub.number_of_nodes()
        m = sub.number_of_edges()
        density = 2 * m / (n * (n - 1)) if n > 1 else 0.0
        if density > 0.8:
            return True
        edge_cost = 0.0
        for i, u in enumerate(nodes):
            for v in nodes[i+1:]:
                if not sub.has_edge(u, v):
                    edge_cost += abs(self.instance.get_weight(u, v))
        cut_cost = 0.0
        others = set(self.instance.graph.nodes()) - set(nodes)
        for u in nodes:
            for v in others:
                if self.instance.graph.has_edge(u, v):
                    cut_cost += abs(self.instance.get_weight(u, v))
        return edge_cost < 0.5 * cut_cost

    #merge_queue wird in mehreren Regeln gefüllt (heavy_edge_*, almost_clique, similar_neighboorhood)
    #Union-Find liefert die Komponenten, die dann in _apply_merge_batch einmalig zusammengeführt werden
    def _apply_merge_batch(self):
        if not self.merge_queue:
            return

        # Alle Paare aus der Queue zunächst im Union-Find vereinigen
        for u, v in self.merge_queue:
            self.instance.union_find.union(u, v)

        # Dann Komponenten bestimmen und je Komponente zu einem Basis-Knoten mergen
        components = self.instance.union_find.get_components()
        for comp in components:
            if len(comp) > 1:
                base = next(iter(comp))
                for v in list(comp):
                    if v != base and v in self.instance.graph:
                        self._merge_vertices(base, v)
        self.merge_queue.clear()

        #cache clearen, um Methode robust zu machen: damit funktionieren auch merges aus preprocess() oder anderen Regeln, auch wenn dort kein union() aufgerufen wurde
        self.instance.cache._cache.clear()


    def _apply_forbid_batch(self):
        if not self.forbid_queue:
            return
        G = self.instance.graph
        changed_nodes = set()
        for u, v in self.forbid_queue:
            if G.has_edge(u, v):
                G.remove_edge(u, v)
                changed_nodes.add(u)
                changed_nodes.add(v)
            self.instance.weights[(min(u, v), max(u, v))] = FORBIDDEN_WEIGHT
            self.stats['forbidden_edges'] += 1

        # neighbor_sums für betroffene Knoten und deren Nachbarn neu berechnen
        for x in list(changed_nodes):
            # x selbst
            self.instance._neighbor_sums[x] = sum(
                self.instance.get_weight(x, w) for w in G.neighbors(x)
            )
            # Nachbarn von x
            for w in G.neighbors(x):
                self.instance._neighbor_sums[w] = sum(
                    self.instance.get_weight(w, z) for z in G.neighbors(w)
                )

        # Cache leeren, weil Gewichte/Graph geändert wurden
        self.instance.cache._cache.clear()
        self.forbid_queue.clear()

    def _merge_vertices(self, u: int, v: int):
        if u == v or v not in self.instance.graph:
            return
        G = self.instance.graph

        for nbr in list(G.neighbors(v)):
            if nbr != u:
                e_un = (min(u, nbr), max(u, nbr))
                e_vn = (min(v, nbr), max(v, nbr))
                new_w = float(self.instance.weights.get(e_un, 0.0))
                new_w += float(self.instance.weights.get(e_vn, 0.0))
                self.instance.weights[e_un] = new_w
                if not G.has_edge(u, nbr):
                    G.add_edge(u, nbr)

        G.remove_node(v)
        # Gewichte des entfernten Knotens säubern
        self.instance.weights = {e: w for e, w in self.instance.weights.items() if v not in e}

        # Mapping der Supernodes zusammenführen lets see
        if u not in self.instance.supernode_members:
            self.instance.supernode_members[u] = {u}
        members_v = self.instance.supernode_members.get(v, {v})
        self.instance.supernode_members[u].update(members_v)
        self.instance.supernode_members.pop(v, None)

        # neighbor_sums für u und seine Nachbarn aktualisieren
        affected = set([u]) | set(G.neighbors(u))
        for x in affected:
            self.instance._neighbor_sums[x] = sum(
                self.instance.get_weight(x, w) for w in G.neighbors(x)
            )

        self.stats['vertices_merged'] += 1
        self.reduction_history.append(('merge', u, v))


    def kernelize(self, max_iterations: int = 100, target_reduction: float = 0.9) -> AdvancedClusterEditingInstance:
        initial = self.instance.graph.number_of_nodes()
        self.stats['initial_nodes'] = int(initial)

        if self.use_preprocessing:
            logger.info("Running preprocessing...")
            self.preprocess()

        iteration = 0
        no_improve = 0

        while iteration < max_iterations:
            prev = self.instance.graph.number_of_nodes()

            if iteration % self.instance._checkpoint_interval == 0:
                self.instance.create_checkpoint()

            applied = self.apply_rules_smart_order() if self.use_smart_ordering else self.apply_rules_standard_order()

            curr = self.instance.graph.number_of_nodes()
            if curr == prev:
                no_improve += 1
                if no_improve >= 3:
                    logger.info(f"Converged after {iteration} iterations")
                    break
            else:
                no_improve = 0

            red = 1.0 - (curr / float(initial))
            if red >= target_reduction:
                logger.info(f"Target reduction {target_reduction:.1%} achieved")
                break

            if not applied:
                break

            iteration += 1

        self.stats['iterations'] = iteration
        self.stats['final_nodes'] = self.instance.graph.number_of_nodes()
        self.stats['final_edges'] = self.instance.graph.number_of_edges()
        return self.instance

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        initial_nodes = self.stats.get('initial_nodes', 0)
        final_nodes = self.instance.graph.number_of_nodes()
        return {
            'reduction_ratio': 1 - (final_nodes / initial_nodes) if initial_nodes > 0 else 0,
            'iterations': self.stats.get('iterations', 0),
            'vertices_merged': self.stats.get('vertices_merged', 0),
            'forbidden_edges': self.stats.get('forbidden_edges', 0),
            'rule_effectiveness': {
                rule.value: stats for rule, stats in self.rule_effectiveness.rule_stats.items()
            },
            'cache_stats': self.instance.cache.get_stats(),
            'final_graph': {
                'nodes': final_nodes,
                'edges': self.instance.graph.number_of_edges(),
                'density': nx.density(self.instance.graph),
                'components': nx.number_connected_components(self.instance.graph)
            }
        }

    def apply_similar_neighborhood_advanced(self,
                                            tau_merge: float = 0.92,
                                            tau_forbid: float = 0.85,
                                            max_pairs_per_round: int = 2000) -> bool:
        """
        Heuristik: Fasse Knoten mit sehr ähnlicher geschlossener Nachbarschaft zusammen
        (Merge) oder verbiete sie (wenn starke negative Gewichtung und nur 'ähnlich').

        - Jaccard(N[u], N[v]) >= tau_merge  -> merge_queue.append((u, v))
        - sonst, wenn Jaccard >= tau_forbid und w(u,v) << 0 -> forbid_queue.append((u, v))

        Rückgabe: True, wenn etwas in Queues gelegt wurde (und anschließend angewandt wird).
        """
        G = self.instance.graph
        if G.number_of_nodes() < 4:
            return False

        added = 0
        degrees = dict(G.degree())
        # Kandidaten: Paare aus lokalen Umfeldern (reduziert O(n^2))
        # Für jeden Knoten nur Nachbarn und 2-Hop-Umfeld prüfen.
        for u in G.nodes():
            Nu_closed = set(G.neighbors(u)) | {u}
            # Grobes Degree-Matching zur Vorselektion
            deg_u = degrees[u]
            # Kandidaten: direkte Nachbarn + Knoten mit ähnlichem Grad (±2) in 2-Hop
            candidates = set(G.neighbors(u))
            for w in list(G.neighbors(u)):
                candidates.update(G.neighbors(w))
            # Entferne u selbst
            candidates.discard(u)

            for v in candidates:
                if u >= v:  # paare nur einmal
                    continue
                # schneller Degree-Check
                if abs(deg_u - degrees[v]) > 2:
                    continue

                Nv_closed = set(G.neighbors(v)) | {v}
                inter = len(Nu_closed & Nv_closed)
                union = len(Nu_closed | Nv_closed)
                if union == 0:
                    continue
                jacc = inter / float(union)

                if jacc >= tau_merge:
                    self.instance.union_find.union(u, v)
                    self.merge_queue.append((u, v))
                    added += 1
                elif jacc >= tau_forbid:
                    w_uv = self.instance.get_weight(u, v)
                    if w_uv < 0 and abs(w_uv) >= 1.0:
                        self.forbid_queue.append((u, v))
                        added += 1

                if added >= max_pairs_per_round:
                    break
            if added >= max_pairs_per_round:
                break

        if self.merge_queue:
            self._apply_merge_batch()
            return True
        if self.forbid_queue:
            self._apply_forbid_batch()
            return True
        return False


def load_instance_from_txt(txt_path):
    txt_path = Path(txt_path)
    G = txt_to_networkx(str(txt_path))
    nodes = list(G.nodes())
    edgeset = set((min(u, v), max(u, v)) for u, v in G.edges())
    weights: Dict[Tuple[int, int], float] = {}
    for i, u in enumerate(nodes):
        for v in nodes[i + 1:]:
            e = (min(u, v), max(u, v))
            weights[e] = 1.0 if e in edgeset else -1.0
    return AdvancedClusterEditingInstance(G, weights)

