# src/algorithms/cluster_editing_kernelization_fixed.py
import logging
import time
import hashlib
import pickle
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Tuple, Set, List, Optional, FrozenSet, Any
import math
from concurrent.futures import ThreadPoolExecutor

import networkx as nx
import numpy as np

from src.utils import txt_to_networkx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Weight convention documentation:
# - Positive weights: edges present in graph (deletion cost)
# - Negative weights: edges absent from graph (insertion cost)
# - FORBIDDEN_WEIGHT: Very negative weight indicating edge must not exist
# - PERMANENT_WEIGHT: Very positive weight indicating edge must exist
FORBIDDEN_WEIGHT = -1e9  # Large negative value to prevent edge creation
PERMANENT_WEIGHT = 1e9  # Large positive value to prevent edge deletion


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
    _max_size: int = field(default=5000)  # Reduced cache size limit

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
        # Implement LRU-style eviction when cache is full
        if len(self._cache) >= self._max_size:
            # Remove 20% of oldest entries
            num_to_remove = self._max_size // 5
            for _ in range(num_to_remove):
                if self._cache:
                    oldest = next(iter(self._cache))
                    del self._cache[oldest]

        key = self._make_key(*args)
        self._cache[key] = value

    def clear(self):
        self._cache.clear()

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
            if math.isfinite(weight):  # Skip infinite weights
                bucket = int(weight / 10)
                self._weight_index[bucket].append(edge)

    def get_weight(self, u: int, v: int) -> float:
        """Get edge weight with validation."""
        if u == v:
            return 0.0

        cached = self.cache.get('weight', u, v)
        if cached is not None:
            return cached

        if self._use_sparse and hasattr(self, '_sparse_weights'):
            if u in self._node_to_idx and v in self._node_to_idx:
                i, j = self._node_to_idx[u], self._node_to_idx[v]
                weight = float(self._sparse_weights[i, j])
        else:
            weight = float(self.weights.get((min(u, v), max(u, v)), 0.0))

        # Validate weight
        if not math.isfinite(weight):
            logger.warning(f"Non-finite weight {weight} for edge ({u},{v}), replacing with 0")
            weight = 0.0

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
            self.cache.clear()

    def get_neighbor_sum(self, u: int) -> float:
        return float(self._neighbor_sums.get(u, 0.0))


class AdvancedKernelization:
    def __init__(self, instance: AdvancedClusterEditingInstance = None,
                 use_parallel: bool = True,
                 use_preprocessing: bool = True,
                 use_smart_ordering: bool = True):
        self.instance = instance
        self.use_parallel = use_parallel and instance is not None and instance.graph.number_of_nodes() > 100 #Nullprüfung required
        self.use_preprocessing = use_preprocessing
        self.use_smart_ordering = use_smart_ordering

        self.rule_effectiveness = RuleEffectiveness()
        self.stats = defaultdict(int)
        self.reduction_history = []

        # Initialize thread pool if parallel processing is enabled
        self.executor = None
        if self.use_parallel:
            self.executor = ThreadPoolExecutor(max_workers=4)

        self.merge_queue: List[Tuple[int, int]] = []
        self.forbid_queue: List[Tuple[int, int]] = []

    def __del__(self):
        if hasattr(self, 'executor') and self.executor:
            self.executor.shutdown(wait=False)

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
                        if node in self.instance.graph:
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

            for (u, v), weight in list(self.instance.weights.items()):
                if weight > heavy_threshold:
                    nu = set(self.instance.graph.neighbors(u)) if u in self.instance.graph else set()
                    nv = set(self.instance.graph.neighbors(v)) if v in self.instance.graph else set()
                    if len(nu | nv) > 0 and len(nu & nv) / float(len(nu | nv)) > 0.8:
                        self.merge_queue.append((u, v))
                        reduced = True

            for (u, v), weight in list(self.instance.weights.items()):
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
                ok = self.apply_almost_clique_mincut()
            elif rule == ReductionRule.SIMILAR_NEIGHBORHOOD:
                ok = self.apply_similar_neighborhood_dp()

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
        added = False
        for u in self.instance.graph.nodes():
            u_sum = float(self.instance._neighbor_sums.get(u, 0.0))
            for v in self.instance.graph.nodes():
                if u < v and not self.instance.graph.has_edge(u, v):
                    w = self.instance.get_weight(u, v)
                    if w < 0 and abs(w) >= u_sum:
                        self.forbid_queue.append((u, v))
                        added = True
        if self.forbid_queue:
            self._apply_forbid_batch()
            return True
        return False

    def apply_heavy_edge_single_batch(self) -> bool:
        added = False
        for u, v in list(self.instance.graph.edges()):
            w = self.instance.get_weight(u, v)
            if w > 0:
                total = 0.0
                for wnb in self.instance.graph.nodes():
                    if wnb != u and wnb != v:
                        total += abs(self.instance.get_weight(u, wnb))
                if w >= total:
                    self.merge_queue.append((u, v))
                    added = True
        if self.merge_queue:
            self._apply_merge_batch()
            return True
        return False

    def apply_heavy_edge_both_batch(self) -> bool:
        added = False
        for u, v in list(self.instance.graph.edges()):
            w = self.instance.get_weight(u, v)
            if w > 0:
                su = sum(self.instance.get_weight(u, wnb)
                         for wnb in self.instance.graph.neighbors(u) if wnb != v)
                sv = sum(self.instance.get_weight(v, wnb)
                         for wnb in self.instance.graph.neighbors(v) if wnb != u)
                if w >= su + sv:
                    self.merge_queue.append((u, v))
                    added = True
        if self.merge_queue:
            self._apply_merge_batch()
            return True
        return False

    def apply_almost_clique_mincut(self) -> bool:
        """Apply Rule 4 using proper min-cut as in paper."""
        if self.instance.graph.number_of_nodes() < 10:
            return False

        applied = False
        # Greedy vertex selection strategy from paper
        nodes = list(self.instance.graph.nodes())

        # Start with vertex maximizing sum of absolute weights
        start_scores = {}
        for u in nodes:
            score = sum(abs(self.instance.get_weight(u, v)) for v in nodes if v != u)
            start_scores[u] = score

        if not start_scores:
            return False

        best_start = max(start_scores, key=start_scores.get)
        C = {best_start}
        remaining = set(nodes) - C

        while remaining:
            # Find vertex with maximal connectivity to C
            best_v = None
            best_connectivity = -float('inf')

            for v in remaining:
                connectivity = sum(self.instance.get_weight(v, u) for u in C)
                if connectivity > best_connectivity:
                    best_connectivity = connectivity
                    best_v = v

            if best_v is None:
                break

            # Check if connectivity to C is greater than to V\C
            external_connectivity = sum(
                self.instance.get_weight(best_v, u)
                for u in remaining if u != best_v
            )

            if best_connectivity <= external_connectivity:
                break  # Stop if vertex is more connected externally

            C.add(best_v)
            remaining.remove(best_v)

            # Try to apply Rule 4 to current set C
            if len(C) >= 3:
                if self._check_almost_clique_rule(C):
                    base = next(iter(C))
                    for v in C:
                        if v != base:
                            self.instance.union_find.union(base, v)
                            self.merge_queue.append((base, v))
                    applied = True
                    break

        if self.merge_queue:
            self._apply_merge_batch()

        return applied

    def _check_almost_clique_rule(self, C: Set[int]) -> bool:
        """Check if Rule 4 condition holds for set C."""
        # Calculate min-cut value using NetworkX
        subgraph = self.instance.graph.subgraph(C)

        # Create weighted graph for min-cut calculation
        weighted_sub = nx.Graph()
        for u in C:
            for v in C:
                if u < v:
                    w = self.instance.get_weight(u, v)
                    if w > 0 and self.instance.graph.has_edge(u, v):
                        weighted_sub.add_edge(u, v, capacity=w)

        if weighted_sub.number_of_edges() == 0:
            return False

        # Calculate min-cut
        try:
            min_cut_value = nx.minimum_cut_value(weighted_sub)
        except:
            return False

        # Calculate thresholds from paper
        edge_cost = sum(
            abs(self.instance.get_weight(u, v))
            for u in C for v in C
            if u < v and not self.instance.graph.has_edge(u, v)
        )

        cut_cost = sum(
            self.instance.get_weight(u, v)
            for u in C
            for v in self.instance.graph.nodes()
            if v not in C and self.instance.graph.has_edge(u, v)
        )

        return min_cut_value >= edge_cost + cut_cost

    def apply_similar_neighborhood_dp(self) -> bool:
        """Apply Rule 5 using dynamic programming as in paper."""
        applied = False
        G = self.instance.graph

        for u, v in list(G.edges()):
            # Define exclusive neighborhoods
            Nu = set(G.neighbors(u)) - set(G.neighbors(v)) - {v}
            Nv = set(G.neighbors(v)) - set(G.neighbors(u)) - {u}
            W = set(G.nodes()) - Nu - Nv - {u, v}

            if not W:
                continue

            # Calculate delta values
            delta_u = sum(self.instance.get_weight(u, w) for w in Nu) - \
                      sum(self.instance.get_weight(u, w) for w in Nv)
            delta_v = sum(self.instance.get_weight(v, w) for w in Nv) - \
                      sum(self.instance.get_weight(v, w) for w in Nu)

            # Build DP table
            B = [(self.instance.get_weight(u, w), self.instance.get_weight(v, w))
                 for w in W]

            # Calculate bound using DP
            bound = self._compute_dp_bound(B, delta_u, delta_v)

            # Check if edge should be merged
            edge_weight = self.instance.get_weight(u, v)
            if edge_weight >= bound:
                self.instance.union_find.union(u, v)
                self.merge_queue.append((u, v))
                applied = True

        if self.merge_queue:
            self._apply_merge_batch()

        return applied

    def _compute_dp_bound(self, B: List[Tuple[float, float]],
                          delta_u: float, delta_v: float) -> float:
        """Compute bound using dynamic programming from paper."""
        if not B:
            return min(delta_u, delta_v)

        # Calculate bounds for DP
        X = sum(abs(x) for x, y in B)
        Y = sum(abs(y) for x, y in B)

        # Initialize DP table
        # M[x] = maximum y such that we can achieve (x, y)
        M_prev = {0: 0}

        for x_val, y_val in B:
            M_curr = {}
            for x, y in M_prev.items():
                # Don't assign to any bucket
                if x not in M_curr or M_curr[x] < y:
                    M_curr[x] = y

                # Assign to B1
                new_x = x + x_val
                new_y = y - y_val
                if new_x not in M_curr or M_curr[new_x] < new_y:
                    M_curr[new_x] = new_y

                # Assign to B2
                new_x = x - x_val
                new_y = y + y_val
                if new_x not in M_curr or M_curr[new_x] < new_y:
                    M_curr[new_x] = new_y

            M_prev = M_curr

        # Find maximum
        max_val = 0
        for x, y in M_prev.items():
            val = min(x + delta_u, y + delta_v)
            if val > max_val:
                max_val = val

        return max_val

    def _apply_merge_batch(self):
        """Apply all pending merges with proper validation."""
        if not self.merge_queue:
            return

        # Get components from union-find
        components = self.instance.union_find.get_components()

        for comp in components:
            if len(comp) > 1:
                # Validate all nodes exist
                valid_nodes = [v for v in comp if v in self.instance.graph]
                if len(valid_nodes) > 1:
                    base = valid_nodes[0]
                    for v in valid_nodes[1:]:
                        self._merge_vertices(base, v)

        self.merge_queue.clear()
        # Clear cache only once after all merges
        self.instance.cache.clear()

    def _apply_forbid_batch(self):
        if not self.forbid_queue:
            return
        G = self.instance.graph
        changed_nodes = set()

        for u, v in self.forbid_queue:
            if u in G and v in G and G.has_edge(u, v):
                G.remove_edge(u, v)
                changed_nodes.add(u)
                changed_nodes.add(v)
            self.instance.weights[(min(u, v), max(u, v))] = FORBIDDEN_WEIGHT
            self.stats['forbidden_edges'] += 1

        # Update neighbor sums for affected nodes
        for x in changed_nodes:
            if x in G:
                self.instance._neighbor_sums[x] = sum(
                    self.instance.get_weight(x, w) for w in G.neighbors(x)
                )
                # Update neighbors of x
                for w in G.neighbors(x):
                    self.instance._neighbor_sums[w] = sum(
                        self.instance.get_weight(w, z) for z in G.neighbors(w)
                    )

        self.instance.cache.clear()
        self.forbid_queue.clear()

    def _merge_vertices(self, u: int, v: int):
        """Merge vertices with validation."""
        if u == v or v not in self.instance.graph or u not in self.instance.graph:
            return

        G = self.instance.graph

        # Merge edges
        for nbr in list(G.neighbors(v)):
            if nbr != u:
                e_un = (min(u, nbr), max(u, nbr))
                e_vn = (min(v, nbr), max(v, nbr))
                new_w = float(self.instance.weights.get(e_un, 0.0))
                new_w += float(self.instance.weights.get(e_vn, 0.0))

                # Validate weight
                if not math.isfinite(new_w):
                    logger.warning(f"Non-finite weight during merge: {new_w}")
                    new_w = 0.0

                self.instance.weights[e_un] = new_w
                if not G.has_edge(u, nbr):
                    G.add_edge(u, nbr)

        G.remove_node(v)

        # Clean up weights
        self.instance.weights = {e: w for e, w in self.instance.weights.items()
                                 if v not in e}

        # Update supernode mapping
        if u not in self.instance.supernode_members:
            self.instance.supernode_members[u] = {u}
        members_v = self.instance.supernode_members.get(v, {v})
        self.instance.supernode_members[u].update(members_v)
        self.instance.supernode_members.pop(v, None)

        # Update neighbor sums
        if u in G:
            self.instance._neighbor_sums[u] = sum(
                self.instance.get_weight(u, w) for w in G.neighbors(u)
            )
            for w in G.neighbors(u):
                self.instance._neighbor_sums[w] = sum(
                    self.instance.get_weight(w, z) for z in G.neighbors(w)
                )

        self.stats['vertices_merged'] += 1
        self.reduction_history.append(('merge', u, v))

    def kernelize(self, max_iterations: int = 100,
                  target_reduction: float = 0.9) -> AdvancedClusterEditingInstance:
        """Main kernelization routine."""
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

            applied = (self.apply_rules_smart_order() if self.use_smart_ordering
                       else self.apply_rules_standard_order())

            curr = self.instance.graph.number_of_nodes()
            if curr == prev:
                no_improve += 1
                if no_improve >= 3:
                    logger.info(f"Converged after {iteration} iterations")
                    break
            else:
                no_improve = 0

            red = 1.0 - (curr / float(initial)) if initial > 0 else 0
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