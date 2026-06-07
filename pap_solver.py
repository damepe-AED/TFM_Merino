"""
Perfect Awareness Problem (PAP) Solver
=======================================
GRASP constructive phase + Simulated Annealing improvement
on Barabási-Albert (BA) networks.

Reference:
  Pereira et al. "Effective Heuristics for the Perfect Awareness Problem"
  Procedia Computer Science 195 (2021) 489–498.

Author: Daniela Meriño Pérez  (TFM – Máster en Ciencia de Datos, UV)
"""

import math
import random
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np


# ──────────────────────────────────────────────────────────────
# 1.  GRAPH GENERATION
# ──────────────────────────────────────────────────────────────

def generate_ba_graph(n: int, m: int, seed: Optional[int] = None) -> nx.Graph:
    """
    Generate a connected Barabási-Albert graph with n nodes and
    attachment parameter m (each new node connects to m existing nodes).

    The majority threshold function t(v) = ceil(0.5 * deg(v)) is
    assigned as a node attribute, matching the PAP benchmark convention
    (Pereira et al., 2021).
    """
    G = nx.barabasi_albert_graph(n, m, seed=seed)
    for v in G.nodes():
        G.nodes[v]["threshold"] = max(1, math.ceil(0.5 * G.degree(v)))
    return G


# ──────────────────────────────────────────────────────────────
# 2.  DIFFUSION PROCESS
# ──────────────────────────────────────────────────────────────

# Node states
IGNORANT  = 0
AWARE     = 1
SPREADER  = 2


def spreading_process(G: nx.Graph,
                      seed_set: Set[int],
                      nd_init: Optional[Dict[int, int]] = None,
                      state_init: Optional[Dict[int, int]] = None
                      ) -> Tuple[Set[int], Set[int]]:
    """
    Simulate the full diffusion from seed_set (or resume from a partial
    state).  Implements Algorithm 1 of Pereira et al. (2021).

    Returns
    -------
    aware_set    : set of nodes that are aware at the end
    spreader_set : set of nodes that are spreaders at the end
    """
    if state_init is None:
        state  = {v: IGNORANT for v in G.nodes()}
        nd     = {v: 0        for v in G.nodes()}
        queue: List[int] = []
        for v in seed_set:
            state[v] = SPREADER
            queue.append(v)
    else:
        state  = dict(state_init)
        nd     = dict(nd_init)  # type: ignore[arg-type]
        queue  = [v for v in seed_set if state[v] == SPREADER]

    # BFS propagation
    head = 0
    while head < len(queue):
        v = queue[head]; head += 1
        for u in G.neighbors(v):
            nd[u] += 1
            if state[u] == IGNORANT:
                state[u] = AWARE
            if state[u] != SPREADER and nd[u] >= G.nodes[u]["threshold"]:
                state[u] = SPREADER
                queue.append(u)

    aware_set    = {v for v in G.nodes() if state[v] >= AWARE}
    spreader_set = {v for v in G.nodes() if state[v] == SPREADER}
    return aware_set, spreader_set


def is_perfect_seed(G: nx.Graph, seed_set: Set[int]) -> bool:
    """Return True iff seed_set is a perfect seed set (all nodes become aware)."""
    aware, _ = spreading_process(G, seed_set)
    return len(aware) == G.number_of_nodes()


# ──────────────────────────────────────────────────────────────
# 3.  CENTRALITY MEASURES  (computed once per graph)
# ──────────────────────────────────────────────────────────────

@dataclass
class CentralityCache:
    degree:      Dict[int, float]
    eigenvector: Dict[int, float]
    betweenness: Dict[int, float]

    @classmethod
    def compute(cls, G: nx.Graph) -> "CentralityCache":
        deg = nx.degree_centrality(G)
        try:
            eig = nx.eigenvector_centrality(G, max_iter=500)
        except nx.PowerIterationFailedConvergence:
            eig = deg  # fallback
        bet = nx.betweenness_centrality(G, normalized=True)
        return cls(degree=deg, eigenvector=eig, betweenness=bet)


# ──────────────────────────────────────────────────────────────
# 4.  HEURISTIC BENEFIT FUNCTION  g(v)
# ──────────────────────────────────────────────────────────────

def benefit(v: int,
            G: nx.Graph,
            aware_set: Set[int],
            cent: CentralityCache,
            ad: float = 1.0,
            ae: float = 1.0,
            ab: float = 1.0) -> float:
    """
    Combined benefit of adding node v to the current partial seed set.

    g(v) = (ad*C_D + ae*C_E + ab*C_B) / (ad+ae+ab)
           * (deg(v) - |aware_neighbours(v)|) / deg(v)

    The first factor captures the topological importance of v;
    the second factor captures how many new nodes v could activate
    that are not yet aware.
    """
    deg = G.degree(v)
    if deg == 0:
        return 0.0
    aware_neighbours = sum(1 for u in G.neighbors(v) if u in aware_set)
    topological = (ad * cent.degree[v]
                   + ae * cent.eigenvector[v]
                   + ab * cent.betweenness[v]) / (ad + ae + ab)
    coverage = (deg - aware_neighbours) / deg
    return topological * coverage


# ──────────────────────────────────────────────────────────────
# 5.  REFINEMENT  (remove redundant seeds)
# ──────────────────────────────────────────────────────────────

def refine(G: nx.Graph, seed_set: Set[int]) -> Set[int]:
    """
    Iteratively remove seeds whose absence still leaves a perfect seed set.
    Seeds with fewer 'dependent' aware-only neighbours are tried first.
    """
    refined = set(seed_set)
    # Sort candidates: try to remove seeds that are least critical first
    candidates = sorted(
        refined,
        key=lambda v: sum(1 for u in G.neighbors(v)
                          if u not in refined)
    )
    for v in candidates:
        trial = refined - {v}
        if trial and is_perfect_seed(G, trial):
            refined = trial
    return refined


# ──────────────────────────────────────────────────────────────
# 6.  GRASP CONSTRUCTIVE PHASE
# ──────────────────────────────────────────────────────────────

def grasp_construct(G: nx.Graph,
                    cent: CentralityCache,
                    alpha: float = 0.3,
                    ad: float = 1.0,
                    ae: float = 1.0,
                    ab: float = 1.0,
                    rng: Optional[random.Random] = None) -> Set[int]:
    """
    GRASP construction phase (Algorithm 2 of the thesis).

    Parameters
    ----------
    alpha : greediness parameter in [0, 1].
            alpha=0 → fully random; alpha=1 → fully greedy.
    ad, ae, ab : weights for degree, eigenvector and betweenness centrality.
    rng : optional seeded Random instance for reproducibility.
    """
    if rng is None:
        rng = random.Random()

    n = G.number_of_nodes()
    state = {v: IGNORANT for v in G.nodes()}
    nd    = {v: 0        for v in G.nodes()}
    seed_set: Set[int]  = set()
    aware_set: Set[int] = set()
    queue: List[int]    = []

    def propagate_from(v: int) -> None:
        """Incremental propagation after adding v as a spreader."""
        state[v] = SPREADER
        aware_set.add(v)
        local_q = [v]
        head = 0
        while head < len(local_q):
            u = local_q[head]; head += 1
            for w in G.neighbors(u):
                nd[w] += 1
                if state[w] == IGNORANT:
                    state[w] = AWARE
                    aware_set.add(w)
                if state[w] != SPREADER and nd[w] >= G.nodes[w]["threshold"]:
                    state[w] = SPREADER
                    local_q.append(w)

    # Candidate list: nodes not yet spreaders
    cl = {v for v in G.nodes() if state[v] != SPREADER}

    while len(aware_set) < n:
        # Compute benefits for all candidates
        benefits = {v: benefit(v, G, aware_set, cent, ad, ae, ab)
                    for v in cl if state[v] != SPREADER}
        if not benefits:
            break

        g_min = min(benefits.values())
        g_max = max(benefits.values())
        threshold = g_min + alpha * (g_max - g_min)
        rcl = [v for v, g in benefits.items() if g >= threshold]

        if not rcl:
            rcl = list(benefits.keys())

        v_star = rng.choice(rcl)
        seed_set.add(v_star)
        propagate_from(v_star)
        cl.discard(v_star)

    return seed_set


# ──────────────────────────────────────────────────────────────
# 7.  ENERGY FUNCTION  (for Simulated Annealing)
# ──────────────────────────────────────────────────────────────

def energy(G: nx.Graph, seed_set: Set[int], lam: float = 2.0) -> float:
    """
    E(S0) = |S0| + λ * (|V| - |A(S0)|)

    Perfect solutions have E = |S0|; infeasible ones are penalised
    proportionally to the number of unaware nodes.
    """
    aware, _ = spreading_process(G, seed_set)
    return len(seed_set) + lam * (G.number_of_nodes() - len(aware))


# ──────────────────────────────────────────────────────────────
# 8.  SIMULATED ANNEALING IMPROVEMENT
# ──────────────────────────────────────────────────────────────

def simulated_annealing(
                            G: nx.Graph,
                            initial_seed: Set[int],
                            T_init: float = 5.0,
                            T_min: float = 0.01,
                            cooling: float = 0.97,
                            steps_per_temp: int = 30,
                            lam: float = 2.0,
                            rng: Optional[random.Random] = None,
                            apply_final_refinement: bool = True,
) -> Tuple[Set[int], List[float]]:
    """
    Simulated Annealing improvement of a GRASP solution.

    Neighbourhood moves
    -------------------
    - ADD    : add a random non-seed node
    - REMOVE : remove a random seed node
    - SWAP   : replace one seed node with a non-seed node

    Acceptance criterion  (Metropolis):
      if ΔE ≤ 0  → accept always
      else       → accept with probability exp(−ΔE / T)

    Returns
    -------
    best_seed : best (feasible) seed set found
    history   : list of energy values per temperature step (for plotting)
    """
    if rng is None:
        rng = random.Random()

    nodes = list(G.nodes())
    current  = set(initial_seed)
    best     = set(initial_seed)
    E_curr   = energy(G, current, lam)
    E_best   = E_curr
    T        = T_init
    history: List[float] = []

    while T > T_min:
        for _ in range(steps_per_temp):
            non_seeds = [v for v in nodes if v not in current]

            # Choose a random move type
            move = rng.choice(["add", "remove", "swap"])

            if move == "add" and non_seeds:
                v = rng.choice(non_seeds)
                candidate = current | {v}
            elif move == "remove" and len(current) > 1:
                v = rng.choice(list(current))
                candidate = current - {v}
            elif non_seeds and current:
                v_in  = rng.choice(list(current))
                v_out = rng.choice(non_seeds)
                candidate = (current - {v_in}) | {v_out}
            else:
                continue

            E_cand = energy(G, candidate, lam)
            delta  = E_cand - E_curr

            if delta <= 0 or rng.random() < math.exp(-delta / T):
                current = candidate
                E_curr  = E_cand

            if is_perfect_seed(G, current) and len(current) < len(best):
                best   = set(current)
                E_best = E_curr

        history.append(E_curr)
        T *= cooling

    # Final refinement pass on the best found solution
    if apply_final_refinement and is_perfect_seed(G, best):
        best = refine(G, best)

    return best, history


# ──────────────────────────────────────────────────────────────
# 9.  FULL GRASP + SA PIPELINE
# ──────────────────────────────────────────────────────────────

@dataclass
class PAPResult:
    seed_set:        Set[int]
    seed_size:       int
    is_perfect:      bool
    time_grasp:      float          # seconds
    time_sa:         float          # seconds
    time_total:      float          # seconds
    sa_history:      List[float]    = field(default_factory=list)
    graph_n:         int            = 0
    graph_m:         int            = 0
    graph_edges:     int            = 0
    avg_degree:      float          = 0.0
    density:         float          = 0.0


def solve_pap(G: nx.Graph,
              n_grasp_iter: int = 10,
              alpha: float = 0.3,
              ad: float = 1.0,
              ae: float = 1.0,
              ab: float = 1.0,
              T_init: float = 5.0,
              T_min: float = 0.01,
              cooling: float = 0.97,
              steps_per_temp: int = 30,
              lam: float = 2.0,
              seed: Optional[int] = None) -> PAPResult:
    """
    Full pipeline: GRASP (multi-start) → best solution → SA improvement.
    """
    rng  = random.Random(seed)
    cent = CentralityCache.compute(G)
    n    = G.number_of_nodes()

    # ── GRASP phase ──────────────────────────────────────────
    t0 = time.perf_counter()
    best_grasp: Set[int] = set(G.nodes())   # worst-case initialisation
    for _ in range(n_grasp_iter):
        sol = grasp_construct(G, cent, alpha=alpha, ad=ad, ae=ae, ab=ab, rng=rng)
        sol = refine(G, sol)
        if len(sol) < len(best_grasp):
            best_grasp = sol
    t_grasp = time.perf_counter() - t0

    # ── SA phase ─────────────────────────────────────────────
    t1 = time.perf_counter()
    best_sa, history = simulated_annealing(
        G, best_grasp,
        T_init=T_init, T_min=T_min, cooling=cooling,
        steps_per_temp=steps_per_temp, lam=lam, rng=rng
    )
    t_sa = time.perf_counter() - t1

    perfect = is_perfect_seed(G, best_sa)
    avg_deg = sum(d for _, d in G.degree()) / n

    return PAPResult(
        seed_set    = best_sa,
        seed_size   = len(best_sa),
        is_perfect  = perfect,
        time_grasp  = t_grasp,
        time_sa     = t_sa,
        time_total  = t_grasp + t_sa,
        sa_history  = history,
        graph_n     = n,
        graph_m     = G.number_of_edges(),
        graph_edges = G.number_of_edges(),
        avg_degree  = avg_deg,
        density     = nx.density(G),
    )
