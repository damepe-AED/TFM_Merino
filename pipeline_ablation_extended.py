"""
pipeline_ablation_extended.py
=============================
Ablation study for the PAP pipeline on Barabási-Albert graphs.

The script compares six paired variants on the same graph instance:
  1. Random + R
  2. Greedy degree + R
  3. GRASP
  4. GRASP + R
  5. GRASP + R + SA
  6. GRASP + R + SA + R

It preserves compatibility with experiments.py:
- graph seed: 1000 * n + 10 * m + run
- algorithm random stream: random.Random(graph_seed)
- exact centralities: degree, eigenvector and betweenness

The output is written progressively and can be resumed safely.
"""

import csv
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import networkx as nx

from pap_solver import (
    CentralityCache,
    generate_ba_graph,
    grasp_construct,
    is_perfect_seed,
    refine,
    simulated_annealing,
)


# ──────────────────────────────────────────────────────────────
# 1. Experimental configuration
# ──────────────────────────────────────────────────────────────

N_VALUES = [20, 40, 60, 80, 100, 150, 200, 500, 1000, 3000, 5000]
M_VALUES = [1, 2, 3, 4, 5]
N_RUNS = 5

GRASP_ITER = 8
ALPHA = 0.3
AD, AE, AB = 1.0, 1.0, 1.0

T_INIT = 5.0
T_MIN = 0.01
COOLING = 0.97
STEPS_PER_TEMP = 25
LAM = 2.0

OUTPUT_FILE = Path("results_pipeline_ablation_extended.csv")

METHODS = [
    "Random + R",
    "Greedy degree + R",
    "GRASP",
    "GRASP + R",
    "GRASP + R + SA",
    "GRASP + R + SA + R",
]

FIELDNAMES = [
    "n",
    "m",
    "run",
    "method",
    "graph_seed",
    "algorithm_seed",
    "random_baseline_seed",
    "edges",
    "density",
    "seed_size",
    "seed_ratio",
    "is_perfect",
    "time_centrality",
    "time_incremental",
    "time_method_total_excl_centrality",
    "time_method_total_incl_centrality",
]


# ──────────────────────────────────────────────────────────────
# 2. Baseline heuristics
# ──────────────────────────────────────────────────────────────

def greedy_degree(G: nx.Graph) -> Set[int]:
    """Add nodes in decreasing degree order until the seed set is perfect."""
    seed_set: Set[int] = set()

    ordered_nodes = sorted(
        G.nodes(),
        key=lambda node: G.degree(node),
        reverse=True,
    )

    for node in ordered_nodes:
        if is_perfect_seed(G, seed_set):
            break

        seed_set.add(node)

    return seed_set


def greedy_random(G: nx.Graph, rng: random.Random) -> Set[int]:
    """Add nodes in a random order until the seed set is perfect."""
    seed_set: Set[int] = set()
    ordered_nodes = list(G.nodes())
    rng.shuffle(ordered_nodes)

    for node in ordered_nodes:
        if is_perfect_seed(G, seed_set):
            break

        seed_set.add(node)

    return seed_set


def best_by_size(seed_sets: List[Set[int]]) -> Set[int]:
    """Return a copy of the smallest seed set."""
    if not seed_sets:
        raise ValueError("No seed sets were provided.")

    return set(min(seed_sets, key=len))


# ──────────────────────────────────────────────────────────────
# 3. Row helper
# ──────────────────────────────────────────────────────────────

def build_row(
    *,
    n: int,
    m: int,
    run: int,
    method: str,
    graph_seed: int,
    algorithm_seed: int,
    random_baseline_seed: int,
    G: nx.Graph,
    solution: Set[int],
    time_centrality: float,
    time_incremental: float,
    time_method_total_excl_centrality: float,
    time_method_total_incl_centrality: float,
) -> Dict[str, object]:
    """Build one CSV row for a method and one graph instance."""
    return {
        "n": n,
        "m": m,
        "run": run,
        "method": method,
        "graph_seed": graph_seed,
        "algorithm_seed": algorithm_seed,
        "random_baseline_seed": random_baseline_seed,
        "edges": G.number_of_edges(),
        "density": round(nx.density(G), 8),
        "seed_size": len(solution),
        "seed_ratio": len(solution) / n,
        "is_perfect": int(is_perfect_seed(G, solution)),
        "time_centrality": round(time_centrality, 6),
        "time_incremental": round(time_incremental, 6),
        "time_method_total_excl_centrality": round(time_method_total_excl_centrality, 6),
        "time_method_total_incl_centrality": round(time_method_total_incl_centrality, 6),
    }


# ──────────────────────────────────────────────────────────────
# 4. Evaluate all paired variants on one graph
# ──────────────────────────────────────────────────────────────

def evaluate_variants(n: int, m: int, run: int) -> List[Dict[str, object]]:
    """
    Evaluate all variants on the same BA graph.

    The GRASP and SA random stream intentionally matches experiments.py:
      seed = 1000 * n + 10 * m + run
      rng = random.Random(seed)

    This makes the final variant directly comparable with the main experiment.
    """
    base_seed = 1000 * n + 10 * m + run

    graph_seed = base_seed
    algorithm_seed = base_seed
    random_baseline_seed = 700_000 + base_seed

    G = generate_ba_graph(n, m, seed=graph_seed)

    # Exact degree, eigenvector and betweenness centralities.
    # This matches experiments.py through pap_solver.CentralityCache.compute().
    t0 = time.perf_counter()
    centralities = CentralityCache.compute(G)
    time_centrality = time.perf_counter() - t0

    # --------------------------------------------------------
    # Baseline 1: degree greedy + refinement
    # --------------------------------------------------------
    t0 = time.perf_counter()
    degree_solution = refine(G, greedy_degree(G))
    time_degree = time.perf_counter() - t0

    # --------------------------------------------------------
    # Baseline 2: random construction + refinement (multi-start)
    # --------------------------------------------------------
    t0 = time.perf_counter()
    random_solutions: List[Set[int]] = []

    for restart in range(GRASP_ITER):
        rng_random = random.Random(random_baseline_seed + restart)
        solution = greedy_random(G, rng_random)
        random_solutions.append(refine(G, solution))

    random_solution = best_by_size(random_solutions)
    time_random = time.perf_counter() - t0

    # --------------------------------------------------------
    # GRASP stream: same seed and continuous stream as solve_pap
    # --------------------------------------------------------
    algorithm_rng = random.Random(algorithm_seed)

    t0 = time.perf_counter()
    raw_grasp_solutions: List[Set[int]] = []

    for _ in range(GRASP_ITER):
        raw_grasp_solutions.append(
            grasp_construct(
                G,
                centralities,
                alpha=ALPHA,
                ad=AD,
                ae=AE,
                ab=AB,
                rng=algorithm_rng,
            )
        )

    grasp_solution = best_by_size(raw_grasp_solutions)
    time_grasp_construct = time.perf_counter() - t0

    # --------------------------------------------------------
    # Initial refinement
    # --------------------------------------------------------
    t0 = time.perf_counter()
    refined_grasp_solutions = [
        refine(G, solution)
        for solution in raw_grasp_solutions
    ]
    grasp_refined_solution = best_by_size(refined_grasp_solutions)
    time_initial_refinement = time.perf_counter() - t0

    # --------------------------------------------------------
    # SA without final refinement
    # --------------------------------------------------------
    t0 = time.perf_counter()
    sa_solution, _ = simulated_annealing(
        G,
        grasp_refined_solution,
        T_init=T_INIT,
        T_min=T_MIN,
        cooling=COOLING,
        steps_per_temp=STEPS_PER_TEMP,
        lam=LAM,
        rng=algorithm_rng,
        apply_final_refinement=False,
    )
    time_sa = time.perf_counter() - t0

    # --------------------------------------------------------
    # Final refinement
    # --------------------------------------------------------
    t0 = time.perf_counter()
    full_solution = refine(G, sa_solution)
    time_final_refinement = time.perf_counter() - t0

    # Cumulative times for pipeline variants. The centrality preprocessing
    # is included explicitly so end-to-end costs are transparent.
    time_grasp_total = time_centrality + time_grasp_construct
    time_grasp_r_total = time_grasp_total + time_initial_refinement
    time_grasp_r_sa_total = time_grasp_r_total + time_sa
    time_full_total = time_grasp_r_sa_total + time_final_refinement

    rows = [
        build_row(
            n=n,
            m=m,
            run=run,
            method="Random + R",
            graph_seed=graph_seed,
            algorithm_seed=algorithm_seed,
            random_baseline_seed=random_baseline_seed,
            G=G,
            solution=random_solution,
            time_centrality=0.0,
            time_incremental=time_random,
            time_method_total_excl_centrality=time_random,
            time_method_total_incl_centrality=time_random,
        ),
        build_row(
            n=n,
            m=m,
            run=run,
            method="Greedy degree + R",
            graph_seed=graph_seed,
            algorithm_seed=algorithm_seed,
            random_baseline_seed=random_baseline_seed,
            G=G,
            solution=degree_solution,
            time_centrality=0.0,
            time_incremental=time_degree,
            time_method_total_excl_centrality=time_degree,
            time_method_total_incl_centrality=time_degree,
        ),
        build_row(
            n=n,
            m=m,
            run=run,
            method="GRASP",
            graph_seed=graph_seed,
            algorithm_seed=algorithm_seed,
            random_baseline_seed=random_baseline_seed,
            G=G,
            solution=grasp_solution,
            time_centrality=time_centrality,
            time_incremental=time_grasp_construct,
            time_method_total_excl_centrality=time_grasp_construct,
            time_method_total_incl_centrality=time_grasp_total,
        ),
        build_row(
            n=n,
            m=m,
            run=run,
            method="GRASP + R",
            graph_seed=graph_seed,
            algorithm_seed=algorithm_seed,
            random_baseline_seed=random_baseline_seed,
            G=G,
            solution=grasp_refined_solution,
            time_centrality=time_centrality,
            time_incremental=time_initial_refinement,
            time_method_total_excl_centrality=time_grasp_construct + time_initial_refinement,
            time_method_total_incl_centrality=time_grasp_r_total,
        ),
        build_row(
            n=n,
            m=m,
            run=run,
            method="GRASP + R + SA",
            graph_seed=graph_seed,
            algorithm_seed=algorithm_seed,
            random_baseline_seed=random_baseline_seed,
            G=G,
            solution=sa_solution,
            time_centrality=time_centrality,
            time_incremental=time_sa,
            time_method_total_excl_centrality=time_grasp_construct + time_initial_refinement + time_sa,
            time_method_total_incl_centrality=time_grasp_r_sa_total,
        ),
        build_row(
            n=n,
            m=m,
            run=run,
            method="GRASP + R + SA + R",
            graph_seed=graph_seed,
            algorithm_seed=algorithm_seed,
            random_baseline_seed=random_baseline_seed,
            G=G,
            solution=full_solution,
            time_centrality=time_centrality,
            time_incremental=time_final_refinement,
            time_method_total_excl_centrality=time_grasp_construct + time_initial_refinement + time_sa + time_final_refinement,
            time_method_total_incl_centrality=time_full_total,
        ),
    ]

    return rows


# ──────────────────────────────────────────────────────────────
# 5. Safe resume helpers
# ──────────────────────────────────────────────────────────────

def read_complete_previous_rows(
    output_file: Path,
) -> Tuple[List[Dict[str, str]], Set[Tuple[int, int, int]]]:
    """
    Read previous results and retain only complete instance blocks.

    A block is complete only when all six methods are present exactly once.
    Partial rows are discarded before resuming, preventing duplicates.
    """
    if not output_file.exists():
        return [], set()

    with output_file.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames != FIELDNAMES:
            raise ValueError(
                "The existing output CSV has a different schema. "
                "Rename or delete it before running this script."
            )

        previous_rows = list(reader)

    grouped_rows: Dict[Tuple[int, int, int], List[Dict[str, str]]] = defaultdict(list)

    for row in previous_rows:
        key = (
            int(row["n"]),
            int(row["m"]),
            int(row["run"]),
        )
        grouped_rows[key].append(row)

    complete_rows: List[Dict[str, str]] = []
    complete_keys: Set[Tuple[int, int, int]] = set()
    expected_methods = set(METHODS)

    for key, rows in grouped_rows.items():
        method_counts = Counter(row["method"] for row in rows)

        is_complete = (
            set(method_counts) == expected_methods
            and all(method_counts[method] == 1 for method in METHODS)
        )

        if is_complete:
            complete_rows.extend(rows)
            complete_keys.add(key)

    discarded = len(previous_rows) - len(complete_rows)

    if discarded:
        print(
            f"Discarded {discarded} partial or duplicated rows "
            "from the previous CSV."
        )

    return complete_rows, complete_keys


# ──────────────────────────────────────────────────────────────
# 6. Main execution
# ──────────────────────────────────────────────────────────────

def main() -> None:
    total_instances = len(N_VALUES) * len(M_VALUES) * N_RUNS

    previous_rows, completed_instances = read_complete_previous_rows(
        OUTPUT_FILE
    )

    # Rewrite a clean file first, then append new complete blocks.
    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()

        if previous_rows:
            writer.writerows(previous_rows)

        file.flush()

        done = len(completed_instances)

        print("Running extended pipeline ablation study...")
        print(f"  n values          : {N_VALUES}")
        print(f"  m values          : {M_VALUES}")
        print(f"  runs per (n, m)   : {N_RUNS}")
        print(f"  paired instances  : {total_instances}")
        print(f"  completed already : {done}")
        print(f"  output file       : {OUTPUT_FILE.resolve()}\n")

        for n in N_VALUES:
            for m in M_VALUES:
                for run in range(N_RUNS):
                    instance_key = (n, m, run)

                    if instance_key in completed_instances:
                        continue

                    start = time.perf_counter()

                    instance_rows = evaluate_variants(
                        n=n,
                        m=m,
                        run=run,
                    )

                    writer.writerows(instance_rows)
                    file.flush()

                    done += 1
                    elapsed = time.perf_counter() - start

                    print(
                        f"[{done:3d}/{total_instances}] "
                        f"n={n:4d}, m={m}, run={run} completed "
                        f"in {elapsed:.2f}s",
                        flush=True,
                    )

    print(f"\nSaved: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
