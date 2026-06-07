import csv
import random
from typing import Set

from pap_solver import (
    CentralityCache,
    generate_ba_graph,
    grasp_construct,
    is_perfect_seed,
    refine,
    simulated_annealing,
)

N_VALUES  = [20, 40, 60, 80, 100, 150, 200, 500, 1000, 3000, 5000]
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


def greedy_degree(G) -> Set[int]:
    """Añade nodos por grado decreciente hasta obtener una semilla perfecta."""
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


def greedy_random(G, rng: random.Random) -> Set[int]:
    """Añade nodos en orden aleatorio hasta obtener una semilla perfecta."""
    seed_set: Set[int] = set()
    ordered_nodes = list(G.nodes())
    rng.shuffle(ordered_nodes)

    for node in ordered_nodes:
        if is_perfect_seed(G, seed_set):
            break

        seed_set.add(node)

    return seed_set


def best_by_size(seed_sets: list[Set[int]]) -> Set[int]:
    """Devuelve la solución de menor cardinalidad."""
    return min(seed_sets, key=len)


def evaluate_variants(n: int, m: int, run: int) -> list[dict]:
    """
    Ejecuta todas las variantes sobre el mismo grafo BA.
    Las semillas no dependen del método comparado.
    """
    base_seed = 1000 * n + 10 * m + run

    graph_seed = base_seed
    grasp_seed = 500_000 + base_seed
    sa_seed = 600_000 + base_seed
    random_seed = 700_000 + base_seed

    G = generate_ba_graph(n, m, seed=graph_seed)
    centralities = CentralityCache.compute(G)

    # --------------------------------------------------------
    # Baselines
    # --------------------------------------------------------
    degree_solution = refine(G, greedy_degree(G))

    random_solutions = []
    for restart in range(GRASP_ITER):
        rng_random = random.Random(random_seed + restart)
        solution = greedy_random(G, rng_random)
        random_solutions.append(refine(G, solution))

    random_solution = best_by_size(random_solutions)

    # --------------------------------------------------------
    # GRASP: generar los mismos arranques para todas las variantes
    # --------------------------------------------------------
    rng_grasp = random.Random(grasp_seed)

    raw_grasp_solutions = [
        grasp_construct(
            G,
            centralities,
            alpha=ALPHA,
            ad=AD,
            ae=AE,
            ab=AB,
            rng=rng_grasp,
        )
        for _ in range(GRASP_ITER)
    ]

    grasp_solution = best_by_size(raw_grasp_solutions)

    refined_grasp_solutions = [
        refine(G, solution)
        for solution in raw_grasp_solutions
    ]

    grasp_refined_solution = best_by_size(refined_grasp_solutions)

    # --------------------------------------------------------
    # SA sin refinamiento final
    # --------------------------------------------------------
    sa_solution, _ = simulated_annealing(
        G,
        grasp_refined_solution,
        T_init=T_INIT,
        T_min=T_MIN,
        cooling=COOLING,
        steps_per_temp=STEPS_PER_TEMP,
        lam=LAM,
        rng=random.Random(sa_seed),
        apply_final_refinement=False,
    )

    full_solution = refine(G, sa_solution)

    methods = {
        "Random + R": random_solution,
        "Greedy degree + R": degree_solution,
        "GRASP": grasp_solution,
        "GRASP + R": grasp_refined_solution,
        "GRASP + R + SA": sa_solution,
        "GRASP + R + SA + R": full_solution,
    }

    rows = []

    for method, solution in methods.items():
        rows.append(
            {
                "n": n,
                "m": m,
                "run": run,
                "method": method,
                "seed_size": len(solution),
                "seed_ratio": len(solution) / n,
                "is_perfect": int(is_perfect_seed(G, solution)),
            }
        )

    return rows


rows = []

for n in N_VALUES:
    for m in M_VALUES:
        for run in range(N_RUNS):
            rows.extend(evaluate_variants(n, m, run))

with open("results_pipeline_ablation2.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("Saved: results_pipeline_ablation2.csv")