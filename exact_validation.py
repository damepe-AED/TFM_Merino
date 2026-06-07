import csv
import time
from itertools import combinations
from typing import Optional, Set, Tuple

from pap_solver import generate_ba_graph, is_perfect_seed, solve_pap


# ──────────────────────────────────────────────────────────────
# Experimental configuration
# ──────────────────────────────────────────────────────────────

N_VALUES = [42, 44, 46, 48, 50]
M_VALUES = [1, 2, 3]
N_RUNS = 5

# Maximum time allowed for the exact search on each instance
MAX_SECONDS = 900.0  # 15 minutes

GRASP_ITER = 8
ALPHA = 0.3
T_INIT = 5.0
T_MIN = 0.01
COOLING = 0.97
STEPS_PER_TEMP = 25
LAM = 2.0

OUTPUT_FILE = "results_exact_validation_test.csv"


# ──────────────────────────────────────────────────────────────
# Exact enumeration with timeout
# ──────────────────────────────────────────────────────────────

def exact_seed_set(
    G,
    max_seconds: float,
) -> Tuple[Optional[Set[int]], float, bool]:
    """
    Enumerate subsets by increasing cardinality.

    The first perfect seed set found is optimal.

    Returns
    -------
    solution:
        Optimal perfect seed set if found before the timeout.
        None if the time limit is reached.

    elapsed:
        Exact-search execution time in seconds.

    completed:
        True if the exact optimum was found.
        False if the time limit was reached.
    """
    start = time.perf_counter()
    nodes = list(G.nodes())

    for size in range(len(nodes) + 1):
        for candidate in combinations(nodes, size):
            elapsed = time.perf_counter() - start

            if elapsed > max_seconds:
                return None, elapsed, False

            candidate_set = set(candidate)

            if is_perfect_seed(G, candidate_set):
                elapsed = time.perf_counter() - start
                return candidate_set, elapsed, True

    elapsed = time.perf_counter() - start
    return None, elapsed, False


# ──────────────────────────────────────────────────────────────
# Run experiments and save each row immediately
# ──────────────────────────────────────────────────────────────

fieldnames = [
    "n",
    "m",
    "run",
    "optimal_size",
    "heuristic_size",
    "absolute_gap",
    "relative_gap_pct",
    "exact_time",
    "completed",
    "is_optimal",
]

total = len(N_VALUES) * len(M_VALUES) * N_RUNS
done = 0

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    file.flush()

    for n in N_VALUES:
        for m in M_VALUES:
            if m >= n:
                continue

            for run in range(N_RUNS):
                seed = 1000 * n + 10 * m + run

                G = generate_ba_graph(
                    n=n,
                    m=m,
                    seed=seed,
                )

                exact_solution, exact_time, completed = exact_seed_set(
                    G,
                    max_seconds=MAX_SECONDS,
                )

                heuristic = solve_pap(
                    G,
                    n_grasp_iter=GRASP_ITER,
                    alpha=ALPHA,
                    T_init=T_INIT,
                    T_min=T_MIN,
                    cooling=COOLING,
                    steps_per_temp=STEPS_PER_TEMP,
                    lam=LAM,
                    seed=500_000 + seed,
                )

                if completed and exact_solution is not None:
                    optimal_size = len(exact_solution)
                    absolute_gap = heuristic.seed_size - optimal_size

                    relative_gap_pct = (
                        100 * absolute_gap / optimal_size
                        if optimal_size > 0
                        else 0.0
                    )

                    is_optimal = int(absolute_gap == 0)
                    status = "OK"

                else:
                    # The optimum is unknown because the exact search timed out
                    optimal_size = None
                    absolute_gap = None
                    relative_gap_pct = None
                    is_optimal = None
                    status = "TIMEOUT"

                row = {
                    "n": n,
                    "m": m,
                    "run": run,
                    "optimal_size": optimal_size,
                    "heuristic_size": heuristic.seed_size,
                    "absolute_gap": absolute_gap,
                    "relative_gap_pct": relative_gap_pct,
                    "exact_time": round(exact_time, 4),
                    "completed": int(completed),
                    "is_optimal": is_optimal,
                }

                # Save immediately so partial results survive interruptions
                writer.writerow(row)
                file.flush()

                done += 1

                print(
                    f"[{done:2d}/{total}] "
                    f"n={n:3d}, "
                    f"m={m}, "
                    f"run={run}, "
                    f"status={status:7s}, "
                    f"exact_time={exact_time:8.2f}s, "
                    f"heuristic_size={heuristic.seed_size}"
                )

print(f"\nSaved: {OUTPUT_FILE}")