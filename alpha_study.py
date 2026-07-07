import csv
import random
import time
import warnings
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pap_solver import (
    generate_ba_graph,
    CentralityCache,
    grasp_construct,
    refine,
    is_perfect_seed,
)

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────
# 1. PARAMETERS
# ──────────────────────────────────────────────────────────────

INSTANCES = [
    (50, 2),
    (100, 2),
    (100, 3),
    (150, 2),
    (200, 3),
]

ALPHA_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

N_RUNS = 5
GRASP_ITER = 8

# Configuración de la función de aporte. Se usa solo grado para que el
# estudio sea coherente con la versión simplificada/final del algoritmo.
AD = 1.0
AE = 0.0
AB = 0.0


# ──────────────────────────────────────────────────────────────
# 2. AUXILIARY FUNCTIONS
# ──────────────────────────────────────────────────────────────

def get_rows(rows, alpha=None, n=None, m=None):
    out = rows

    if alpha is not None:
        out = [r for r in out if r["alpha"] == alpha]

    if n is not None:
        out = [r for r in out if r["n"] == n]

    if m is not None:
        out = [r for r in out if r["m"] == m]

    return out


def mean_value(rows, key):
    return np.nanmean([r[key] for r in rows])


def std_value(rows, key):
    return np.nanstd([r[key] for r in rows])


def compute_degree_only_cache(G):
    """
    Centrality cache using only degree centrality.
    Eigenvector and betweenness are set to zero because AE=AB=0.
    This avoids computing unnecessary centralities in the alpha study.
    """
    n = G.number_of_nodes()

    if n <= 1:
        degree = {v: 0.0 for v in G.nodes()}
    else:
        degree = {v: G.degree(v) / (n - 1) for v in G.nodes()}

    zero = {v: 0.0 for v in G.nodes()}

    return CentralityCache(
        degree=degree,
        eigenvector=zero,
        betweenness=zero,
    )


def solve_grasp_refine(G, alpha, seed):
    """
    Run only GRASP multi-start + refinement, without Simulated Annealing.
    """
    rng = random.Random(seed)
    cent = compute_degree_only_cache(G)

    t0 = time.perf_counter()

    best_seed = set(G.nodes())

    for _ in range(GRASP_ITER):
        sol = grasp_construct(
            G,
            cent,
            alpha=alpha,
            ad=AD,
            ae=AE,
            ab=AB,
            rng=rng,
        )

        sol = refine(G, sol)

        if len(sol) < len(best_seed):
            best_seed = set(sol)

    time_total = time.perf_counter() - t0
    perfect = is_perfect_seed(G, best_seed)

    return best_seed, perfect, time_total


# ──────────────────────────────────────────────────────────────
# 3. RUN EXPERIMENT
# ──────────────────────────────────────────────────────────────

print("Running alpha study with GRASP + refinement only...")

rows = []

total = len(ALPHA_VALUES) * len(INSTANCES) * N_RUNS
done = 0

for alpha in ALPHA_VALUES:
    for (n, m) in INSTANCES:
        for run in range(N_RUNS):

            seed = 1000 * n + 10 * m + run

            G = generate_ba_graph(n=n, m=m, seed=seed)

            seed_set, is_perfect, time_total = solve_grasp_refine(
                G=G,
                alpha=alpha,
                seed=seed,
            )

            rows.append({
                "alpha": alpha,
                "n": n,
                "m": m,
                "run": run,
                "seed_size": len(seed_set),
                "seed_ratio": len(seed_set) / n,
                "is_perfect": int(is_perfect),
                "time_total": round(time_total, 4),
            })

            done += 1

            if done % 55 == 0 or done == total:
                print(
                    f"[{done:3d}/{total}] "
                    f"alpha={alpha:.1f}, n={n}, m={m}, run={run}, "
                    f"|S*|={len(seed_set)}, ratio={len(seed_set)/n:.3f}, "
                    f"ok={is_perfect}"
                )
                sys.stdout.flush()


# ──────────────────────────────────────────────────────────────
# 4. COMPUTE %DEV AND #BEST
# ──────────────────────────────────────────────────────────────

# Mejor solución para cada instancia emparejada entre todos los alpha
best_by_instance = {}

for row in rows:
    key = (row["n"], row["m"], row["run"])

    if key not in best_by_instance:
        best_by_instance[key] = row["seed_size"]
    else:
        best_by_instance[key] = min(best_by_instance[key], row["seed_size"])

for row in rows:
    key = (row["n"], row["m"], row["run"])
    best = best_by_instance[key]

    row["best_seed_size_for_instance"] = best
    row["pct_dev"] = 100.0 * (row["seed_size"] - best) / best
    row["is_best"] = int(row["seed_size"] == best)


# ──────────────────────────────────────────────────────────────
# 5. SAVE CSV
# ──────────────────────────────────────────────────────────────

csv_keys = [
    "alpha",
    "n",
    "m",
    "run",
    "seed_size",
    "seed_ratio",
    "best_seed_size_for_instance",
    "pct_dev",
    "is_best",
    "is_perfect",
    "time_total",
]

with open("results_alpha.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=csv_keys)
    writer.writeheader()
    writer.writerows(rows)

print("CSV saved: results_alpha.csv")


# ──────────────────────────────────────────────────────────────
# 6. PLOT STYLE
# ──────────────────────────────────────────────────────────────

STYLE = {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
}


# ──────────────────────────────────────────────────────────────
# 7. FIGURE 1: %DEV BOXPLOT
# ──────────────────────────────────────────────────────────────

with plt.rc_context(STYLE):

    fig, ax = plt.subplots(figsize=(8, 5))

    data = [
        [r["pct_dev"] for r in rows if r["alpha"] == alpha]
        for alpha in ALPHA_VALUES
    ]

    ax.boxplot(
        data,
        labels=[str(alpha) for alpha in ALPHA_VALUES],
        showmeans=True,
        showfliers=False,
        meanprops=dict(
            marker="^",
            markerfacecolor="green",
            markeredgecolor="green",
            markersize=9,
            markeredgewidth=1.5,
        ),
        medianprops=dict(
            color="orange",
            linewidth=2.5,
        ),
        boxprops=dict(linewidth=1.8),
        whiskerprops=dict(linewidth=1.8),
        capprops=dict(linewidth=1.8),
    )

    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("%dev")
    ax.set_title("Desviación porcentual respecto a la mejor solución por alpha")

    fig.tight_layout()
    fig.savefig("fig_alpha_pctdev_boxplot.png", bbox_inches="tight")
    plt.close(fig)

print("Saved fig_alpha_pctdev_boxplot.png")


# ──────────────────────────────────────────────────────────────
# 8. FIGURE 2: #BEST AND MEAN %DEV
# ──────────────────────────────────────────────────────────────

with plt.rc_context(STYLE):

    fig, ax1 = plt.subplots(figsize=(8, 5))

    x = np.arange(len(ALPHA_VALUES))

    n_best = []
    mean_dev = []

    for alpha in ALPHA_VALUES:
        alpha_rows = get_rows(rows, alpha=alpha)
        n_best.append(sum(r["is_best"] for r in alpha_rows))
        mean_dev.append(mean_value(alpha_rows, "pct_dev"))

    ax1.bar(x, n_best, alpha=0.75)
    ax1.set_xlabel(r"$\alpha$")
    ax1.set_ylabel("Número de mejores soluciones (#best)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(alpha) for alpha in ALPHA_VALUES])

    ax2 = ax1.twinx()
    ax2.plot(x, mean_dev, marker="o", linewidth=1.8)
    ax2.set_ylabel("%dev medio")

    ax1.set_title("Número de mejores soluciones y %dev medio por alpha")

    fig.tight_layout()
    fig.savefig("fig_alpha_best_dev.png", bbox_inches="tight")
    plt.close(fig)

print("Saved fig_alpha_best_dev.png")


# ──────────────────────────────────────────────────────────────
# 9. FIGURE 3: SUPPORT FIGURE WITH SEED RATIO
# ──────────────────────────────────────────────────────────────

with plt.rc_context(STYLE):

    fig, ax = plt.subplots(figsize=(8, 5))

    for n, m in INSTANCES:
        means = []
        stds = []

        for alpha in ALPHA_VALUES:
            alpha_rows = get_rows(rows, alpha=alpha, n=n, m=m)
            means.append(mean_value(alpha_rows, "seed_ratio"))
            stds.append(std_value(alpha_rows, "seed_ratio"))

        ax.errorbar(
            ALPHA_VALUES,
            means,
            yerr=stds,
            marker="o",
            linewidth=1.5,
            capsize=3,
            label=f"n={n}, m={m}",
        )

    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$|S_0^*|/n$")
    ax.set_title(r"Cociente normalizado de la semilla respecto a $\alpha$")
    ax.legend()

    fig.tight_layout()
    fig.savefig("fig_alpha_seed_ratio_support.png", bbox_inches="tight")
    plt.close(fig)

print("Saved fig_alpha_seed_ratio_support.png")


# ──────────────────────────────────────────────────────────────
# 10. SUMMARY TABLE
# ──────────────────────────────────────────────────────────────

print("\n── Alpha sensitivity summary: GRASP + refinement ─────────────────────")
print(
    f'{"alpha":>7}  '
    f'{"mean ratio":>10}  '
    f'{"%dev":>8}  '
    f'{"#best":>6}  '
    f'{"perfect%":>9}  '
    f'{"time":>8}'
)

print("-" * 68)

for alpha in ALPHA_VALUES:

    alpha_rows = get_rows(rows, alpha=alpha)

    mean_ratio = mean_value(alpha_rows, "seed_ratio")
    mean_dev = mean_value(alpha_rows, "pct_dev")
    best_count = sum(r["is_best"] for r in alpha_rows)
    perfect_rate = 100.0 * mean_value(alpha_rows, "is_perfect")
    mean_time = mean_value(alpha_rows, "time_total")

    print(
        f"{alpha:>7.1f}  "
        f"{mean_ratio:>10.4f}  "
        f"{mean_dev:>8.3f}  "
        f"{best_count:>6d}  "
        f"{perfect_rate:>8.1f}%  "
        f"{mean_time:>8.2f}"
    )

print("\nALL DONE")
