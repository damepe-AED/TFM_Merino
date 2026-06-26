import csv
import warnings
from itertools import product

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pap_solver import generate_ba_graph, solve_pap

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────
# 1. PARAMETERS
# ──────────────────────────────────────────────────────────────

LAMBDA_VALUES = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0]

INSTANCES = [
    (50, 2),
    (100, 2),
    (100, 3),
    (150, 2),
    (200, 3),
]

N_RUNS = 5

GRASP_ITER = 8
ALPHA = 0.3

T_INIT = 5.0
T_MIN = 0.01
COOLING = 0.97

STEPS = 25


# ──────────────────────────────────────────────────────────────
# 2. AUXILIARY FUNCTIONS
# ──────────────────────────────────────────────────────────────

def safe_pct(num, den):
    """Return 100*num/den, avoiding divisions by zero."""
    if den is None or den == 0:
        return np.nan
    try:
        if np.isnan(den):
            return np.nan
    except TypeError:
        pass
    return 100.0 * num / den


def get_rows(rows, lam=None, n=None, m=None):
    """Filter rows by lambda and/or instance."""
    out = rows

    if lam is not None:
        out = [r for r in out if r["lambda"] == lam]

    if n is not None:
        out = [r for r in out if r["n"] == n]

    if m is not None:
        out = [r for r in out if r["m"] == m]

    return out


def mean_value(rows, key):
    values = [r[key] for r in rows]
    return np.nanmean(values)


def std_value(rows, key):
    values = [r[key] for r in rows]
    return np.nanstd(values)


# ──────────────────────────────────────────────────────────────
# 3. RUN EXPERIMENT
# ──────────────────────────────────────────────────────────────

print("Running lambda sensitivity study...")

all_rows = []
histories = {}

total = len(LAMBDA_VALUES) * len(INSTANCES) * N_RUNS
done = 0

for (n, m), lam in product(INSTANCES, LAMBDA_VALUES):
    for run in range(N_RUNS):

        seed = 1000 * n + 10 * m + run

        G = generate_ba_graph(n=n, m=m, seed=seed)

        res = solve_pap(
            G,
            n_grasp_iter=GRASP_ITER,
            alpha=ALPHA,
            T_init=T_INIT,
            T_min=T_MIN,
            cooling=COOLING,
            steps_per_temp=STEPS,
            lam=lam,
            seed=seed,
        )

        stats = getattr(res, "sa_stats", None)

        if stats is None:
            raise RuntimeError(
                "res.sa_stats does not exist. "
                "First modify pap_solver.py so that simulated_annealing returns SA statistics."
            )

        required_keys = [
            "total_moves",
            "accepted_moves",
            "infeasible_proposed",
            "accepted_infeasible",
            "current_infeasible_steps",
            "uphill_proposed",
            "uphill_accepted",
            "best_updates",
        ]

        missing = [k for k in required_keys if k not in stats]
        if missing:
            raise RuntimeError(
                f"Missing SA statistics in res.sa_stats: {missing}. "
                "Check the modified simulated_annealing function."
            )

        total_moves = stats["total_moves"]
        accepted_moves = stats["accepted_moves"]
        infeasible_proposed = stats["infeasible_proposed"]
        accepted_infeasible = stats["accepted_infeasible"]
        current_infeasible_steps = stats["current_infeasible_steps"]
        uphill_proposed = stats["uphill_proposed"]
        uphill_accepted = stats["uphill_accepted"]
        best_updates = stats["best_updates"]

        row = {
            "n": n,
            "m": m,
            "lambda": lam,
            "run": run,

            # Final quality
            "seed_size": res.seed_size,
            "seed_ratio": res.seed_size / n,
            "is_perfect": int(res.is_perfect),
            "time_total": round(res.time_total, 4),

            # Raw SA statistics
            "sa_total_moves": total_moves,
            "sa_accepted_moves": accepted_moves,
            "sa_infeasible_proposed": infeasible_proposed,
            "sa_accepted_infeasible": accepted_infeasible,
            "sa_current_infeasible_steps": current_infeasible_steps,
            "sa_uphill_proposed": uphill_proposed,
            "sa_uphill_accepted": uphill_accepted,
            "sa_best_updates": best_updates,

            # Percentages for interpretation
            "pct_infeasible_proposed": safe_pct(infeasible_proposed, total_moves),
            "pct_current_infeasible": safe_pct(current_infeasible_steps, total_moves),
            "pct_accepted_infeasible": safe_pct(accepted_infeasible, accepted_moves),
            "pct_uphill_accepted": safe_pct(uphill_accepted, uphill_proposed),
        }

        all_rows.append(row)
        histories[(n, m, lam, run)] = res.sa_history

        done += 1

        if done % 20 == 0 or done == total:
            print(
                f"[{done:3d}/{total}] "
                f"n={n}, m={m}, lambda={lam}, run={run}, "
                f"|S*|={res.seed_size}, perfect={res.is_perfect}"
            )


# ──────────────────────────────────────────────────────────────
# 4. COMPUTE %DEV AND #BEST
# ──────────────────────────────────────────────────────────────

# Best seed size for each paired instance across lambda values
best_by_instance = {}

for row in all_rows:
    key = (row["n"], row["m"], row["run"])

    if key not in best_by_instance:
        best_by_instance[key] = row["seed_size"]
    else:
        best_by_instance[key] = min(best_by_instance[key], row["seed_size"])

for row in all_rows:
    key = (row["n"], row["m"], row["run"])
    best = best_by_instance[key]

    row["best_seed_size_for_instance"] = best
    row["pct_dev"] = 100.0 * (row["seed_size"] - best) / best
    row["is_best"] = int(row["seed_size"] == best)


# ──────────────────────────────────────────────────────────────
# 5. SAVE CSV
# ──────────────────────────────────────────────────────────────

csv_keys = [
    "n",
    "m",
    "lambda",
    "run",

    "seed_size",
    "seed_ratio",
    "best_seed_size_for_instance",
    "pct_dev",
    "is_best",
    "is_perfect",
    "time_total",

    "sa_total_moves",
    "sa_accepted_moves",
    "sa_infeasible_proposed",
    "sa_accepted_infeasible",
    "sa_current_infeasible_steps",
    "sa_uphill_proposed",
    "sa_uphill_accepted",
    "sa_best_updates",

    "pct_infeasible_proposed",
    "pct_current_infeasible",
    "pct_accepted_infeasible",
    "pct_uphill_accepted",
]

with open("results_lambda.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=csv_keys)
    writer.writeheader()
    writer.writerows(all_rows)

print("CSV saved: results_lambda.csv")


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
        [r["pct_dev"] for r in all_rows if r["lambda"] == lam]
        for lam in LAMBDA_VALUES
    ]

    ax.boxplot(
        data,
        labels=[str(lam) for lam in LAMBDA_VALUES],
        showmeans=True,
    )

    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\%dev$")
    ax.set_title(r"Desviación porcentual respecto a la mejor solución por $\lambda$")

    fig.tight_layout()
    fig.savefig("fig_lambda_pctdev_boxplot.png", bbox_inches="tight")
    plt.close(fig)

print("Saved fig_lambda_pctdev_boxplot.png")


# ──────────────────────────────────────────────────────────────
# 8. FIGURE 2: #BEST AND MEAN %DEV
# ──────────────────────────────────────────────────────────────

with plt.rc_context(STYLE):

    fig, ax1 = plt.subplots(figsize=(8, 5))

    x = np.arange(len(LAMBDA_VALUES))

    n_best = []
    mean_dev = []

    for lam in LAMBDA_VALUES:
        rows = get_rows(all_rows, lam=lam)
        n_best.append(sum(r["is_best"] for r in rows))
        mean_dev.append(mean_value(rows, "pct_dev"))

    ax1.bar(x, n_best, alpha=0.75)
    ax1.set_xlabel(r"$\lambda$")
    ax1.set_ylabel("Número de mejores soluciones (#best)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(lam) for lam in LAMBDA_VALUES])

    ax2 = ax1.twinx()
    ax2.plot(x, mean_dev, marker="o", linewidth=1.8)
    ax2.set_ylabel(r"$\%dev$ medio")

    ax1.set_title(r"Número de mejores soluciones y $\%dev$ medio por $\lambda$")

    fig.tight_layout()
    fig.savefig("fig_lambda_best_dev.png", bbox_inches="tight")
    plt.close(fig)

print("Saved fig_lambda_best_dev.png")


# ──────────────────────────────────────────────────────────────
# 9. FIGURE 3: INTERNAL SA BEHAVIOUR
# ──────────────────────────────────────────────────────────────

with plt.rc_context(STYLE):

    fig, ax = plt.subplots(figsize=(8, 5))

    mean_current_infeasible = []
    mean_accepted_infeasible = []
    mean_uphill_accepted = []

    for lam in LAMBDA_VALUES:
        rows = get_rows(all_rows, lam=lam)

        mean_current_infeasible.append(
            mean_value(rows, "pct_current_infeasible")
        )

        mean_accepted_infeasible.append(
            mean_value(rows, "pct_accepted_infeasible")
        )

        mean_uphill_accepted.append(
            mean_value(rows, "pct_uphill_accepted")
        )

    ax.plot(
        LAMBDA_VALUES,
        mean_current_infeasible,
        marker="o",
        linewidth=1.8,
        label="Tiempo en espacio no factible (%)",
    )

    ax.plot(
        LAMBDA_VALUES,
        mean_accepted_infeasible,
        marker="s",
        linewidth=1.8,
        label="Movimientos no factibles aceptados (%)",
    )

    ax.plot(
        LAMBDA_VALUES,
        mean_uphill_accepted,
        marker="D",
        linewidth=1.8,
        label="Empeoramientos aceptados (%)",
    )

    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("Porcentaje medio")
    ax.set_title(r"Comportamiento interno del SA según $\lambda$")
    ax.legend()

    fig.tight_layout()
    fig.savefig("fig_lambda_sa_behaviour.png", bbox_inches="tight")
    plt.close(fig)

print("Saved fig_lambda_sa_behaviour.png")


# ──────────────────────────────────────────────────────────────
# 10. FIGURE 4: SUPPORT FIGURE WITH SEED RATIO
# ──────────────────────────────────────────────────────────────

with plt.rc_context(STYLE):

    fig, ax = plt.subplots(figsize=(8, 5))

    for n, m in INSTANCES:
        means = []
        stds = []

        for lam in LAMBDA_VALUES:
            rows = get_rows(all_rows, lam=lam, n=n, m=m)
            means.append(mean_value(rows, "seed_ratio"))
            stds.append(std_value(rows, "seed_ratio"))

        ax.errorbar(
            LAMBDA_VALUES,
            means,
            yerr=stds,
            marker="o",
            linewidth=1.5,
            capsize=3,
            label=f"n={n}, m={m}",
        )

    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$|S_0^*|/n$")
    ax.set_title(r"Cociente normalizado de la semilla respecto a $\lambda$")
    ax.legend()

    fig.tight_layout()
    fig.savefig("fig_lambda_seed_ratio_support.png", bbox_inches="tight")
    plt.close(fig)

print("Saved fig_lambda_seed_ratio_support.png")


# ──────────────────────────────────────────────────────────────
# 11. SUMMARY TABLE
# ──────────────────────────────────────────────────────────────

print("\n── Lambda sensitivity summary ─────────────────────────────────────────")
print(
    f'{"lambda":>7}  '
    f'{"mean ratio":>10}  '
    f'{"%dev":>8}  '
    f'{"#best":>6}  '
    f'{"inf.curr%":>10}  '
    f'{"inf.acc%":>9}  '
    f'{"uphill%":>9}  '
    f'{"time":>8}'
)

print("-" * 88)

for lam in LAMBDA_VALUES:

    rows = get_rows(all_rows, lam=lam)

    mean_ratio = mean_value(rows, "seed_ratio")
    mean_dev = mean_value(rows, "pct_dev")
    best_count = sum(r["is_best"] for r in rows)

    mean_inf_current = mean_value(rows, "pct_current_infeasible")
    mean_inf_acc = mean_value(rows, "pct_accepted_infeasible")
    mean_uphill = mean_value(rows, "pct_uphill_accepted")
    mean_time = mean_value(rows, "time_total")

    print(
        f"{lam:>7.1f}  "
        f"{mean_ratio:>10.4f}  "
        f"{mean_dev:>8.3f}  "
        f"{best_count:>6d}  "
        f"{mean_inf_current:>10.2f}  "
        f"{mean_inf_acc:>9.2f}  "
        f"{mean_uphill:>9.2f}  "
        f"{mean_time:>8.2f}"
    )

print("\nALL DONE")
