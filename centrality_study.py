import csv
import warnings
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pap_solver import generate_ba_graph, solve_pap

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

CENTRALITY_CONFIGS = [
    ("Todas", 1.0, 1.0, 1.0),
    ("Solo grado", 1.0, 0.0, 0.0),
    ("Solo eigenvector", 0.0, 1.0, 0.0),
    ("Solo betweenness", 0.0, 0.0, 1.0),
    ("Sin grado", 0.0, 1.0, 1.0),
    ("Sin eigenvector", 1.0, 0.0, 1.0),
    ("Sin betweenness", 1.0, 1.0, 0.0),
]

N_RUNS = 5

GRASP_ITER = 8

# Valores seleccionados en los estudios anteriores
ALPHA = 0.8
LAM = 1.5

T_INIT = 5.0
T_MIN = 0.01
COOLING = 0.97
STEPS = 25


# ──────────────────────────────────────────────────────────────
# 2. AUXILIARY FUNCTIONS
# ──────────────────────────────────────────────────────────────

def get_rows(rows, config=None, n=None, m=None):
    out = rows

    if config is not None:
        out = [r for r in out if r["config"] == config]

    if n is not None:
        out = [r for r in out if r["n"] == n]

    if m is not None:
        out = [r for r in out if r["m"] == m]

    return out


def mean_value(rows, key):
    return np.nanmean([r[key] for r in rows])


def std_value(rows, key):
    return np.nanstd([r[key] for r in rows])


def get_attr(obj, name, default=np.nan):
    return getattr(obj, name, default)


# ──────────────────────────────────────────────────────────────
# 3. RUN EXPERIMENT
# ──────────────────────────────────────────────────────────────

print("Running centrality ablation and efficiency study...")

rows = []

total = len(CENTRALITY_CONFIGS) * len(INSTANCES) * N_RUNS
done = 0

for config_name, ad, ae, ab in CENTRALITY_CONFIGS:
    for (n, m) in INSTANCES:
        for run in range(N_RUNS):

            seed = 1000 * n + 10 * m + run

            G = generate_ba_graph(n=n, m=m, seed=seed)

            res = solve_pap(
                G,
                n_grasp_iter=GRASP_ITER,
                alpha=ALPHA,
                ad=ad,
                ae=ae,
                ab=ab,
                T_init=T_INIT,
                T_min=T_MIN,
                cooling=COOLING,
                steps_per_temp=STEPS,
                lam=LAM,
                seed=seed,
            )

            rows.append({
                "config": config_name,
                "ad": ad,
                "ae": ae,
                "ab": ab,
                "n": n,
                "m": m,
                "run": run,

                # Final quality
                "seed_size": res.seed_size,
                "seed_ratio": res.seed_size / n,
                "is_perfect": int(res.is_perfect),

                # Times
                "time_total": round(get_attr(res, "time_total"), 6),
                "time_grasp": round(get_attr(res, "time_grasp"), 6),
                "time_sa": round(get_attr(res, "time_sa"), 6),
                "time_centrality": round(get_attr(res, "time_centrality"), 6),
                "time_degree": round(get_attr(res, "time_degree"), 6),
                "time_eigenvector": round(get_attr(res, "time_eigenvector"), 6),
                "time_betweenness": round(get_attr(res, "time_betweenness"), 6),
            })

            done += 1

            if done % 25 == 0 or done == total:
                print(
                    f"[{done:3d}/{total}] "
                    f"config={config_name}, n={n}, m={m}, run={run}, "
                    f"|S*|={res.seed_size}, ratio={res.seed_size/n:.3f}, "
                    f"centrality_time={get_attr(res, 'time_centrality'):.4f}, "
                    f"ok={res.is_perfect}"
                )
                sys.stdout.flush()


# ──────────────────────────────────────────────────────────────
# 4. COMPUTE %DEV AND #BEST
# ──────────────────────────────────────────────────────────────

# Mejor solución para cada instancia emparejada entre configuraciones
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
    "config",
    "ad",
    "ae",
    "ab",
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
    "time_grasp",
    "time_sa",
    "time_centrality",
    "time_degree",
    "time_eigenvector",
    "time_betweenness",
]

with open("results_centrality.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=csv_keys)
    writer.writeheader()
    writer.writerows(rows)

print("CSV saved: results_centrality.csv")


# ──────────────────────────────────────────────────────────────
# 6. PLOT STYLE
# ──────────────────────────────────────────────────────────────

STYLE = {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
}


# ──────────────────────────────────────────────────────────────
# 7. FIGURE 1: %DEV BOXPLOT
# ──────────────────────────────────────────────────────────────

with plt.rc_context(STYLE):

    fig, ax = plt.subplots(figsize=(9, 5))

    data = [
        [r["pct_dev"] for r in rows if r["config"] == config_name]
        for config_name, _, _, _ in CENTRALITY_CONFIGS
    ]

    labels = [config_name for config_name, _, _, _ in CENTRALITY_CONFIGS]

    ax.boxplot(
        data,
        labels=labels,
        showmeans=True,
    )

    ax.set_xlabel("Configuración de centralidad")
    ax.set_ylabel("%dev")
    ax.set_title("Desviación porcentual respecto a la mejor solución por centralidad")
    ax.tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig.savefig("fig_centrality_pctdev_boxplot.png", bbox_inches="tight")
    plt.close(fig)

print("Saved fig_centrality_pctdev_boxplot.png")


# ──────────────────────────────────────────────────────────────
# 8. FIGURE 2: #BEST AND MEAN %DEV
# ──────────────────────────────────────────────────────────────

with plt.rc_context(STYLE):

    fig, ax1 = plt.subplots(figsize=(9, 5))

    x = np.arange(len(CENTRALITY_CONFIGS))
    labels = [config_name for config_name, _, _, _ in CENTRALITY_CONFIGS]

    n_best = []
    mean_dev = []

    for config_name, _, _, _ in CENTRALITY_CONFIGS:
        config_rows = get_rows(rows, config=config_name)
        n_best.append(sum(r["is_best"] for r in config_rows))
        mean_dev.append(mean_value(config_rows, "pct_dev"))

    ax1.bar(x, n_best, alpha=0.75)
    ax1.set_xlabel("Configuración de centralidad")
    ax1.set_ylabel("Número de mejores soluciones (#best)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(x, mean_dev, marker="o", linewidth=1.8)
    ax2.set_ylabel("%dev medio")

    ax1.set_title("Número de mejores soluciones y %dev medio por centralidad")

    fig.tight_layout()
    fig.savefig("fig_centrality_best_dev.png", bbox_inches="tight")
    plt.close(fig)

print("Saved fig_centrality_best_dev.png")


# ──────────────────────────────────────────────────────────────
# 9. FIGURE 3: CENTRALITY COMPUTATION TIME
# ──────────────────────────────────────────────────────────────

with plt.rc_context(STYLE):

    fig, ax = plt.subplots(figsize=(9, 5))

    labels = []
    mean_times = []

    for config_name, _, _, _ in CENTRALITY_CONFIGS:
        config_rows = get_rows(rows, config=config_name)
        labels.append(config_name)
        mean_times.append(mean_value(config_rows, "time_centrality"))

    x = np.arange(len(labels))

    ax.bar(x, mean_times, alpha=0.75)
    ax.set_xlabel("Configuración de centralidad")
    ax.set_ylabel("Tiempo medio de centralidades (s)")
    ax.set_title("Coste medio de preprocesamiento de centralidades")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")

    fig.tight_layout()
    fig.savefig("fig_centrality_time.png", bbox_inches="tight")
    plt.close(fig)

print("Saved fig_centrality_time.png")


# ──────────────────────────────────────────────────────────────
# 10. FIGURE 4: QUALITY-TIME TRADE-OFF
# ──────────────────────────────────────────────────────────────

with plt.rc_context(STYLE):

    fig, ax = plt.subplots(figsize=(8, 5))

    for config_name, _, _, _ in CENTRALITY_CONFIGS:
        config_rows = get_rows(rows, config=config_name)

        x_time = mean_value(config_rows, "time_centrality")
        y_dev = mean_value(config_rows, "pct_dev")

        ax.scatter(x_time, y_dev, s=80)
        ax.text(
            x_time,
            y_dev,
            " " + config_name,
            fontsize=9,
            va="center",
        )

    ax.set_xlabel("Tiempo medio de centralidades (s)")
    ax.set_ylabel("%dev medio")
    ax.set_title("Compromiso entre calidad y coste de centralidades")

    fig.tight_layout()
    fig.savefig("fig_centrality_tradeoff.png", bbox_inches="tight")
    plt.close(fig)

print("Saved fig_centrality_tradeoff.png")


# ──────────────────────────────────────────────────────────────
# 11. SUMMARY TABLE
# ──────────────────────────────────────────────────────────────

print("\n── Centrality ablation and efficiency summary ─────────────────────────")
print(
    f'{"config":>18}  '
    f'{"mean ratio":>10}  '
    f'{"%dev":>8}  '
    f'{"#best":>6}  '
    f'{"cent.time":>10}  '
    f'{"total":>8}  '
    f'{"perfect%":>9}'
)

print("-" * 86)

for config_name, _, _, _ in CENTRALITY_CONFIGS:

    config_rows = get_rows(rows, config=config_name)

    mean_ratio = mean_value(config_rows, "seed_ratio")
    mean_dev = mean_value(config_rows, "pct_dev")
    best_count = sum(r["is_best"] for r in config_rows)
    mean_cent_time = mean_value(config_rows, "time_centrality")
    mean_total_time = mean_value(config_rows, "time_total")
    perfect_rate = 100.0 * mean_value(config_rows, "is_perfect")

    print(
        f"{config_name:>18}  "
        f"{mean_ratio:>10.4f}  "
        f"{mean_dev:>8.3f}  "
        f"{best_count:>6d}  "
        f"{mean_cent_time:>10.4f}  "
        f"{mean_total_time:>8.3f}  "
        f"{perfect_rate:>8.1f}%"
    )

print("\nALL DONE")
