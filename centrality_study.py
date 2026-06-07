"""
centrality_study.py
===================
Sensitivity and ablation study for the centrality weights
(a_d, a_e, a_b) used in the GRASP benefit function.

The script performs two complementary experiments:

1. One-at-a-time sweep:
   Each weight is varied in [0, 2], while the other two remain fixed at 1.

2. Ablation study:
   Relevant combinations are compared, including single-measure,
   leave-one-out and double-weight configurations.

The BA graph seed and the algorithm seed remain constant when the
centrality weights change. Therefore, differences can be attributed
to the weights rather than to changes in the generated graphs.

Outputs
-------
results_centrality_sweep.csv
results_centrality_ablation.csv
fig_centrality_sweep_aggregated.png
fig_centrality_sweep_per_instance.png
fig_centrality_ablation.png
"""

import csv
import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from pap_solver import generate_ba_graph, solve_pap

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────
# 1. Experimental configuration
# ──────────────────────────────────────────────────────────────

INSTANCES = [
    (50, 2),
    (100, 2),
    (100, 3),
    (150, 2),
    (200, 3),
]

# Sweep from 0 to 2 with increments of 0.25
SWEEP_VALUES = np.linspace(0.0, 2.0, 9).round(2).tolist()

N_RUNS = 5

# Final GRASP + SA configuration
GRASP_ITER = 8
ALPHA = 0.3
T_INIT = 5.0
T_MIN = 0.01
COOLING = 0.97
STEPS_PER_TEMP = 25
LAM = 2.0

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ──────────────────────────────────────────────────────────────
# 2. Ablation configurations
# ──────────────────────────────────────────────────────────────

ABLATION_CONFIGS = [
    ("Todas iguales",       1.0, 1.0, 1.0),

    # One measure only
    ("Solo grado",          1.0, 0.0, 0.0),
    ("Solo vector propio",  0.0, 1.0, 0.0),
    ("Solo intermediación", 0.0, 0.0, 1.0),

    # Leave-one-out
    ("Sin grado",           0.0, 1.0, 1.0),
    ("Sin vector propio",   1.0, 0.0, 1.0),
    ("Sin intermediación",  1.0, 1.0, 0.0),

    # Double weight
    ("Doble grado",         2.0, 1.0, 1.0),
    ("Doble vector propio", 1.0, 2.0, 1.0),
    ("Doble intermediación", 1.0, 1.0, 2.0),
]


# ──────────────────────────────────────────────────────────────
# 3. Reproducible seeds
# ──────────────────────────────────────────────────────────────

def get_seeds(n: int, m: int, run: int) -> tuple[int, int]:
    """
    Return deterministic seeds for:
      1. BA graph generation.
      2. GRASP + SA pseudo-random decisions.

    The seeds do not depend on the centrality weights.
    Therefore, each weight configuration is evaluated on the
    same graphs and under comparable pseudo-random conditions.
    """
    base_seed = 1000 * n + 10 * m + run

    graph_seed = base_seed
    algorithm_seed = 500_000 + base_seed

    return graph_seed, algorithm_seed


# ──────────────────────────────────────────────────────────────
# 4. Execute one experiment
# ──────────────────────────────────────────────────────────────

def run_one(
    n: int,
    m: int,
    ad: float,
    ae: float,
    ab: float,
    run: int,
) -> dict:
    """
    Run the solver for one BA instance and one centrality-weight
    configuration.
    """
    if ad + ae + ab <= 0:
        raise ValueError(
            "At least one centrality weight must be strictly positive."
        )

    graph_seed, algorithm_seed = get_seeds(n, m, run)

    graph = generate_ba_graph(
        n=n,
        m=m,
        seed=graph_seed,
    )

    result = solve_pap(
        graph,
        n_grasp_iter=GRASP_ITER,
        alpha=ALPHA,
        ad=ad,
        ae=ae,
        ab=ab,
        T_init=T_INIT,
        T_min=T_MIN,
        cooling=COOLING,
        steps_per_temp=STEPS_PER_TEMP,
        lam=LAM,
        seed=algorithm_seed,
    )

    return {
        "n": n,
        "m": m,
        "run": run,
        "graph_seed": graph_seed,
        "algorithm_seed": algorithm_seed,
        "ad": ad,
        "ae": ae,
        "ab": ab,
        "seed_size": result.seed_size,
        "seed_ratio": result.seed_size / n,
        "is_perfect": int(result.is_perfect),
        "time_total": round(result.time_total, 4),
    }


# ──────────────────────────────────────────────────────────────
# 5. Sweep study
# ──────────────────────────────────────────────────────────────

def run_sweep_study() -> list[dict]:
    """
    Vary one weight at a time while keeping the other two fixed at 1.
    """
    dimensions = [
        ("ad", "Grado", 0),
        ("ae", "Vector propio", 1),
        ("ab", "Intermediación", 2),
    ]

    rows: list[dict] = []

    total = (
        len(dimensions)
        * len(SWEEP_VALUES)
        * len(INSTANCES)
        * N_RUNS
    )

    done = 0

    print("Running centrality-weight sweep...")
    print(f"  Sweep values : {SWEEP_VALUES}")
    print(f"  Instances    : {INSTANCES}")
    print(f"  Runs each    : {N_RUNS}")
    print(f"  Total runs   : {total}\n")

    for dimension, label, position in dimensions:
        for sweep_value in SWEEP_VALUES:
            for n, m in INSTANCES:
                for run in range(N_RUNS):
                    weights = [1.0, 1.0, 1.0]
                    weights[position] = sweep_value

                    ad, ae, ab = weights

                    result = run_one(
                        n=n,
                        m=m,
                        ad=ad,
                        ae=ae,
                        ab=ab,
                        run=run,
                    )

                    result.update(
                        {
                            "dimension": dimension,
                            "dimension_label": label,
                            "sweep_value": sweep_value,
                        }
                    )

                    rows.append(result)
                    done += 1

                    if done % 50 == 0 or done == total:
                        print(
                            f"  [{done:4d}/{total}] "
                            f"{dimension}={sweep_value:.2f} "
                            f"n={n:3d} "
                            f"m={m} "
                            f"run={run} "
                            f"|S*|={result['seed_size']:3d} "
                            f"ratio={result['seed_ratio']:.3f} "
                            f"perfect={bool(result['is_perfect'])}"
                        )
                        sys.stdout.flush()

    print("\nSweep completed.\n")
    return rows


# ──────────────────────────────────────────────────────────────
# 6. Ablation study
# ──────────────────────────────────────────────────────────────

def run_ablation_study() -> list[dict]:
    """
    Compare relevant combinations of centrality weights.
    """
    rows: list[dict] = []

    total = len(ABLATION_CONFIGS) * len(INSTANCES) * N_RUNS
    done = 0

    print("Running centrality-weight ablation study...")
    print(f"  Configurations : {len(ABLATION_CONFIGS)}")
    print(f"  Instances      : {INSTANCES}")
    print(f"  Runs each      : {N_RUNS}")
    print(f"  Total runs     : {total}\n")

    for config_name, ad, ae, ab in ABLATION_CONFIGS:
        for n, m in INSTANCES:
            for run in range(N_RUNS):
                result = run_one(
                    n=n,
                    m=m,
                    ad=ad,
                    ae=ae,
                    ab=ab,
                    run=run,
                )

                result.update(
                    {
                        "config_name": config_name,
                    }
                )

                rows.append(result)
                done += 1

                if done % 25 == 0 or done == total:
                    print(
                        f"  [{done:3d}/{total}] "
                        f"{config_name:<20} "
                        f"n={n:3d} "
                        f"m={m} "
                        f"|S*|={result['seed_size']:3d} "
                        f"ratio={result['seed_ratio']:.3f} "
                        f"perfect={bool(result['is_perfect'])}"
                    )
                    sys.stdout.flush()

    print("\nAblation study completed.\n")
    return rows


# ──────────────────────────────────────────────────────────────
# 7. Save CSV files
# ──────────────────────────────────────────────────────────────

def save_csv(
    rows: list[dict],
    filename: str,
) -> str:
    """Save experimental results to CSV."""
    if not rows:
        raise ValueError("Cannot save an empty collection of rows.")

    path = os.path.join(OUT_DIR, filename)

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV saved → {path}")
    return path


# ──────────────────────────────────────────────────────────────
# 8. Aggregation helper
# ──────────────────────────────────────────────────────────────

def mean_std(
    rows: list[dict],
    key: str,
) -> tuple[float, float]:
    """Return mean and population standard deviation."""
    values = np.array(
        [row[key] for row in rows],
        dtype=float,
    )

    if len(values) == 0:
        raise ValueError("Cannot aggregate an empty collection of rows.")

    return float(np.mean(values)), float(np.std(values, ddof=0))


# ──────────────────────────────────────────────────────────────
# 9. Plot sweep: aggregated results
# ──────────────────────────────────────────────────────────────

def plot_sweep_aggregated(rows: list[dict]) -> None:
    """
    Plot the aggregated effect of varying each centrality weight.
    """
    dimensions = [
        ("ad", r"$a_d$: grado"),
        ("ae", r"$a_e$: vector propio"),
        ("ab", r"$a_b$: intermediación"),
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4.5),
        sharey=True,
    )

    for ax, (dimension, title) in zip(axes, dimensions):
        means: list[float] = []
        stds: list[float] = []

        for value in SWEEP_VALUES:
            filtered = [
                row
                for row in rows
                if row["dimension"] == dimension
                and row["sweep_value"] == value
            ]

            mean_ratio, std_ratio = mean_std(
                filtered,
                key="seed_ratio",
            )

            means.append(mean_ratio)
            stds.append(std_ratio)

        ax.errorbar(
            SWEEP_VALUES,
            means,
            yerr=stds,
            marker="o",
            linewidth=1.8,
            capsize=4,
        )

        ax.axvline(
            1.0,
            linestyle=":",
            linewidth=1,
        )

        ax.set_xlabel("Valor del peso")
        ax.set_title(title)

    axes[0].set_ylabel(r"Media de $|S_0^*|/n$")

    fig.suptitle(
        "Sensibilidad respecto de los pesos de centralidad",
        fontsize=13,
    )

    fig.tight_layout()

    path = os.path.join(
        OUT_DIR,
        "fig_centrality_sweep_aggregated.png",
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)
    print(f"Figure saved → {path}")


# ──────────────────────────────────────────────────────────────
# 10. Plot sweep: results per instance
# ──────────────────────────────────────────────────────────────

def plot_sweep_per_instance(rows: list[dict]) -> None:
    """
    Plot one panel per centrality weight and one line per BA instance.
    """
    dimensions = [
        ("ad", r"$a_d$: grado"),
        ("ae", r"$a_e$: vector propio"),
        ("ab", r"$a_b$: intermediación"),
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 4.8),
        sharey=True,
    )

    for ax, (dimension, title) in zip(axes, dimensions):
        for n, m in INSTANCES:
            means: list[float] = []
            stds: list[float] = []

            for value in SWEEP_VALUES:
                filtered = [
                    row
                    for row in rows
                    if row["dimension"] == dimension
                    and row["sweep_value"] == value
                    and row["n"] == n
                    and row["m"] == m
                ]

                mean_ratio, std_ratio = mean_std(
                    filtered,
                    key="seed_ratio",
                )

                means.append(mean_ratio)
                stds.append(std_ratio)

            ax.errorbar(
                SWEEP_VALUES,
                means,
                yerr=stds,
                marker="o",
                linewidth=1.5,
                capsize=3,
                label=f"n={n}, m={m}",
            )

        ax.axvline(
            1.0,
            linestyle=":",
            linewidth=1,
        )

        ax.set_xlabel("Valor del peso")
        ax.set_title(title)

    axes[0].set_ylabel(r"$|S_0^*|/n$")
    axes[-1].legend()

    fig.suptitle(
        "Sensibilidad respecto de los pesos de centralidad por instancia",
        fontsize=13,
    )

    fig.tight_layout()

    path = os.path.join(
        OUT_DIR,
        "fig_centrality_sweep_per_instance.png",
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)
    print(f"Figure saved → {path}")


# ──────────────────────────────────────────────────────────────
# 11. Plot ablation study
# ──────────────────────────────────────────────────────────────

def plot_ablation(rows: list[dict]) -> None:
    """
    Plot the mean normalised seed ratio for each ablation configuration.
    """
    labels: list[str] = []
    means: list[float] = []
    stds: list[float] = []

    for config_name, _, _, _ in ABLATION_CONFIGS:
        filtered = [
            row
            for row in rows
            if row["config_name"] == config_name
        ]

        mean_ratio, std_ratio = mean_std(
            filtered,
            key="seed_ratio",
        )

        labels.append(config_name)
        means.append(mean_ratio)
        stds.append(std_ratio)

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(12, 5.5))

    ax.bar(
        x,
        means,
        yerr=stds,
        capsize=4,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        rotation=35,
        ha="right",
    )

    ax.set_ylabel(r"Media de $|S_0^*|/n$")
    ax.set_title("Estudio de ablación de las medidas de centralidad")

    fig.tight_layout()

    path = os.path.join(
        OUT_DIR,
        "fig_centrality_ablation.png",
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)
    print(f"Figure saved → {path}")


# ──────────────────────────────────────────────────────────────
# 12. Print summaries
# ──────────────────────────────────────────────────────────────

def print_sweep_summary(rows: list[dict]) -> None:
    """Print aggregated sweep results."""
    print("\n── Centrality sweep summary ────────────────────────────")
    print(
        f'{"dimension":>12}  '
        f'{"value":>7}  '
        f'{"mean ratio":>11}  '
        f'{"std":>8}  '
        f'{"perfect%":>10}'
    )

    print("-" * 58)

    for dimension in ["ad", "ae", "ab"]:
        for value in SWEEP_VALUES:
            filtered = [
                row
                for row in rows
                if row["dimension"] == dimension
                and row["sweep_value"] == value
            ]

            mean_ratio, std_ratio = mean_std(
                filtered,
                key="seed_ratio",
            )

            perfect_rate, _ = mean_std(
                filtered,
                key="is_perfect",
            )

            print(
                f"{dimension:>12}  "
                f"{value:>7.2f}  "
                f"{mean_ratio:>11.4f}  "
                f"{std_ratio:>8.4f}  "
                f"{100 * perfect_rate:>9.1f}%"
            )

    print("-" * 58)


def print_ablation_summary(rows: list[dict]) -> None:
    """Print aggregated ablation results."""
    print("\n── Centrality ablation summary ─────────────────────────")
    print(
        f'{"configuration":>22}  '
        f'{"mean ratio":>11}  '
        f'{"std":>8}  '
        f'{"perfect%":>10}'
    )

    print("-" * 58)

    for config_name, _, _, _ in ABLATION_CONFIGS:
        filtered = [
            row
            for row in rows
            if row["config_name"] == config_name
        ]

        mean_ratio, std_ratio = mean_std(
            filtered,
            key="seed_ratio",
        )

        perfect_rate, _ = mean_std(
            filtered,
            key="is_perfect",
        )

        print(
            f"{config_name:>22}  "
            f"{mean_ratio:>11.4f}  "
            f"{std_ratio:>8.4f}  "
            f"{100 * perfect_rate:>9.1f}%"
        )

    print("-" * 58)


# ──────────────────────────────────────────────────────────────
# 13. Main
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sweep_results = run_sweep_study()

    save_csv(
        sweep_results,
        filename="results_centrality_sweep.csv",
    )

    plot_sweep_aggregated(sweep_results)
    plot_sweep_per_instance(sweep_results)
    print_sweep_summary(sweep_results)

    ablation_results = run_ablation_study()

    save_csv(
        ablation_results,
        filename="results_centrality_ablation.csv",
    )

    plot_ablation(ablation_results)
    print_ablation_summary(ablation_results)

    print("\nDONE")