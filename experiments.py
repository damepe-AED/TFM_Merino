"""
experiments.py
==============
Runs the PAP solver over a grid of Barabási–Albert parameters,
collects results and produces all plots needed for the thesis.

Usage
-----
    python experiments.py

Output files (written to the same directory)
--------------------------------------------
    results_summary.csv          - one row per (n, m, run)
    fig_seed_vs_n.png            - |S*| as a function of n  (fixed m)
    fig_seed_vs_m.png            - |S*| as a function of m  (fixed n)
    fig_seed_vs_density.png      - |S*| / n  vs. graph density
    fig_time_vs_n.png            - wall-clock time vs. n
    fig_sa_convergence.png       - SA energy history for one instance
    fig_seed_ratio_heatmap.png   - |S*|/n heatmap over (n, m)
"""

import csv
import os
import random
import warnings
from itertools import product

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from pap_solver import generate_ba_graph, solve_pap

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# Experiment grid
# ──────────────────────────────────────────────────────────────
N_VALUES  = [20, 40, 60, 80, 100, 150, 200, 500, 1000, 3000, 5000]   # number of nodes
M_VALUES  = [1, 2, 3, 4, 5]                   # BA attachment parameter
N_RUNS    = 5                                  # independent runs per (n, m)

# Fixed values for univariate plots
FIXED_M   = 2
FIXED_N   = 100

# SA / GRASP hyper-parameters
GRASP_ITER     = 8
ALPHA          = 0.3
T_INIT         = 5.0
T_MIN          = 0.01
COOLING        = 0.97
STEPS_PER_TEMP = 25
LAM            = 2.0

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ──────────────────────────────────────────────────────────────
# Helper: run one experiment
# ──────────────────────────────────────────────────────────────

def run_experiment(n: int, m: int, run_id: int) -> dict:
    seed = 1000 * n + 10 * m + run_id
    G    = generate_ba_graph(n, m, seed=seed)
    res  = solve_pap(
        G,
        n_grasp_iter   = GRASP_ITER,
        alpha          = ALPHA,
        T_init         = T_INIT,
        T_min          = T_MIN,
        cooling        = COOLING,
        steps_per_temp = STEPS_PER_TEMP,
        lam            = LAM,
        seed           = seed,
    )
    return {
        "n":          n,
        "m":          m,
        "run":        run_id,
        "seed_size":  res.seed_size,
        "seed_ratio": res.seed_size / n,
        "is_perfect": int(res.is_perfect),
        "avg_degree": round(res.avg_degree, 4),
        "density":    round(res.density, 6),
        "edges":      res.graph_edges,
        "time_grasp": round(res.time_grasp, 4),
        "time_sa":    round(res.time_sa, 4),
        "time_total": round(res.time_total, 4),
        "sa_history": res.sa_history,   # kept in memory, not in CSV
    }


# ──────────────────────────────────────────────────────────────
# Run all experiments
# ──────────────────────────────────────────────────────────────

print("Running experiments …")
print(f"  Grid: n ∈ {N_VALUES},  m ∈ {M_VALUES},  {N_RUNS} runs each")
print(f"  Total instances: {len(N_VALUES) * len(M_VALUES) * N_RUNS}\n")

all_results = []
total = len(N_VALUES) * len(M_VALUES) * N_RUNS
done  = 0

for n, m in product(N_VALUES, M_VALUES):
    for run in range(N_RUNS):
        r = run_experiment(n, m, run)
        all_results.append(r)
        done += 1
        if done % 10 == 0 or done == total:
            print(f"  [{done:3d}/{total}]  n={n:4d}  m={m}  "
                  f"seed={r['seed_size']:3d}  "
                  f"perfect={'✓' if r['is_perfect'] else '✗'}  "
                  f"t={r['time_total']:.2f}s")

print("\nAll experiments done.\n")


# ──────────────────────────────────────────────────────────────
# Save CSV
# ──────────────────────────────────────────────────────────────

csv_path = os.path.join(OUT_DIR, "results_summary.csv")
csv_keys  = [k for k in all_results[0].keys() if k != "sa_history"]
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=csv_keys)
    writer.writeheader()
    for r in all_results:
        writer.writerow({k: r[k] for k in csv_keys})

print(f"Results saved → {csv_path}\n")


# ──────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────

STYLE = {
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.35,
    "grid.linestyle":     "--",
    "font.size":          11,
    "axes.titlesize":     13,
    "axes.labelsize":     12,
    "legend.fontsize":    10,
    "figure.dpi":         150,
}

COLORS = plt.cm.tab10.colors   # type: ignore[attr-defined]


def get_mean_std(results, key_x, val_x, key_y, key_filter=None, val_filter=None):
    """Filter rows and return (mean, std) of key_y where key_x == val_x."""
    rows = [r for r in results if r[key_x] == val_x]
    if key_filter is not None:
        rows = [r for r in rows if r[key_filter] == val_filter]
    vals = [r[key_y] for r in rows]
    return np.mean(vals), np.std(vals)


# ──────────────────────────────────────────────────────────────
# Fig 1 – Seed size vs. n  (one line per m value)
# ──────────────────────────────────────────────────────────────

with plt.rc_context(STYLE):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, m in enumerate(M_VALUES):
        means, stds = [], []
        for n in N_VALUES:
            mu, sigma = get_mean_std(all_results, "n", n, "seed_size",
                                     key_filter="m", val_filter=m)
            means.append(mu); stds.append(sigma)
        ax.errorbar(N_VALUES, means, yerr=stds,
                    label=f"m = {m}", marker="o", linewidth=1.6,
                    color=COLORS[i], capsize=3)
    ax.set_xlabel("Number of nodes $n$")
    ax.set_ylabel("Seed set size $|S_0^*|$")
    ax.set_title("Perfect seed set size as a function of $n$\n"
                 "(Barabási–Albert networks, majority threshold)")
    ax.legend(title="Attachment\nparameter $m$")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig_seed_vs_n.png")
    fig.savefig(path); plt.close(fig)
    print(f"Saved → {path}")


# ──────────────────────────────────────────────────────────────
# Fig 2 – Seed size vs. m  (one line per n value)
# ──────────────────────────────────────────────────────────────

with plt.rc_context(STYLE):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, n in enumerate(N_VALUES):
        means, stds = [], []
        for m in M_VALUES:
            mu, sigma = get_mean_std(all_results, "m", m, "seed_size",
                                     key_filter="n", val_filter=n)
            means.append(mu); stds.append(sigma)
        ax.errorbar(M_VALUES, means, yerr=stds,
                    label=f"n = {n}", marker="s", linewidth=1.6,
                    color=COLORS[i % 10], capsize=3)
    ax.set_xlabel("Attachment parameter $m$")
    ax.set_ylabel("Seed set size $|S_0^*|$")
    ax.set_title("Perfect seed set size as a function of $m$\n"
                 "(Barabási–Albert networks, majority threshold)")
    ax.legend(title="Number of\nnodes $n$", ncol=2)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig_seed_vs_m.png")
    fig.savefig(path); plt.close(fig)
    print(f"Saved → {path}")


# ──────────────────────────────────────────────────────────────
# Fig 3 – Normalised seed ratio vs. density
# ──────────────────────────────────────────────────────────────

with plt.rc_context(STYLE):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    densities   = [r["density"]    for r in all_results]
    seed_ratios = [r["seed_ratio"] for r in all_results]
    m_vals      = [r["m"]          for r in all_results]

    scatter = ax.scatter(densities, seed_ratios,
                         c=m_vals, cmap="tab10", alpha=0.5,
                         s=30, vmin=min(M_VALUES), vmax=max(M_VALUES))
    cbar = fig.colorbar(scatter, ax=ax, ticks=M_VALUES)
    cbar.set_label("Attachment parameter $m$")

    # Trend line
    z = np.polyfit(densities, seed_ratios, 1)
    xs = np.linspace(min(densities), max(densities), 200)
    ax.plot(xs, np.polyval(z, xs), "k--", linewidth=1.2, label="Linear fit")
    ax.legend()
    ax.set_xlabel("Graph density")
    ax.set_ylabel("$|S_0^*| / n$")
    ax.set_title("Normalised seed ratio vs. graph density")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig_seed_vs_density.png")
    fig.savefig(path); plt.close(fig)
    print(f"Saved → {path}")


# ──────────────────────────────────────────────────────────────
# Fig 4 – Computation time vs. n
# ──────────────────────────────────────────────────────────────

with plt.rc_context(STYLE):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, m in enumerate(M_VALUES):
        means_g, means_s = [], []
        for n in N_VALUES:
            rows = [r for r in all_results if r["n"] == n and r["m"] == m]
            means_g.append(np.mean([r["time_grasp"] for r in rows]))
            means_s.append(np.mean([r["time_sa"]    for r in rows]))
        total = [g + s for g, s in zip(means_g, means_s)]
        ax.plot(N_VALUES, total, marker="o", label=f"m={m}",
                color=COLORS[i], linewidth=1.6)
    ax.set_xlabel("Number of nodes $n$")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("Total computation time vs. $n$\n(GRASP + SA pipeline)")
    ax.legend(title="$m$")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig_time_vs_n.png")
    fig.savefig(path); plt.close(fig)
    print(f"Saved → {path}")


# ──────────────────────────────────────────────────────────────
# Fig 5 – SA convergence for one representative instance
# ──────────────────────────────────────────────────────────────

rep = next(r for r in all_results
           if r["n"] == FIXED_N and r["m"] == FIXED_M and r["run"] == 0)

with plt.rc_context(STYLE):
    fig, ax = plt.subplots(figsize=(7, 4))
    hist = rep["sa_history"]
    ax.plot(range(len(hist)), hist, color="steelblue", linewidth=1.6)
    ax.set_xlabel("Temperature step")
    ax.set_ylabel("Energy $E(S_0)$")
    ax.set_title(f"SA energy convergence  "
                 f"(n={FIXED_N}, m={FIXED_M}, run 0)\n"
                 f"Final seed size = {rep['seed_size']}")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig_sa_convergence.png")
    fig.savefig(path); plt.close(fig)
    print(f"Saved → {path}")


# ──────────────────────────────────────────────────────────────
# Fig 6 – Heatmap of normalised seed ratio over (n, m)
# ──────────────────────────────────────────────────────────────

matrix = np.zeros((len(N_VALUES), len(M_VALUES)))
for i, n in enumerate(N_VALUES):
    for j, m in enumerate(M_VALUES):
        rows = [r for r in all_results if r["n"] == n and r["m"] == m]
        matrix[i, j] = np.mean([r["seed_ratio"] for r in rows])

with plt.rc_context(STYLE):
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd",
                   origin="lower", interpolation="nearest")
    fig.colorbar(im, ax=ax, label="$|S_0^*| / n$")
    ax.set_xticks(range(len(M_VALUES))); ax.set_xticklabels(M_VALUES)
    ax.set_yticks(range(len(N_VALUES))); ax.set_yticklabels(N_VALUES)
    ax.set_xlabel("Attachment parameter $m$")
    ax.set_ylabel("Number of nodes $n$")
    ax.set_title("Heatmap of normalised seed ratio $|S_0^*|/n$\n"
                 "over BA network parameters")
    # Annotate cells
    for i in range(len(N_VALUES)):
        for j in range(len(M_VALUES)):
            ax.text(j, i, f"{matrix[i,j]:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="black" if matrix[i,j] < 0.5 else "white")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig_seed_ratio_heatmap.png")
    fig.savefig(path); plt.close(fig)
    print(f"Saved → {path}")


# ──────────────────────────────────────────────────────────────
# Summary statistics
# ──────────────────────────────────────────────────────────────

print("\n── Summary ──────────────────────────────────────────────")
perfect_pct = 100 * np.mean([r["is_perfect"] for r in all_results])
avg_ratio   = np.mean([r["seed_ratio"]  for r in all_results])
avg_time    = np.mean([r["time_total"]  for r in all_results])
print(f"  Perfect solutions found : {perfect_pct:.1f} %")
print(f"  Mean |S*| / n           : {avg_ratio:.3f}")
print(f"  Mean total time         : {avg_time:.3f} s")
print("─────────────────────────────────────────────────────────\n")
print("All figures saved.  Done.")
