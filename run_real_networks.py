"""
run_real_networks.py  (v4 — zip/gz/plain support + more Jazz mirrors)
======================================================================
Downloads the 6 real social networks and runs GRASP + SA.

Usage (from the folder containing pap_solver.py):
    python run_real_networks.py

Output:
    results_real_networks_final.csv
    fig_real_networks_comparison.png
    fig_real_networks_ratio.png
"""

import math, os, gzip, time, csv, random, warnings
import urllib.request, zipfile, tarfile, io
warnings.filterwarnings('ignore')

import networkx as nx
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pap_solver import (
    CentralityCache,
    solve_pap,
)

# ── Patch: degree-only centrality for large graphs ────────────
_orig = CentralityCache.compute.__func__

@classmethod
def _fast(cls, G):
    if G.number_of_nodes() > DEGREE_ONLY_THRESHOLD:
        deg = nx.degree_centrality(G)
        return cls(
            degree=deg,
            eigenvector=deg,
            betweenness=deg,
        )

    return _orig(cls, G)

CentralityCache.compute = _fast

# ── Helpers ───────────────────────────────────────────────────
def assign_t(G):
    H = nx.Graph(G)
    lcc = max(nx.connected_components(H), key=len)
    H = H.subgraph(lcc).copy()
    for v in H.nodes():
        H.nodes[v]["threshold"] = max(1, math.ceil(0.5 * H.degree(v)))
    return H

def parse_bytes(data):
    """Parse raw bytes as an edge list into a NetworkX graph."""
    lines = data.decode("utf-8", errors="ignore").splitlines()
    edges = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("%"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                edges.append((int(parts[0]), int(parts[1])))
            except ValueError:
                pass
    G = nx.Graph()
    G.add_edges_from(edges)
    return G

def fetch(url):
    """Download a network file (plain, .gz, .zip, .tar.gz/.bz2) and
    return a NetworkX Graph."""
    print(f"    GET {url}")
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read()

    # .zip
    if url.endswith(".zip") or raw[:2] == b'PK':
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                names = zf.namelist()
                # pick the largest file that looks like an edge list
                txt_names = [n for n in names if not n.endswith('/')
                             and any(n.endswith(e)
                                     for e in ['.txt','.edges',
                                               '.edgelist','.net',
                                               '.dat',''])]
                if not txt_names:
                    txt_names = [n for n in names if not n.endswith('/')]
                txt_names.sort(
                    key=lambda n: zf.getinfo(n).file_size, reverse=True)
                content = zf.read(txt_names[0])
                if txt_names[0].endswith('.gz'):
                    content = gzip.decompress(content)
                return parse_bytes(content)
        except zipfile.BadZipFile:
            pass

    # .tar.bz2 / .tar.gz
    if url.endswith(".tar.bz2") or url.endswith(".tar.gz"):
        mode = "r:bz2" if url.endswith(".bz2") else "r:gz"
        try:
            with tarfile.open(fileobj=io.BytesIO(raw), mode=mode) as tf:
                members = sorted(tf.getmembers(),
                                 key=lambda m: m.size, reverse=True)
                for m in members:
                    if m.isfile():
                        f = tf.extractfile(m)
                        if f:
                            return parse_bytes(f.read())
        except Exception:
            pass

    # plain .gz (not tar)
    if url.endswith(".gz"):
        return parse_bytes(gzip.decompress(raw))

    # plain text
    return parse_bytes(raw)

GRASP_ITER = 8
ALPHA = 0.3
T_INIT = 5.0
T_MIN = 0.01
COOLING = 0.97
STEPS_PER_TEMP = 25
LAM = 2.0

DEGREE_ONLY_THRESHOLD = 500

def run_one(G, seed=42):
    """
    Ejecuta el mismo pipeline GRASP + SA utilizado en los
    experimentos sintéticos.
    """
    start = time.perf_counter()

    result = solve_pap(
        G,
        n_grasp_iter=GRASP_ITER,
        alpha=ALPHA,
        T_init=T_INIT,
        T_min=T_MIN,
        cooling=COOLING,
        steps_per_temp=STEPS_PER_TEMP,
        lam=LAM,
        seed=seed,
    )

    end_to_end_time = time.perf_counter() - start

    return {
        "seed_size": result.seed_size,
        "is_perfect": result.is_perfect,
        "pipeline_time": result.time_total,
        "end_to_end_time": end_to_end_time,
    }

# ════════════════════════════════════════════════════════════════
# Network list — multiple URL candidates tried in order
# ════════════════════════════════════════════════════════════════
NETWORKS = [

    ("Karate Club", [], 3, 10, 3),

    ("Jazz",
     [
         # Arenas lab (original source — zip with .net Pajek file)
         "http://deim.urv.cat/~alexandre.arenas/data/xarxes/jazz.zip",
         # networkrepository plain txt
         "https://nrvis.com/download/data/misc/misc-jazz.zip",
         # Direct plain-text mirrors
         "https://raw.githubusercontent.com/KarenYng/network_analysis/"
         "master/data/jazz_edges.txt",
         "https://raw.githubusercontent.com/briatte/awesome-network-analysis/"
         "master/data/jazz.tsv",
         # Konect bz2
         "http://konect.cc/files/download.tsv.jazz-musicians.tar.bz2",
     ],
     15, 6, 3),

    ("Facebook",
     [
         "https://snap.stanford.edu/data/facebook_combined.txt.gz",
         "https://snap.stanford.edu/data/facebook_combined.txt",
     ],
     10, 3, 1),

    ("Power grid",
     [
         # Watts & Strogatz 1998 — hosted on various mirrors
         "https://raw.githubusercontent.com/gephi/gephi-toolkit-demos/"
         "master/src/main/resources/org/gephi/toolkit/demos/power.gml",
         # networkrepository
         "https://nrvis.com/download/data/inf/inf-power.zip",
         # Mark Newman's site (University of Michigan)
         "http://www-personal.umich.edu/~mejn/netdata/power.zip",
     ],
     1367, 3, 1),

    ("CA-GrQc",
     [
         "https://snap.stanford.edu/data/ca-GrQc.txt.gz",
         "https://snap.stanford.edu/data/ca-GrQc.txt",
     ],
     897, 3, 1),

    ("CA-HepTh",
     [
         "https://snap.stanford.edu/data/ca-HepTh.txt.gz",
         "https://snap.stanford.edu/data/ca-HepTh.txt",
     ],
     1531, 3, 1),
]

# ════════════════════════════════════════════════════════════════
# Run
# ════════════════════════════════════════════════════════════════
print("=" * 60)
print("PAP solver — real networks benchmark")
print("=" * 60)

rows = []

for name, urls, pereira_gr, n_iter, nruns in NETWORKS:
    print(f"\n[{name}]")

    if name == "Karate Club":
        G_raw = nx.karate_club_graph()
        G = assign_t(G_raw)

        print(
            f"    Built-in: "
            f"n={G.number_of_nodes()} "
            f"|E|={G.number_of_edges()}"
        )

    else:
        G = None

        for url in urls:
            try:
                G_raw = fetch(url)

                if G_raw.number_of_nodes() < 10:
                    print(
                        f"    Too small "
                        f"({G_raw.number_of_nodes()} nodes), "
                        f"skipping."
                    )
                    continue

                G = assign_t(G_raw)

                print(
                    f"    OK: "
                    f"n={G.number_of_nodes()}  "
                    f"|E|={G.number_of_edges()}"
                )

                break

            except Exception as error:
                print(f"    FAIL: {error}")

        if G is None:
            print(
                f"  All URLs failed for {name}. "
                f"Skipping."
            )
            continue

    n = G.number_of_nodes()
    e = G.number_of_edges()

    dens = (
        2 * e / (n * (n - 1))
        if n > 1
        else 0
    )

    centrality_mode = (
        "degree_only"
        if n > DEGREE_ONLY_THRESHOLD
        else "degree_eigenvector_betweenness"
    )

    seeds_list = []
    times = []
    perfect_list = []

    for run in range(nruns):
        result = run_one(
            G,
            seed=42 + run,
        )

        seed_size = result["seed_size"]
        is_perfect = result["is_perfect"]
        elapsed = result["end_to_end_time"]

        seeds_list.append(seed_size)
        times.append(elapsed)
        perfect_list.append(is_perfect)

        print(
            f"  run {run + 1}/{nruns}: "
            f"|S*|={seed_size}  "
            f"t={elapsed:.1f}s  "
            f"perfect={is_perfect}"
        )

    valid_seeds = [
        seed_size
        for seed_size, is_perfect
        in zip(seeds_list, perfect_list)
        if is_perfect
    ]

    if not valid_seeds:
        print(
            f"  No perfect solutions found for {name}. "
            f"Skipping."
        )
        continue

    best = min(valid_seeds)
    mean_t = round(float(np.mean(times)), 2)

    perfect_rate = round(
        100 * float(np.mean(perfect_list)),
        2,
    )

    delta = best - pereira_gr

    rows.append(
        {
            "network": name,
            "n": n,
            "edges": e,
            "density": round(dens, 6),
            "centrality_mode": centrality_mode,
            "best_seed": best,
            "seed_ratio": round(best / n, 4),
            "mean_time": mean_t,
            "runs": nruns,
            "perfect_rate_pct": perfect_rate,
            "pereira_gr": pereira_gr,
            "delta": delta,
        }
    )


if not rows:
    print("\nNo results produced.")
    sys.exit(1)

# ── Save CSV ──────────────────────────────────────────────────
with open("results_real_networks_final.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)
print("\nSaved results_real_networks_final.csv")

# ── Print table ───────────────────────────────────────────────
print()
print(f"{'Network':15s} {'|V|':>7} {'|E|':>8} "
      f"{'|S*|':>6} {'|S*|/n':>7} {'Pereira':>8} "
      f"{'Delta':>7} {'Time':>7}")
print("-" * 72)
for row in rows:
    sgn = "+" if row["delta"] > 0 else ""
    print(f"{row['network']:15s} {row['n']:>7,} {row['edges']:>8,} "
          f"{row['best_seed']:>6} {row['seed_ratio']:>7.3f} "
          f"{row['pereira_gr']:>8} {sgn}{row['delta']:>6} "
          f"{row['mean_time']:>6.1f}s")

# ── Figures ───────────────────────────────────────────────────
STYLE = {
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.4,
    "grid.linestyle": "--", "grid.color": "#CCCCCC",
    "font.size": 11, "axes.titlesize": 12,
    "axes.labelsize": 11, "legend.fontsize": 9,
    "figure.dpi": 150,
    "axes.facecolor": "white", "figure.facecolor": "white",
}
PAL = plt.cm.tab10.colors

names_l = [r["network"]    for r in rows]
ours_l  = [r["best_seed"]  for r in rows]
per_l   = [r["pereira_gr"] for r in rows]
x = np.arange(len(names_l))
W = 0.35

# Fig 1 — bar chart comparison
with plt.rc_context(STYLE):
    fig, ax = plt.subplots(figsize=(11, 5))
    b1 = ax.bar(x - W/2, ours_l, W,
                label="GRASP + SA (este trabajo)",
                color="#2196F3", alpha=0.85)
    b2 = ax.bar(x + W/2, per_l, W,
                label="GR — Pereira et al. [1]",
                color="#FF7043", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names_l, rotation=15, ha="right")
    ax.set_ylabel("$|S_0^*|$")
    ax.set_title("Tamaño del conjunto semilla perfecto: "
                 "comparativa con Pereira et al. [1]")
    ax.legend()
    scale = max(ours_l + per_l) * 0.012
    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + scale,
                str(int(h)), ha="center", va="bottom",
                fontsize=8, color="#1565C0")
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + scale,
                str(int(h)), ha="center", va="bottom",
                fontsize=8, color="#BF360C")
    fig.tight_layout()
    fig.savefig("fig_real_networks_comparison.png",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved fig_real_networks_comparison.png")

# Fig 2 — ratio vs density scatter
with plt.rc_context(STYLE):
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, row in enumerate(rows):
        ax.scatter(row["density"], row["seed_ratio"],
                   s=140, color=PAL[i], zorder=4,
                   label=f"{row['network']}  (|S*|={row['best_seed']})")
        ax.annotate(row["network"],
                    (row["density"], row["seed_ratio"]),
                    textcoords="offset points", xytext=(6, 3),
                    fontsize=8.5, color=PAL[i])
    ax.set_xlabel(r"Densidad  $\frac{2|E|}{|V|(|V|-1)}$")
    ax.set_ylabel(r"$|S_0^*|\,/\,n$")
    ax.set_title(r"Ratio de semilla $|S_0^*|/n$ vs. densidad — redes reales")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig("fig_real_networks_ratio.png",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved fig_real_networks_ratio.png")

print("\nAll done.")
