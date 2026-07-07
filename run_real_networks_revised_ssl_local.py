"""
run_real_networks_revised.py
============================
Validation on real networks for the Perfect Awareness Problem (PAP/PCP).

This version is designed to address the methodological comments about the
real-network comparison:

1. It runs several independent executions per network and stores all raw runs.
2. It reports both the best value and the mean value over the executions.
3. It distinguishes directly comparable, component-filtered and unverified cases.
4. It independently checks the downloaded graph sizes against expected metadata
   when those values are provided.
5. It does not attribute differences with the reference values to Simulated
   Annealing alone; the comparison is made for the complete final pipeline.
6. It separates graph loading, preprocessing, centrality computation and algorithm time.

Usage, from the folder containing pap_solver.py:
    python run_real_networks_revised.py

Outputs:
    results_real_networks_raw.csv
    results_real_networks_summary.csv
    table_real_networks_summary.tex
    fig_real_networks_comparison_best.png
    fig_real_networks_ratio_density.png
"""

import csv
import gzip
import io
import math
import os
import ssl
import random
import sys
import tarfile
import time
import urllib.request
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pap_solver import (  # noqa: E402
    CentralityCache,
    grasp_construct,
    is_perfect_seed,
    refine,
    simulated_annealing,
)


# ──────────────────────────────────────────────────────────────
# 1. Final algorithm configuration
# ──────────────────────────────────────────────────────────────

GRASP_ITER = 8
ALPHA = 0.7
AD = 1.0
AE = 0.0
AB = 0.0

T_INIT = 5.0
T_MIN = 0.01
COOLING = 0.97
STEPS_PER_TEMP = 25
LAM = 1.5

# Number of independent executions per real network.
# If runtime becomes excessive, this can be reduced, but the value used must be
# reported in the thesis.
N_RUNS_REAL = 5
BASE_ALGORITHM_SEED = 42

OUT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = OUT_DIR / "real_networks_data"
SSL_CONTEXT = ssl._create_unverified_context()
RAW_OUTPUT = OUT_DIR / "results_real_networks_raw.csv"
SUMMARY_OUTPUT = OUT_DIR / "results_real_networks_summary.csv"
LATEX_OUTPUT = OUT_DIR / "table_real_networks_summary.tex"


# ──────────────────────────────────────────────────────────────
# 2. Data structures
# ──────────────────────────────────────────────────────────────

@dataclass
class NetworkSpec:
    name: str
    urls: List[str]
    reference_value: Optional[int]
    expected_raw_n: Optional[int] = None
    expected_raw_edges: Optional[int] = None
    runs: int = N_RUNS_REAL
    note: str = ""


# Expected raw sizes are used only as an independent check of the downloaded data.
# If a source changes or a mirror contains a different format, the script will flag it.
NETWORKS: List[NetworkSpec] = [
    NetworkSpec(
        name="Karate Club",
        urls=[],
        reference_value=3,
        expected_raw_n=34,
        expected_raw_edges=78,
        runs=N_RUNS_REAL,
        note="NetworkX built-in graph.",
    ),
    NetworkSpec(
        name="Jazz",
        urls=[
            "real_networks_data/jazz.zip",
            "real_networks_data/misc-jazz.zip",
            "real_networks_data/jazz_edges.txt",
            "real_networks_data/jazz.tsv",
            "http://deim.urv.cat/~alexandre.arenas/data/xarxes/jazz.zip",
            "https://nrvis.com/download/data/misc/misc-jazz.zip",
            "https://raw.githubusercontent.com/KarenYng/network_analysis/master/data/jazz_edges.txt",
            "https://raw.githubusercontent.com/briatte/awesome-network-analysis/master/data/jazz.tsv",
            "http://konect.cc/files/download.tsv.jazz-musicians.tar.bz2",
        ],
        reference_value=15,
        expected_raw_n=198,
        expected_raw_edges=2742,
        runs=N_RUNS_REAL,
        note="Jazz musicians network.",
    ),
    NetworkSpec(
        name="Facebook",
        urls=[
            "real_networks_data/facebook_combined.txt.gz",
            "real_networks_data/facebook_combined.txt",
            "https://snap.stanford.edu/data/facebook_combined.txt.gz",
            "https://snap.stanford.edu/data/facebook_combined.txt",
        ],
        reference_value=10,
        expected_raw_n=4039,
        expected_raw_edges=88234,
        runs=N_RUNS_REAL,
        note="SNAP Facebook combined ego network. Check especially if results are very different from the reference.",
    ),
    NetworkSpec(
        name="Power grid",
        urls=[
            "real_networks_data/power.gml",
            "real_networks_data/power.zip",
            "real_networks_data/inf-power.zip",
            "https://raw.githubusercontent.com/gephi/gephi-toolkit-demos/master/src/main/resources/org/gephi/toolkit/demos/power.gml",
            "https://nrvis.com/download/data/inf/inf-power.zip",
            "http://www-personal.umich.edu/~mejn/netdata/power.zip",
        ],
        reference_value=1367,
        expected_raw_n=4941,
        expected_raw_edges=6594,
        runs=N_RUNS_REAL,
        note="Power grid network. Verify independently if the obtained value is especially different from the reference.",
    ),
    NetworkSpec(
        name="CA-GrQc",
        urls=[
            "real_networks_data/ca-GrQc.txt.gz",
            "real_networks_data/ca-GrQc.txt",
            "https://snap.stanford.edu/data/ca-GrQc.txt.gz",
            "https://snap.stanford.edu/data/ca-GrQc.txt",
        ],
        reference_value=897,
        expected_raw_n=5242,
        expected_raw_edges=14496,
        runs=N_RUNS_REAL,
        note="SNAP collaboration network. The script uses the largest connected component for the PAP run.",
    ),
    NetworkSpec(
        name="CA-HepTh",
        urls=[
            "real_networks_data/ca-HepTh.txt.gz",
            "real_networks_data/ca-HepTh.txt",
            "https://snap.stanford.edu/data/ca-HepTh.txt.gz",
            "https://snap.stanford.edu/data/ca-HepTh.txt",
        ],
        reference_value=1531,
        expected_raw_n=9877,
        expected_raw_edges=25998,
        runs=N_RUNS_REAL,
        note="SNAP collaboration network. The script uses the largest connected component for the PAP run.",
    ),
]


# ──────────────────────────────────────────────────────────────
# 3. Graph loading and parsing
# ──────────────────────────────────────────────────────────────

def parse_edge_list_bytes(data: bytes) -> nx.Graph:
    """Parse raw bytes as a simple edge list."""
    lines = data.decode("utf-8", errors="ignore").splitlines()
    edges: List[Tuple[int, int]] = []

    matrix_market = False
    skipped_matrix_size_line = False

    for line in lines:
        line = line.strip()

        if not line:
            continue

        if line.startswith("%%MatrixMarket"):
            matrix_market = True
            continue

        if line.startswith("#") or line.startswith("%") or line.startswith("*"):
            continue

        parts = line.replace(",", " ").split()

        if len(parts) < 2:
            continue

        try:
            nums = [int(float(p)) for p in parts[:3]]
        except ValueError:
            continue

        # MatrixMarket files usually contain one numeric size line: n n m.
        if matrix_market and not skipped_matrix_size_line and len(nums) >= 3:
            skipped_matrix_size_line = True
            continue

        u, v = nums[0], nums[1]
        if u != v:
            edges.append((u, v))

    G = nx.Graph()
    G.add_edges_from(edges)
    return G


def parse_gml_bytes(data: bytes) -> nx.Graph:
    """Parse GML bytes using NetworkX."""
    text = data.decode("utf-8", errors="ignore")
    try:
        return nx.parse_gml(text.splitlines(), label="id")
    except Exception:
        try:
            return nx.parse_gml(text.splitlines())
        except Exception:
            return parse_edge_list_bytes(data)


def parse_bytes(data: bytes, source_name: str = "") -> nx.Graph:
    """Parse bytes as GML or edge list depending on the content/name."""
    head = data[:300].decode("utf-8", errors="ignore").lower()

    if source_name.lower().endswith(".gml") or "graph [" in head:
        return parse_gml_bytes(data)

    return parse_edge_list_bytes(data)


def choose_file_from_archive(names: Iterable[str]) -> Optional[str]:
    """Choose the most likely graph file from an archive."""
    candidates = [n for n in names if not n.endswith("/")]

    preferred_ext = (
        ".txt", ".edges", ".edgelist", ".net", ".dat", ".mtx", ".gml", ".tsv"
    )

    preferred = [n for n in candidates if n.lower().endswith(preferred_ext)]
    if preferred:
        return preferred[0]

    return candidates[0] if candidates else None


def resolve_local_path(path_text: str) -> Optional[Path]:
    """Return an existing local path if path_text points to a local data file."""
    path = Path(path_text)

    candidates = []

    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(OUT_DIR / path)
        candidates.append(DATA_DIR / path.name)

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    return None


def read_url_or_local(url: str) -> Tuple[bytes, str]:
    """Read bytes from a local file first, otherwise download with a permissive SSL context."""
    local_path = resolve_local_path(url)

    if local_path is not None:
        print(f"    LOCAL {local_path}")
        return local_path.read_bytes(), str(local_path)

    print(f"    GET {url}")

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "*/*",
            "Connection": "close",
        },
    )

    # Some Windows/Conda setups fail with ASN1/SSL errors when urllib uses the
    # default certificate context. The custom context avoids that download issue.
    if url.lower().startswith("https://"):
        with urllib.request.urlopen(req, timeout=180, context=SSL_CONTEXT) as resp:
            return resp.read(), url

    with urllib.request.urlopen(req, timeout=180) as resp:
        return resp.read(), url


def fetch_graph(url: str) -> Tuple[nx.Graph, str]:
    """Download or read a graph file and return the raw graph and the file used."""
    raw, source_used = read_url_or_local(url)
    lower_source = source_used.lower()

    # zip archive
    if lower_source.endswith(".zip") or raw[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            names = zf.namelist()
            chosen = choose_file_from_archive(names)
            if chosen is None:
                raise ValueError("No file found in ZIP archive.")
            content = zf.read(chosen)
            if chosen.lower().endswith(".gz"):
                content = gzip.decompress(content)
            return parse_bytes(content, chosen), f"{source_used}::{chosen}"

    # tar archives
    if lower_source.endswith(".tar.bz2") or lower_source.endswith(".tar.gz"):
        mode = "r:bz2" if lower_source.endswith(".bz2") else "r:gz"
        with tarfile.open(fileobj=io.BytesIO(raw), mode=mode) as tf:
            members = [m for m in tf.getmembers() if m.isfile()]
            members.sort(key=lambda m: m.size, reverse=True)
            if not members:
                raise ValueError("No file found in TAR archive.")
            member = members[0]
            f = tf.extractfile(member)
            if f is None:
                raise ValueError("Could not extract file from TAR archive.")
            return parse_bytes(f.read(), member.name), f"{source_used}::{member.name}"

    # gz plain file
    if lower_source.endswith(".gz"):
        return parse_bytes(gzip.decompress(raw), lower_source[:-3]), source_used

    # plain text or GML
    return parse_bytes(raw, source_used), source_used


def assign_majority_threshold_lcc(G: nx.Graph) -> nx.Graph:
    """
    Convert to a simple undirected graph, keep the largest connected component,
    and assign the majority threshold t(v)=ceil(0.5*d(v)).
    """
    H = nx.Graph(G)
    H.remove_edges_from(nx.selfloop_edges(H))

    if H.number_of_nodes() == 0:
        raise ValueError("The graph has no nodes after parsing.")

    if not nx.is_connected(H):
        lcc = max(nx.connected_components(H), key=len)
        H = H.subgraph(lcc).copy()

    for v in H.nodes():
        H.nodes[v]["threshold"] = max(1, math.ceil(0.5 * H.degree(v)))

    return H


def load_network(spec: NetworkSpec) -> Tuple[nx.Graph, Dict[str, object]]:
    """Load a network and return the processed graph plus metadata."""
    t0 = time.perf_counter()

    if spec.name == "Karate Club":
        raw_graph = nx.karate_club_graph()
        source_used = "networkx.karate_club_graph"
    else:
        raw_graph = None
        source_used = ""

        for url in spec.urls:
            try:
                candidate_graph, used = fetch_graph(url)

                if candidate_graph.number_of_nodes() < 10:
                    print(
                        f"    Too small ({candidate_graph.number_of_nodes()} nodes), skipping."
                    )
                    continue

                raw_graph = candidate_graph
                source_used = used
                break

            except Exception as error:
                print(f"    FAIL: {error}")

        if raw_graph is None:
            raise RuntimeError(f"All URLs failed for {spec.name}.")

    raw_n = raw_graph.number_of_nodes()
    raw_edges = raw_graph.number_of_edges()

    processed_graph = assign_majority_threshold_lcc(raw_graph)

    proc_n = processed_graph.number_of_nodes()
    proc_edges = processed_graph.number_of_edges()

    load_time = time.perf_counter() - t0

    expected_available = (
        spec.expected_raw_n is not None and spec.expected_raw_edges is not None
    )

    if expected_available:
        data_verified = (
            raw_n == spec.expected_raw_n and raw_edges == spec.expected_raw_edges
        )
    else:
        data_verified = None

    component_reduction = (raw_n != proc_n or raw_edges != proc_edges)

    if data_verified is True and not component_reduction:
        comparison_class = "direct"
    elif data_verified is True and component_reduction:
        comparison_class = "component_filtered"
    elif data_verified is False:
        comparison_class = "metadata_mismatch"
    else:
        comparison_class = "unverified"

    metadata = {
        "source_used": source_used,
        "raw_n": raw_n,
        "raw_edges": raw_edges,
        "n": proc_n,
        "edges": proc_edges,
        "component_reduction": int(component_reduction),
        "expected_raw_n": spec.expected_raw_n,
        "expected_raw_edges": spec.expected_raw_edges,
        "data_verified": data_verified,
        "comparison_class": comparison_class,
        "load_preprocess_time_s": load_time,
    }

    return processed_graph, metadata


# ──────────────────────────────────────────────────────────────
# 4. Final pipeline with separate timing
# ──────────────────────────────────────────────────────────────

def compute_centralities_for_weights(
    G: nx.Graph,
    ad: float,
    ae: float,
    ab: float,
) -> Tuple[CentralityCache, float, str]:
    """Compute only the centralities required by the selected weights."""
    t0 = time.perf_counter()

    try:
        centralities = CentralityCache.compute(
            G,
            use_degree=(ad != 0),
            use_eigenvector=(ae != 0),
            use_betweenness=(ab != 0),
        )
        mode = "conditional"
    except TypeError:
        centralities = CentralityCache.compute(G)
        mode = "fallback_all"

    elapsed = time.perf_counter() - t0
    elapsed = float(getattr(centralities, "time_total", elapsed))

    if ad != 0 and ae == 0 and ab == 0:
        centrality_mode = "degree_only" if mode == "conditional" else "fallback_all"
    elif ad != 0 and ae != 0 and ab != 0:
        centrality_mode = "degree_eigenvector_betweenness"
    else:
        centrality_mode = f"custom_ad{ad}_ae{ae}_ab{ab}"

    return centralities, elapsed, centrality_mode


def call_simulated_annealing_no_final_refinement(
    G: nx.Graph,
    initial_seed: Set[int],
    rng: random.Random,
) -> Set[int]:
    """Call simulated_annealing while remaining compatible with older signatures."""
    try:
        output = simulated_annealing(
            G,
            initial_seed,
            T_init=T_INIT,
            T_min=T_MIN,
            cooling=COOLING,
            steps_per_temp=STEPS_PER_TEMP,
            lam=LAM,
            rng=rng,
            apply_final_refinement=False,
        )
    except TypeError:
        output = simulated_annealing(
            G,
            initial_seed,
            T_init=T_INIT,
            T_min=T_MIN,
            cooling=COOLING,
            steps_per_temp=STEPS_PER_TEMP,
            lam=LAM,
            rng=rng,
        )

    if isinstance(output, tuple):
        return set(output[0])

    return set(output)


def run_final_pipeline(G: nx.Graph, seed: int) -> Dict[str, object]:
    """
    Execute the final GRASP + refinement + SA + refinement pipeline.

    This function mirrors the final configuration of the thesis, but times the
    phases separately and computes only the centralities required by the final
    centrality weights.
    """
    rng = random.Random(seed)

    t_total_0 = time.perf_counter()

    centralities, time_centrality, centrality_mode = compute_centralities_for_weights(
        G,
        ad=AD,
        ae=AE,
        ab=AB,
    )

    t0 = time.perf_counter()
    raw_solutions: List[Set[int]] = []

    for _ in range(GRASP_ITER):
        raw_solutions.append(
            grasp_construct(
                G,
                centralities,
                alpha=ALPHA,
                ad=AD,
                ae=AE,
                ab=AB,
                rng=rng,
            )
        )

    time_constructive = time.perf_counter() - t0

    t0 = time.perf_counter()
    refined_solutions = [refine(G, sol) for sol in raw_solutions]
    best_grasp = set(min(refined_solutions, key=len))
    time_initial_refinement = time.perf_counter() - t0

    t0 = time.perf_counter()
    best_sa = call_simulated_annealing_no_final_refinement(
        G,
        initial_seed=best_grasp,
        rng=rng,
    )
    time_sa = time.perf_counter() - t0

    t0 = time.perf_counter()
    final_solution = refine(G, best_sa)
    time_final_refinement = time.perf_counter() - t0

    time_algorithm_excl_preprocessing = (
        time_constructive + time_initial_refinement + time_sa + time_final_refinement
    )

    time_total = time.perf_counter() - t_total_0

    return {
        "seed_size": len(final_solution),
        "seed_ratio": len(final_solution) / G.number_of_nodes(),
        "is_perfect": int(is_perfect_seed(G, final_solution)),
        "centrality_mode": centrality_mode,
        "time_centrality_s": time_centrality,
        "time_constructive_s": time_constructive,
        "time_initial_refinement_s": time_initial_refinement,
        "time_sa_s": time_sa,
        "time_final_refinement_s": time_final_refinement,
        "time_algorithm_excl_preprocessing_s": time_algorithm_excl_preprocessing,
        "time_total_incl_centrality_s": time_total,
    }


# ──────────────────────────────────────────────────────────────
# 5. Summary helpers
# ──────────────────────────────────────────────────────────────

def mean_or_nan(values: List[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def std_or_zero(values: List[float]) -> float:
    return float(np.std(values, ddof=0)) if values else 0.0


def bool_to_text(value: object) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "unknown"


def format_float(value: float, decimals: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    return f"{value:.{decimals}f}"


# ──────────────────────────────────────────────────────────────
# 6. Main execution
# ──────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("PAP solver — real network validation")
    print("=" * 70)
    print("Final configuration:")
    print(f"  GRASP_ITER={GRASP_ITER}, ALPHA={ALPHA}, LAM={LAM}")
    print(f"  centrality weights: ad={AD}, ae={AE}, ab={AB}")
    print(f"  runs per network: {N_RUNS_REAL}\n")

    raw_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for spec in NETWORKS:
        print(f"\n[{spec.name}]")

        try:
            G, metadata = load_network(spec)
        except Exception as error:
            print(f"  ERROR loading {spec.name}: {error}")
            continue

        n = G.number_of_nodes()
        e = G.number_of_edges()
        density = nx.density(G)

        print(
            f"  raw: n={metadata['raw_n']}, |E|={metadata['raw_edges']} | "
            f"processed LCC: n={n}, |E|={e}, density={density:.6f}"
        )
        print(
            f"  data_verified={bool_to_text(metadata['data_verified'])}, "
            f"comparison_class={metadata['comparison_class']}"
        )

        seed_values: List[int] = []
        valid_seed_values: List[int] = []
        total_times: List[float] = []
        algo_times: List[float] = []
        centrality_times: List[float] = []
        perfect_flags: List[int] = []

        for run in range(spec.runs):
            algorithm_seed = BASE_ALGORITHM_SEED + run

            result = run_final_pipeline(G, seed=algorithm_seed)

            seed_size = int(result["seed_size"])
            is_perfect = int(result["is_perfect"])

            seed_values.append(seed_size)
            perfect_flags.append(is_perfect)

            if is_perfect:
                valid_seed_values.append(seed_size)

            total_times.append(float(result["time_total_incl_centrality_s"]))
            algo_times.append(float(result["time_algorithm_excl_preprocessing_s"]))
            centrality_times.append(float(result["time_centrality_s"]))

            row = {
                "network": spec.name,
                "run": run,
                "algorithm_seed": algorithm_seed,
                "source_used": metadata["source_used"],
                "raw_n": metadata["raw_n"],
                "raw_edges": metadata["raw_edges"],
                "n": n,
                "edges": e,
                "density": round(density, 8),
                "expected_raw_n": metadata["expected_raw_n"],
                "expected_raw_edges": metadata["expected_raw_edges"],
                "data_verified": bool_to_text(metadata["data_verified"]),
                "comparison_class": metadata["comparison_class"],
                "component_reduction": metadata["component_reduction"],
                "reference_value": spec.reference_value,
                "runs_planned": spec.runs,
                "seed_size": seed_size,
                "seed_ratio": round(float(result["seed_ratio"]), 6),
                "is_perfect": is_perfect,
                "centrality_mode": result["centrality_mode"],
                "time_load_preprocess_s": round(float(metadata["load_preprocess_time_s"]), 6),
                "time_centrality_s": round(float(result["time_centrality_s"]), 6),
                "time_constructive_s": round(float(result["time_constructive_s"]), 6),
                "time_initial_refinement_s": round(float(result["time_initial_refinement_s"]), 6),
                "time_sa_s": round(float(result["time_sa_s"]), 6),
                "time_final_refinement_s": round(float(result["time_final_refinement_s"]), 6),
                "time_algorithm_excl_preprocessing_s": round(float(result["time_algorithm_excl_preprocessing_s"]), 6),
                "time_total_incl_centrality_s": round(float(result["time_total_incl_centrality_s"]), 6),
                "note": spec.note,
            }

            raw_rows.append(row)

            print(
                f"  run {run + 1}/{spec.runs}: "
                f"|S*|={seed_size}, perfect={bool(is_perfect)}, "
                f"t_total={float(result['time_total_incl_centrality_s']):.2f}s"
            )

        if not valid_seed_values:
            print(f"  No perfect solutions found for {spec.name}; skipping summary row.")
            continue

        best_seed = min(valid_seed_values)
        mean_seed = mean_or_nan(valid_seed_values)
        std_seed = std_or_zero(valid_seed_values)
        median_seed = float(np.median(valid_seed_values))

        reference = spec.reference_value
        delta_best = best_seed - reference if reference is not None else None
        delta_mean = mean_seed - reference if reference is not None else None

        summary_row = {
            "network": spec.name,
            "source_used": metadata["source_used"],
            "raw_n": metadata["raw_n"],
            "raw_edges": metadata["raw_edges"],
            "n": n,
            "edges": e,
            "density": round(density, 8),
            "expected_raw_n": metadata["expected_raw_n"],
            "expected_raw_edges": metadata["expected_raw_edges"],
            "data_verified": bool_to_text(metadata["data_verified"]),
            "comparison_class": metadata["comparison_class"],
            "component_reduction": metadata["component_reduction"],
            "runs": spec.runs,
            "perfect_runs": int(sum(perfect_flags)),
            "perfect_rate_pct": round(100.0 * mean_or_nan(perfect_flags), 2),
            "reported_value": "best",
            "best_seed": best_seed,
            "mean_seed": round(mean_seed, 4),
            "std_seed": round(std_seed, 4),
            "median_seed": round(median_seed, 4),
            "best_seed_ratio": round(best_seed / n, 6),
            "mean_seed_ratio": round(mean_seed / n, 6),
            "reference_value": reference,
            "delta_best": delta_best,
            "delta_mean": round(delta_mean, 4) if delta_mean is not None else None,
            "mean_time_load_preprocess_s": round(float(metadata["load_preprocess_time_s"]), 6),
            "mean_time_centrality_s": round(mean_or_nan(centrality_times), 6),
            "mean_time_algorithm_s": round(mean_or_nan(algo_times), 6),
            "mean_time_total_s": round(mean_or_nan(total_times), 6),
            "centrality_mode": raw_rows[-1]["centrality_mode"],
            "note": spec.note,
        }

        summary_rows.append(summary_row)

    if not raw_rows:
        print("\nNo raw results produced.")
        sys.exit(1)

    # Save raw results
    with RAW_OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(raw_rows[0].keys()))
        writer.writeheader()
        writer.writerows(raw_rows)

    print(f"\nRaw results saved: {RAW_OUTPUT}")

    if not summary_rows:
        print("No summary rows produced.")
        sys.exit(1)

    # Save summary results
    with SUMMARY_OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Summary saved: {SUMMARY_OUTPUT}")

    # Print compact table
    print()
    print(
        f"{'Network':15s} {'|V|':>7} {'|E|':>8} {'runs':>5} "
        f"{'best':>6} {'mean±std':>15} {'ref':>6} {'Δbest':>7} "
        f"{'class':>20} {'time(s)':>9}"
    )
    print("-" * 110)

    for row in summary_rows:
        delta = row["delta_best"]
        delta_text = "--" if delta is None else f"{delta:+d}"
        mean_std = f"{float(row['mean_seed']):.2f}±{float(row['std_seed']):.2f}"
        ref_text = "--" if row["reference_value"] is None else str(row["reference_value"])
        print(
            f"{row['network']:15s} {int(row['n']):>7,} {int(row['edges']):>8,} "
            f"{int(row['runs']):>5} {int(row['best_seed']):>6} "
            f"{mean_std:>15} {ref_text:>6} {delta_text:>7} "
            f"{row['comparison_class']:>20} {float(row['mean_time_total_s']):>9.2f}"
        )

    write_latex_table(summary_rows)
    make_figures(summary_rows)

    print("\nAll done.")


def write_latex_table(rows: List[Dict[str, object]]) -> None:
    """Create a compact LaTeX table for the thesis."""
    lines: List[str] = []

    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{l r r c r c r r c c}")
    lines.append(r"\toprule")
    lines.append(
        r"Red & $|V|$ & $|E|$ & Ejec. & Mejor & Media $\pm\sigma$ & Ref. & "
        r"$\Delta$ mejor & Verif. & Comp. \\")
    lines.append(r"\midrule")

    for row in rows:
        mean_std = f"{float(row['mean_seed']):.2f} $\\pm$ {float(row['std_seed']):.2f}"
        ref = "--" if row["reference_value"] is None else str(row["reference_value"])
        delta = "--" if row["delta_best"] is None else f"{int(row['delta_best']):+d}"
        verified = row["data_verified"]
        comp = row["comparison_class"]

        lines.append(
            f"{row['network']} & "
            f"{int(row['n'])} & "
            f"{int(row['edges'])} & "
            f"{int(row['runs'])} & "
            f"{int(row['best_seed'])} & "
            f"{mean_std} & "
            f"{ref} & "
            f"{delta} & "
            f"{verified} & "
            f"{comp} \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(
        r"\caption{Resultados sobre redes reales. Para cada red se indican el número de "
        r"ejecuciones independientes realizadas, el mejor valor obtenido, la media y "
        r"desviación típica de $|S_0^*|$, el valor de referencia reportado en la literatura "
        r"y la diferencia respecto al mejor valor obtenido en este trabajo. La columna "
        r"Verif. indica si el tamaño de la red descargada coincide con los metadatos "
        r"esperados, mientras que Comp. distingue las instancias directamente comparables "
        r"de aquellas que requieren cautela por filtrado de componentes o discrepancias "
        r"en los datos.}"
    )
    lines.append(r"\label{tab:real_networks_summary}")
    lines.append(r"\end{table}")

    with LATEX_OUTPUT.open("w", encoding="utf-8") as file:
        file.write("\n".join(lines))

    print(f"LaTeX table saved: {LATEX_OUTPUT}")


def make_figures(rows: List[Dict[str, object]]) -> None:
    """Create figures for the real-network validation."""
    style = {
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

    names = [str(r["network"]) for r in rows]
    ours_best = [float(r["best_seed"]) for r in rows]
    ref_values = [float(r["reference_value"]) if r["reference_value"] is not None else np.nan for r in rows]

    x = np.arange(len(names))
    width = 0.35

    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(11, 5))
        b1 = ax.bar(x - width / 2, ours_best, width, label="Este trabajo, mejor valor")
        b2 = ax.bar(x + width / 2, ref_values, width, label="Referencia")

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_ylabel(r"$|S_0^*|$")
        ax.set_title("Comparación del mejor conjunto semilla obtenido en redes reales")
        ax.legend()

        scale = max([v for v in ours_best + ref_values if not np.isnan(v)]) * 0.012
        for bars in [b1, b2]:
            for bar in bars:
                h = bar.get_height()
                if not np.isnan(h):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        h + scale,
                        str(int(h)),
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

        fig.tight_layout()
        path = OUT_DIR / "fig_real_networks_comparison_best.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")

    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(9, 5))

        for row in rows:
            ax.scatter(
                float(row["density"]),
                float(row["best_seed_ratio"]),
                s=110,
                label=f"{row['network']} (best={row['best_seed']})",
            )
            ax.annotate(
                str(row["network"]),
                (float(row["density"]), float(row["best_seed_ratio"])),
                textcoords="offset points",
                xytext=(6, 3),
                fontsize=8.5,
            )

        ax.set_xlabel(r"Densidad $\rho(G)=\frac{2|E|}{|V|(|V|-1)}$")
        ax.set_ylabel(r"Mejor $|S_0^*|/n$")
        ax.set_title(r"Cociente normalizado de la semilla en redes reales")
        ax.legend(loc="best", fontsize=8)

        fig.tight_layout()
        path = OUT_DIR / "fig_real_networks_ratio_density.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
