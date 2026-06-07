# Perfect Awareness Problem — GRASP + Simulated Annealing

This repository contains the implementation and experimental evaluation of a hybrid metaheuristic for the **Perfect Awareness Problem (PAP)**. The project was developed as part of a Master's Thesis in Data Science.

The proposed method combines a multi-start **GRASP** construction phase, a redundancy-removal refinement procedure and a **Simulated Annealing** improvement phase. The objective is to identify small perfect seed sets capable of ensuring that every node in a network eventually becomes aware of a piece of information under a threshold-based diffusion process.

## Problem description

The Perfect Awareness Problem models information propagation over an undirected and unweighted graph:

$G=(V,E),$

where each vertex represents an individual and each edge represents a communication link.

Each node can be in one of three states:

- **ignorant**: the node has not received the information;
- **aware**: the node knows the information but does not actively spread it;
- **spreader**: the node knows the information and transmits it to its neighbors.

A node becomes aware when at least one neighboring spreader exists. A node becomes a spreader when the number of neighboring spreaders reaches its activation threshold:

$t(v)=[ 0.5 \cdot d(v)]$,

where \(d(v)\) denotes the degree of node \(v\).

The goal is to find a minimum-cardinality seed set:

$S_0^* \subseteq V$

such that every node in the graph eventually becomes aware of the information after the diffusion process.

## Proposed method

The solver follows a hybrid pipeline:

```text
Multi-start GRASP
        ↓
Refinement of each constructed solution
        ↓
Selection of the best GRASP solution
        ↓
Simulated Annealing
        ↓
Final refinement
        ↓
Perfect seed set
````

### GRASP construction phase

The GRASP phase incrementally builds a feasible seed set. At each step, candidate nodes are evaluated using a contribution function based on centrality measures and the current diffusion state of their neighbors.

The contribution of a node (v) is defined as:

$g(v)=
\frac{
a_d C_v^D +
a_e C_v^E +
a_b C_v^{\beta}
}{
a_d+a_e+a_b
}
\cdot
\frac{
d(v)-|A_v(S_0)|
}{
d(v)
},$

where:

* $C_v^D$ is degree centrality;
* $C_v^E$ is eigenvector centrality;
* $C_v^{\beta}$ is betweenness centrality;
* $A_v(S_0)$ is the set of aware neighbors of $v$;
* $a_d$, $a_e$ and $a_b$ are centrality weights.

### Simulated Annealing phase

The Simulated Annealing phase explores alternative configurations through three neighborhood moves:

* `ADD`: add a non-seed node;
* `REMOVE`: remove a seed node;
* `SWAP`: replace a seed node with a non-seed node.

The search is guided by the penalized energy function:

$E(S_0) =
|S_0|
+
\lambda
\left(
|V|-|A(S_0)|
\right),$

where:

* ($|S_0|$) is the size of the candidate seed set;
* ($A(S_0)$) is the set of aware nodes after the diffusion process;
* ($\lambda$) is the penalty parameter for non-feasible solutions.

When ($S_0$) is a perfect seed set, ($A(S_0)=V$), and therefore:

$E(S_0)=|S_0|.$

The energy function allows the algorithm to temporarily explore non-feasible solutions while preserving the best feasible solution found during the search.

### Refinement phase

The refinement procedure removes redundant seed nodes whenever their elimination does not compromise complete diffusion. This guarantees that the final solution is locally minimal with respect to individual node removals.

## Default configuration

The experiments use the following default parameters:

| Parameter        | Value | Description                                   |
| ---------------- | ----: | --------------------------------------------- |
| `n_grasp_iter`   |     8 | Number of GRASP multi-start iterations        |
| `alpha`          |   0.3 | GRASP greediness parameter                    |
| `lambda`         |   2.0 | Penalty for non-aware nodes                   |
| `a_d`            |   1.0 | Degree-centrality weight                      |
| `a_e`            |   1.0 | Eigenvector-centrality weight                 |
| `a_b`            |   1.0 | Betweenness-centrality weight                 |
| `T_init`         |   5.0 | Initial SA temperature                        |
| `T_min`          |  0.01 | Minimum SA temperature                        |
| `cooling`        |  0.97 | Geometric cooling factor                      |
| `steps_per_temp` |    25 | Number of SA iterations per temperature level |

## Repository contents

The repository contains:

```text
.
├── pap_solver.py
├── run_real_networks.py
├── experiments/
├── figures/
├── results/
└── README.md
```

The main components are:

* `pap_solver.py`: core PAP solver, including diffusion simulation, GRASP construction, refinement and Simulated Annealing;
* `run_real_networks.py`: benchmark execution over real-world networks;
* `experiments/`: scripts used for parameter sensitivity studies, ablation experiments and exact validation;
* `figures/`: generated plots;
* `results/`: CSV files containing the experimental outputs.

The exact folder structure may be adjusted depending on the final organization of the repository.

## Installation

The implementation was developed with Python 3.11.

Install the required dependencies with:

```bash
pip install networkx numpy pandas matplotlib
```

## Basic usage

The solver can be imported directly from `pap_solver.py`.

```python
from pap_solver import generate_ba_graph, solve_pap

n = 500
m = 3
seed = 1000 * n + 10 * m

G = generate_ba_graph(
    n=n,
    m=m,
    seed=seed,
)

result = solve_pap(
    G,
    n_grasp_iter=8,
    alpha=0.3,
    T_init=5.0,
    T_min=0.01,
    cooling=0.97,
    steps_per_temp=25,
    lam=2.0,
    seed=seed,
)

print("Perfect seed set:", result.seed_set)
print("Seed-set size:", result.seed_size)
print("Is perfect:", result.is_perfect)
print("Execution time:", result.time_total)
```

## Synthetic experiments

The main experimental analysis uses Barabási–Albert graphs with:

$n \in
{
20,40,60,80,100,150,200,500,1000,3000,5000
},$

and:

$m \in {1,2,3,4,5}.$

Five independent pseudo-random executions are performed for each pair (n,m), using deterministic seeds:

$\text{seed}=1000n+10m+r,$

where:

$r \in {0,1,2,3,4}.$

The experimental evaluation includes:

* analysis of the normalized seed-set size $|S_0^*|/n$;
* parameter-sensitivity analysis for $\lambda$;
* parameter-sensitivity analysis for $\alpha$;
* centrality-weight analysis;
* component ablation study;
* execution-time analysis;
* exact validation on small instances.

## Exact validation

For small graphs, the metaheuristic is compared against an exhaustive enumeration procedure. Candidate seed sets are evaluated in increasing order of cardinality. The first perfect seed set found is therefore an optimal solution.

The exact-validation experiments cover graphs with up to 40 nodes.

## Real-world networks

The algorithm is also evaluated on the following real-world networks:

| Network     | Description                                                                  |
| ----------- | ---------------------------------------------------------------------------- |
| Karate Club | Social interactions in a university karate club                              |
| Jazz        | Collaboration network between jazz musicians                                 |
| Facebook    | Ego-network social circles                                                   |
| Power grid  | Western United States power-grid network                                     |
| CA-GrQc     | Scientific collaboration network in General Relativity and Quantum Cosmology |
| CA-HepTh    | Scientific collaboration network in High-Energy Physics Theory               |

For networks with more than 500 nodes, the construction phase uses a degree-only adaptation to reduce preprocessing costs. In this setting, degree centrality replaces the complete combination of degree, eigenvector and betweenness centralities.

For CA-GrQc and CA-HepTh, the experiments use the largest connected component of the downloaded network.

## Reproducibility

The experiments use deterministic pseudo-random seeds to ensure reproducibility. The same seed is used both for the generation of each Barabási–Albert graph and for the stochastic components of the solver.

Execution times may vary depending on the hardware and operating system.

## Academic context

This repository accompanies the Master's Thesis:

> **Metaheurística híbrida GRASP + Simulated Annealing para el Problema de Concienciación Perfecta en redes: diseño y validación experimental**

The thesis studies the design, implementation and validation of a hybrid metaheuristic for information diffusion in synthetic and real-world networks.

## References

The main reference for the Perfect Awareness Problem benchmark and the comparison with previous heuristics is:

```text
Pereira, F. de C., de Rezende, P. J., and de Souza, C. C. (2021).
Effective heuristics for the Perfect Awareness Problem.
Procedia Computer Science, 195, 489–498.
```

The Simulated Annealing approach follows the general framework introduced in:

```text
Kirkpatrick, S., Gelatt, C. D., and Vecchi, M. P. (1983).
Optimization by simulated annealing.
Science, 220(4598), 671–680.
```

## Author

**Daniela Meriño Pérez**

Master's Degree in Data Science
July 2026

