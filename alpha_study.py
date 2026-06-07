import csv, warnings, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pap_solver import generate_ba_graph, solve_pap
warnings.filterwarnings('ignore')

INSTANCES   = [(50,2),(100,2),(100,3),(150,2),(200,3)]
ALPHA_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
N_RUNS      = 5
GRASP_ITER  = 8
T_INIT, T_MIN, COOLING, STEPS, LAM = 5.0, 0.01, 0.97, 25, 2.0

print("Running alpha study...")
rows = []
total = len(ALPHA_VALUES)*len(INSTANCES)*N_RUNS; done = 0
for alpha in ALPHA_VALUES:
    for (n, m) in INSTANCES:
        for run in range(N_RUNS):
            seed = 1000 * n + 10 * m + run
            G    = generate_ba_graph(n, m, seed=seed)
            res  = solve_pap(G, n_grasp_iter=GRASP_ITER, alpha=alpha,
                             T_init=T_INIT, T_min=T_MIN, cooling=COOLING,
                             steps_per_temp=STEPS, lam=LAM, seed=seed)
            rows.append({'alpha':alpha,'n':n,'m':m,'run':run,
                         'seed_ratio':res.seed_size/n,
                         'is_perfect':int(res.is_perfect)})
            done += 1
            if done % 55 == 0 or done == total:
                print(f"  [{done:3d}/{total}] alpha={alpha:.1f} n={n} m={m} "
                      f"ratio={res.seed_size/n:.3f} ok={res.is_perfect}")
                sys.stdout.flush()

# Save CSV
with open('results_alpha.csv','w',newline='') as f:
    w = csv.DictWriter(f, fieldnames=['alpha','n','m','run','seed_ratio','is_perfect'])
    w.writeheader(); w.writerows(rows)
print("CSV saved.")
print("DONE_DATA")
