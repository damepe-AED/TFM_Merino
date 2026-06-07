import csv, warnings
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from pap_solver import generate_ba_graph, solve_pap
warnings.filterwarnings('ignore')

LAMBDA_VALUES = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0]
INSTANCES = [(50,2),(100,2),(100,3),(150,2),(200,3)]
N_RUNS=5; GRASP_ITER=8; ALPHA=0.3
T_INIT=5.0; T_MIN=0.01; COOLING=0.97; STEPS=2

print('Running lambda sensitivity study...')
all_rows=[]; histories={}
total=len(LAMBDA_VALUES)*len(INSTANCES)*N_RUNS; done=0

for (n,m),lam in product(INSTANCES,LAMBDA_VALUES):
    for run in range(N_RUNS):
        seed = 1000 * n + 10 * m + run
        G=generate_ba_graph(n,m,seed=seed)
        res=solve_pap(G,n_grasp_iter=GRASP_ITER,alpha=ALPHA,
                      T_init=T_INIT,T_min=T_MIN,cooling=COOLING,
                      steps_per_temp=STEPS,lam=lam,seed=seed)
        all_rows.append({'n':n,'m':m,'lambda':lam,'run':run,
            'seed_size':res.seed_size,'seed_ratio':res.seed_size/n,
            'is_perfect':int(res.is_perfect),'time_total':round(res.time_total,4)})
        histories[(n,m,lam,run)]=res.sa_history
        done+=1
        if done%40==0 or done==total:
            print(f'  [{done:3d}/{total}] n={n} m={m} lam={lam} seed={res.seed_size} ok={res.is_perfect}')

# Save CSV
csv_keys=['n','m','lambda','run','seed_size','seed_ratio','is_perfect','time_total']
with open('results_lambda.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=csv_keys)
    w.writeheader(); w.writerows(all_rows)
print('CSV saved.')

def agg(rows, lam, key, n=None, m=None):
    r = [x for x in rows if x['lambda']==lam]
    if n is not None: r=[x for x in r if x['n']==n]
    if m is not None: r=[x for x in r if x['m']==m]
    v=[x[key] for x in r]
    return np.mean(v), np.std(v)

STYLE={
    'axes.spines.top':False,'axes.spines.right':False,
    'axes.grid':True,'grid.alpha':0.35,'grid.linestyle':'--',
    'font.size':11,'axes.titlesize':12,'axes.labelsize':11,
    'legend.fontsize':9,'figure.dpi':150
}
C=plt.cm.tab10.colors

# Fig 1 – quality per instance
with plt.rc_context(STYLE):
    fig,axes=plt.subplots(2,len(INSTANCES),figsize=(14,7),sharex=True)
    for col,(n,m) in enumerate(INSTANCES):
        means,stds,perfs=[],[],[]
        for lam in LAMBDA_VALUES:
            mu,sig=agg(all_rows,lam,'seed_size',n=n,m=m)
            pf,_=agg(all_rows,lam,'is_perfect',n=n,m=m)
            means.append(mu); stds.append(sig); perfs.append(pf*100)
        axes[0,col].errorbar(LAMBDA_VALUES,means,yerr=stds,marker='o',
                             linewidth=1.8,color=C[col],capsize=3)
        axes[0,col].set_title(f'n={n}, m={m}')
        if col==0: axes[0,col].set_ylabel('$|S_0^*|$')
        axes[1,col].plot(LAMBDA_VALUES,perfs,marker='s',linewidth=1.8,color=C[col])
        axes[1,col].set_ylim(-5,105)
        axes[1,col].axhline(100,color='grey',linestyle=':',linewidth=1)
        if col==0: axes[1,col].set_ylabel('Perfect solutions (%)')
        axes[1,col].set_xlabel('lambda')
    fig.suptitle('Effect of lambda on solution quality',fontsize=13,y=1.01)
    fig.tight_layout()
    fig.savefig('fig_lambda_quality.png',bbox_inches='tight')
    plt.close(fig)
    print('Saved fig_lambda_quality.png')

# Fig 2 – aggregated
with plt.rc_context(STYLE):
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,4.5))
    for i,(n,m) in enumerate(INSTANCES):
        means,stds=[],[]
        for lam in LAMBDA_VALUES:
            mu,sig=agg(all_rows,lam,'seed_ratio',n=n,m=m)
            means.append(mu); stds.append(sig)
        ax1.errorbar(LAMBDA_VALUES,means,yerr=stds,label=f'n={n},m={m}',
                     marker='o',linewidth=1.6,color=C[i],capsize=3)
    ax1.set_xlabel('lambda'); ax1.set_ylabel('|S0*|/n')
    ax1.set_title('Normalised seed ratio vs lambda'); ax1.legend()
    pm,ps=[],[]
    for lam in LAMBDA_VALUES:
        mu,sig=agg(all_rows,lam,'is_perfect')
        pm.append(mu*100); ps.append(sig*100)
    ax2.errorbar(LAMBDA_VALUES,pm,yerr=ps,marker='D',linewidth=2,
                 color='steelblue',capsize=4)
    ax2.axhline(100,color='grey',linestyle=':',linewidth=1)
    ax2.set_ylim(-5,110); ax2.set_xlabel('lambda')
    ax2.set_ylabel('Perfect solutions (%)'); ax2.set_title('Feasibility rate vs lambda')
    fig.tight_layout(); fig.savefig('fig_lambda_aggregated.png'); plt.close(fig)
    print('Saved fig_lambda_aggregated.png')

# Fig 3 – convergence curves
n_c,m_c=100,2
with plt.rc_context(STYLE):
    fig,axes=plt.subplots(2,4,figsize=(14,6))
    axes=axes.flatten()
    for idx,lam in enumerate(LAMBDA_VALUES):
        ax=axes[idx]
        hist=histories.get((n_c,m_c,lam,0),[])
        ax.plot(range(len(hist)),hist,color=C[idx%10],linewidth=1.5)
        if hist:
            bs=int(np.argmin(hist))
            ax.axvline(bs,color='red',linestyle='--',linewidth=0.9,alpha=0.7)
            ax.scatter([bs],[hist[bs]],color='red',zorder=5,s=25)
        sr=next((r['seed_size'] for r in all_rows
                 if r['n']==n_c and r['m']==m_c
                 and r['lambda']==lam and r['run']==0),'?')
        pr=next((r['is_perfect'] for r in all_rows
                 if r['n']==n_c and r['m']==m_c
                 and r['lambda']==lam and r['run']==0),'?')
        ok = 'OK' if pr else 'FAIL'
        ax.set_title(f'lam={lam}  |S*|={sr} {ok}',fontsize=10)
        ax.set_xlabel('Temp step',fontsize=9); ax.set_ylabel('Energy',fontsize=9)
    fig.suptitle(f'SA convergence for different lambda  (n={n_c}, m={m_c})',fontsize=13)
    fig.tight_layout(); fig.savefig('fig_lambda_convergence.png'); plt.close(fig)
    print('Saved fig_lambda_convergence.png')

# Fig 4 – heatmap
inst_labels=[f'n={n},m={m}' for n,m in INSTANCES]
matrix=np.zeros((len(LAMBDA_VALUES),len(INSTANCES)))
for j,(n,m) in enumerate(INSTANCES):
    for i,lam in enumerate(LAMBDA_VALUES):
        mu,_=agg(all_rows,lam,'seed_ratio',n=n,m=m)
        matrix[i,j]=mu
with plt.rc_context(STYLE):
    fig,ax=plt.subplots(figsize=(8,5))
    im=ax.imshow(matrix,aspect='auto',cmap='YlOrRd',origin='upper',interpolation='nearest')
    fig.colorbar(im,ax=ax,label='|S0*|/n')
    ax.set_xticks(range(len(INSTANCES))); ax.set_xticklabels(inst_labels,rotation=20,ha='right')
    ax.set_yticks(range(len(LAMBDA_VALUES))); ax.set_yticklabels([str(l) for l in LAMBDA_VALUES])
    ax.set_ylabel('lambda'); ax.set_title('Mean |S0*|/n over lambda and instance')
    for i2 in range(len(LAMBDA_VALUES)):
        for j2 in range(len(INSTANCES)):
            ax.text(j2,i2,f'{matrix[i2,j2]:.3f}',ha='center',va='center',fontsize=8,
                    color='black' if matrix[i2,j2]<0.18 else 'white')
    fig.tight_layout(); fig.savefig('fig_lambda_heatmap.png'); plt.close(fig)
    print('Saved fig_lambda_heatmap.png')

# Summary table
print('\n── Lambda sensitivity summary ────────────────────')
print(f'{"lambda":>7}  {"mean ratio":>10}  {"std":>6}  {"perfect%":>9}')
print('-'*40)
for lam in LAMBDA_VALUES:
    mu,sig=agg(all_rows,lam,'seed_ratio')
    pf,_=agg(all_rows,lam,'is_perfect')
    print(f'{lam:>7.1f}  {mu:>10.4f}  {sig:>6.4f}  {pf*100:>8.1f}%')
print('ALL DONE')
