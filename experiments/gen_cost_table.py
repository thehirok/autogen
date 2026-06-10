"""Generate LaTeX table for computational cost comparison."""
import pandas as pd
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

df = pd.read_csv('results/algorithm_performance.csv')
cost = pd.read_csv('results/computational_cost.csv')
total_train = df.groupby('dataset')['fit_time'].sum().reset_index()
total_train.columns = ['dataset', 'total_train_sec']
m = cost.merge(total_train, on='dataset', how='left')
m['speedup'] = m['total_train_sec'] / m['early_select_sec']

lines = []
lines.append(r'\begin{table}[htbp]')
lines.append(r'\centering')
lines.append(r'\caption{Computational cost: Early Selection ($N\!=\!250$) vs.\ full evaluation (training all 6 algorithms). Speedup is the ratio of full training time to early selection time.}')
lines.append(r'\label{tab:cost}')
lines.append(r'\begin{tabular}{lrrr}')
lines.append(r'\toprule')
lines.append(r'Dataset & Early (s) & Full Train (s) & Speedup \\')
lines.append(r'\midrule')
for _, r in m.iterrows():
    name = r['dataset'].replace('_', r'\_')
    early = r['early_select_sec']
    train = r['total_train_sec']
    spd = r['speedup']
    lines.append(f'{name} & {early:.2f} & {train:.0f} & {spd:.0f}$\\times$ \\\\')
lines.append(r'\midrule')
mean_early = m['early_select_sec'].mean()
mean_train = m['total_train_sec'].mean()
median_spd = m['speedup'].median()
lines.append(f'\\textbf{{Mean/Median}} & \\textbf{{{mean_early:.2f}}} & \\textbf{{{mean_train:.0f}}} & \\textbf{{{median_spd:.0f}$\\times$}} \\\\')
lines.append(r'\bottomrule')
lines.append(r'\end{tabular}')
lines.append(r'\end{table}')

tex = '\n'.join(lines)

for path in ['paper/tables/tab_cost.tex', 'latex/tables/tab_cost.tex']:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(tex)
    print(f"Saved -> {path}")

print()
print(tex)
