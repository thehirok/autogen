"""
Generate all publication-ready figures AND LaTeX tables for the paper.
Reads from results/ and writes PDFs to paper/figures/, .tex to paper/tables/.

Run this after all experiment scripts have completed.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import logging

from config import (RESULTS_DIR, PAPER_FIGURES_DIR, PAPER_TABLES_DIR,
                    FIGURE_DPI, FONT_FAMILY, FONT_SIZE,
                    ONE_COL_WIDTH, TWO_COL_WIDTH, SAMPLE_SIZES)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# --- global matplotlib settings (ACM two-column) ---
matplotlib.rcParams.update({
    'font.family':       FONT_FAMILY,
    'font.size':         FONT_SIZE,
    'axes.titlesize':    FONT_SIZE,
    'axes.labelsize':    FONT_SIZE,
    'xtick.labelsize':   FONT_SIZE - 1,
    'ytick.labelsize':   FONT_SIZE - 1,
    'legend.fontsize':   FONT_SIZE - 1,
    'figure.dpi':        FIGURE_DPI,
    'savefig.dpi':       FIGURE_DPI,
    'savefig.bbox':      'tight',
    'savefig.pad_inches': 0.02,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'grid.linewidth':    0.5,
})

PALETTE = sns.color_palette("tab10")


# ===============================================================
# FIGURES
# ===============================================================

# Fig 1 — Feature stability curves  (two-column)
def plot_fig1_stability(df, top_n=8):
    path = os.path.join(PAPER_FIGURES_DIR, 'fig1_stability_curves.pdf')

    mean_err = (
        df.groupby(['feature', 'sample_size'])['relative_error']
        .mean()
        .reset_index()
    )

    min_n = mean_err['sample_size'].min()
    top_feats = (
        mean_err[mean_err['sample_size'] == min_n]
        .nlargest(top_n, 'relative_error')['feature']
        .tolist()
    )

    fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH, 3.2))
    sizes = sorted(mean_err['sample_size'].unique())

    for i, feat in enumerate(top_feats):
        sub = mean_err[mean_err['feature'] == feat].sort_values('sample_size')
        ax.plot(sub['sample_size'], sub['relative_error'],
                marker='o', markersize=3, linewidth=1.2,
                color=PALETTE[i % len(PALETTE)],
                label=feat.replace('_', ' '))

    ax.axhline(0.05, color='gray', linestyle='--',
               linewidth=0.8, label='5\\% threshold')
    ax.set_xscale('log')
    ax.set_xlabel('Sample size (N)')
    ax.set_ylabel('Mean relative error')
    ax.set_xticks(sizes)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.legend(ncol=2, framealpha=0.7, fontsize=FONT_SIZE - 2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved → {path}")


# Fig 2 — Selection accuracy vs sample size  (one-column)
def plot_fig2_accuracy(df):
    path = os.path.join(PAPER_FIGURES_DIR, 'fig2_selection_accuracy.pdf')

    rf = df[df['model_type'] == 'RandomForest'].copy()
    rf = rf[rf['sample_size'] != 'full']
    rf['sample_size'] = rf['sample_size'].astype(int)

    acc = (
        rf.groupby('sample_size')['correct']
        .agg(['mean', 'std', 'count'])
        .reset_index()
    )
    acc['se'] = acc['std'] / np.sqrt(acc['count'])

    full_acc = df[(df['model_type'] == 'RandomForest') &
                  (df['sample_size'] == 'full')]['correct'].mean()

    fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH, 2.8))
    ax.errorbar(acc['sample_size'], acc['mean'],
                yerr=1.96 * acc['se'],
                marker='o', markersize=4, linewidth=1.4,
                color=PALETTE[0], capsize=3, label='Early selection')
    ax.axhline(full_acc, color=PALETTE[1], linestyle='--',
               linewidth=1.0, label='Full-data selection')

    ax.set_xscale('log')
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Sample size (N)')
    ax.set_ylabel('Selection accuracy')
    ax.set_xticks(sorted(acc['sample_size'].unique()))
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.legend(framealpha=0.7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved → {path}")


# Fig 3 — SHAP + ESS feature importance  (one-column)
def plot_fig3_shap_ess(shap_df, ess_df, top_n=12):
    path = os.path.join(PAPER_FIGURES_DIR, 'fig3_shap_ess.pdf')

    top_shap = shap_df.nlargest(top_n, 'mean_abs_shap').copy()
    top_shap['feature_label'] = (
        top_shap['feature'].str.replace('_', ' '))

    fig, axes = plt.subplots(1, 2,
                             figsize=(TWO_COL_WIDTH, 3.4),
                             sharey=False)

    axes[0].barh(top_shap['feature_label'],
                 top_shap['mean_abs_shap'],
                 color=PALETTE[0], height=0.6)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Mean |SHAP|')
    axes[0].set_title('Feature importance')

    top_ess = ess_df.nlargest(top_n, 'ess').copy()
    top_ess['feature_label'] = top_ess['feature'].str.replace('_', ' ')
    axes[1].barh(top_ess['feature_label'],
                 top_ess['ess'],
                 color=PALETTE[2], height=0.6)
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Early Selection Score')
    axes[1].set_title('ESS ranking')

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved → {path}")


# Fig 4 — End-to-end quality comparison  (two-column)
def plot_fig4_end_to_end(df):
    path = os.path.join(PAPER_FIGURES_DIR, 'fig4_end_to_end.pdf')

    rows = []
    for cond in ['oracle', 'random', 'full_selection']:
        sub = df[df['condition'] == cond]
        rows.append({
            'label':   cond.replace('_', ' '),
            'x':       'full',
            'ndcg@10': sub['ndcg@10'].mean(),
            'se':      sub['ndcg@10'].sem(),
        })

    early = df[df['condition'] == 'early_selection'].copy()
    early = early[early['sample_size'] != 'full']
    early['sample_size'] = early['sample_size'].astype(int)
    for n, grp in early.groupby('sample_size'):
        rows.append({
            'label': f'early N={n}',
            'x':     n,
            'ndcg@10': grp['ndcg@10'].mean(),
            'se':    grp['ndcg@10'].sem(),
        })

    plot_df = pd.DataFrame(rows)
    plot_df = plot_df.sort_values('x', key=lambda s: s.map(
        lambda v: -1 if v == 'full' else int(v)))

    fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH, 2.8))
    colors_map = {
        'oracle':          PALETTE[2],
        'random':          PALETTE[3],
        'full selection':  PALETTE[0],
    }
    x_pos = range(len(plot_df))
    bars  = ax.bar(x_pos,
                   plot_df['ndcg@10'],
                   yerr=1.96 * plot_df['se'],
                   capsize=3,
                   color=[colors_map.get(r['label'], PALETTE[1])
                          for _, r in plot_df.iterrows()],
                   width=0.6,
                   error_kw={'linewidth': 0.8})

    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(plot_df['label'],
                       rotation=30, ha='right', fontsize=FONT_SIZE - 1)
    ax.set_ylabel('Mean NDCG@10')
    ax.set_title('End-to-end recommendation quality by selection strategy')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved → {path}")


# ===============================================================
# LATEX TABLES
# ===============================================================

def _save_tex(content, filename):
    path = os.path.join(PAPER_TABLES_DIR, filename)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    logger.info(f"Saved → {path}")


def table_algorithm_performance(df):
    """Table 1: NDCG@10 per algorithm × dataset."""
    pivot = df.pivot_table(
        values='ndcg@10', index='dataset', columns='algorithm', aggfunc='mean')
    pivot = pivot.round(4)

    # Bold the best (highest) NDCG@10 per dataset
    def bold_max(row):
        max_val = row.max()
        return [f'\\textbf{{{v:.4f}}}' if v == max_val else f'{v:.4f}'
                for v in row]

    algos = [c for c in pivot.columns]
    lines = []
    lines.append('\\begin{table}[t]')
    lines.append('\\centering')
    lines.append('\\caption{NDCG@10 of each algorithm on each dataset. '
                 'Best results per dataset in \\textbf{bold}.}')
    lines.append('\\label{tab:rq1}')
    lines.append('\\small')
    lines.append('\\begin{tabular}{l' + 'c' * len(algos) + '}')
    lines.append('\\toprule')
    lines.append('Dataset & ' + ' & '.join(algos) + ' \\\\')
    lines.append('\\midrule')
    for ds, row in pivot.iterrows():
        formatted = bold_max(row)
        ds_label = ds.replace('_', '\\_')
        lines.append(f'{ds_label} & ' + ' & '.join(formatted) + ' \\\\')
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')

    _save_tex('\n'.join(lines), 'tab_algorithm_performance.tex')


def table_selection_accuracy(df):
    """Table 2: Selection accuracy at each sample size."""
    rf = df[df['model_type'] == 'RandomForest'].copy()

    rows = []
    # Full-data row
    full = rf[rf['sample_size'] == 'full']
    if len(full):
        rows.append({'Sample Size': 'Full', 'Accuracy': full['correct'].mean(),
                     'Std': full['correct'].std(), 'N': len(full)})

    # Per sample size
    sub = rf[rf['sample_size'] != 'full'].copy()
    sub['sample_size'] = sub['sample_size'].astype(int)
    for n, grp in sub.groupby('sample_size'):
        rows.append({'Sample Size': f'{n:,}', 'Accuracy': grp['correct'].mean(),
                     'Std': grp['correct'].std(), 'N': len(grp)})

    lines = []
    lines.append('\\begin{table}[t]')
    lines.append('\\centering')
    lines.append('\\caption{Algorithm selection accuracy by sample size '
                 '(RandomForest meta-model).}')
    lines.append('\\label{tab:rq3}')
    lines.append('\\begin{tabular}{lrrr}')
    lines.append('\\toprule')
    lines.append('Sample Size & Accuracy & Std & N \\\\')
    lines.append('\\midrule')
    for r in rows:
        lines.append(f"{r['Sample Size']} & {r['Accuracy']:.3f} & "
                     f"{r['Std']:.3f} & {r['N']} \\\\")
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')

    _save_tex('\n'.join(lines), 'tab_selection_accuracy.tex')


def table_feature_importance(shap_df, ess_df, top_n=12):
    """Table 3: Top features by SHAP importance and ESS score."""
    top_shap = shap_df.nlargest(top_n, 'mean_abs_shap')[
        ['feature', 'mean_abs_shap']].reset_index(drop=True)
    top_ess = ess_df.nlargest(top_n, 'ess')[
        ['feature', 'ess']].reset_index(drop=True)

    lines = []
    lines.append('\\begin{table}[t]')
    lines.append('\\centering')
    lines.append('\\caption{Top-12 meta-features by SHAP importance '
                 'and Early Selection Score (ESS).}')
    lines.append('\\label{tab:rq2}')
    lines.append('\\small')
    lines.append('\\begin{tabular}{rlrl}')
    lines.append('\\toprule')
    lines.append('\\multicolumn{2}{c}{SHAP Importance} & '
                 '\\multicolumn{2}{c}{ESS Score} \\\\')
    lines.append('\\cmidrule(lr){1-2} \\cmidrule(lr){3-4}')
    lines.append('Feature & |SHAP| & Feature & ESS \\\\')
    lines.append('\\midrule')
    for i in range(top_n):
        sf = top_shap.iloc[i]['feature'].replace('_', '\\_')
        sv = top_shap.iloc[i]['mean_abs_shap']
        ef = top_ess.iloc[i]['feature'].replace('_', '\\_')
        ev = top_ess.iloc[i]['ess']
        lines.append(f'{sf} & {sv:.4f} & {ef} & {ev:.4f} \\\\')
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')

    _save_tex('\n'.join(lines), 'tab_feature_importance.tex')


def table_significance(df):
    """Table 4: Significance test results."""
    lines = []
    lines.append('\\begin{table}[t]')
    lines.append('\\centering')
    lines.append('\\caption{Statistical significance tests '
                 '(Wilcoxon / paired t-test, $\\alpha=0.05$).}')
    lines.append('\\label{tab:rq4}')
    lines.append('\\small')
    lines.append('\\begin{tabular}{llcccc}')
    lines.append('\\toprule')
    lines.append('Condition A & Condition B & Mean A & Mean B & '
                 '$p$-value & Sig. \\\\')
    lines.append('\\midrule')
    for _, row in df.iterrows():
        ca = str(row['condition_a']).replace('_', '\\_')
        cb = str(row['condition_b']).replace('_', '\\_')
        sig = '\\checkmark' if row['significant'] else '--'
        p = f"{row['p_value']:.4f}" if not np.isnan(row['p_value']) else 'N/A'
        lines.append(f"{ca} & {cb} & {row['mean_a']:.4f} & "
                     f"{row['mean_b']:.4f} & {p} & {sig} \\\\")
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')

    _save_tex('\n'.join(lines), 'tab_significance.tex')


def table_ablation(df):
    """Table 5: Meta-model ablation (RF vs XGBoost vs MLP)."""
    summary = df.groupby('model_type').agg(
        accuracy=('correct', 'mean'),
        accuracy_std=('correct', 'std'),
    ).reset_index()

    lines = []
    lines.append('\\begin{table}[t]')
    lines.append('\\centering')
    lines.append('\\caption{Meta-model architecture comparison '
                 '(leave-one-out accuracy).}')
    lines.append('\\label{tab:ablation}')
    lines.append('\\begin{tabular}{lcc}')
    lines.append('\\toprule')
    lines.append('Model & Accuracy & Std \\\\')
    lines.append('\\midrule')
    for _, row in summary.iterrows():
        lines.append(f"{row['model_type']} & {row['accuracy']:.3f} & "
                     f"{row['accuracy_std']:.3f} \\\\")
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')

    _save_tex('\n'.join(lines), 'tab_ablation.tex')


# ===============================================================
# Main
# ===============================================================
def main():
    os.makedirs(PAPER_FIGURES_DIR, exist_ok=True)
    os.makedirs(PAPER_TABLES_DIR, exist_ok=True)

    def load(name):
        p = os.path.join(RESULTS_DIR, name)
        if not os.path.exists(p):
            logger.warning(f"Missing: {p} — skipping related figure/table.")
            return None
        return pd.read_csv(p)

    perf  = load('algorithm_performance.csv')
    stab  = load('stability_curves.csv')
    acc   = load('selection_accuracy_by_n.csv')
    shap  = load('shap_values.csv')
    ess   = load('ess_scores.csv')
    e2e   = load('end_to_end_quality.csv')
    sig   = load('significance_tests.csv')
    abl   = load('ablation_meta_models.csv')

    # --- Figures ---
    if stab is not None:
        plot_fig1_stability(stab)
    if acc is not None:
        plot_fig2_accuracy(acc)
    if shap is not None and ess is not None:
        plot_fig3_shap_ess(shap, ess)
    if e2e is not None:
        plot_fig4_end_to_end(e2e)

    # --- Tables ---
    if perf is not None:
        table_algorithm_performance(perf)
    if acc is not None:
        table_selection_accuracy(acc)
    if shap is not None and ess is not None:
        table_feature_importance(shap, ess)
    if sig is not None:
        table_significance(sig)
    if abl is not None:
        table_ablation(abl)

    # --- Copy to latex/ folder for Overleaf compatibility ---
    try:
        import shutil
        ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        latex_tables_dir = os.path.join(ROOT, 'latex', 'tables')
        latex_figures_dir = os.path.join(ROOT, 'latex', 'figures')
        
        os.makedirs(latex_tables_dir, exist_ok=True)
        os.makedirs(latex_figures_dir, exist_ok=True)
        
        # Copy tables
        for f in os.listdir(PAPER_TABLES_DIR):
            if f.endswith('.tex'):
                shutil.copy(os.path.join(PAPER_TABLES_DIR, f), os.path.join(latex_tables_dir, f))
                
        # Copy figures
        for f in os.listdir(PAPER_FIGURES_DIR):
            if f.endswith('.pdf'):
                shutil.copy(os.path.join(PAPER_FIGURES_DIR, f), os.path.join(latex_figures_dir, f))
                
        logger.info("Copied all generated assets to latex/ folder.")
    except Exception as e:
        logger.warning(f"Failed to copy assets to latex/ folder: {e}")

    logger.info("All figures & tables done.")


if __name__ == '__main__':
    main()