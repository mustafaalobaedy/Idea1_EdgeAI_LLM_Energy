"""EDA, statistics, tables and figures for the Jetson LLM energy study.

python analyze.py            # everything available so far
Outputs: results/tables/*.csv, results/summary.json, figures/*.png|pdf
"""
import glob
import itertools
import json
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
from scipy import stats  # noqa: E402
from statsmodels.stats.contingency_tables import mcnemar  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RES = os.environ.get('BENCH_RESULTS', os.path.join(ROOT, 'results'))
RUNS = os.path.join(RES, 'runs')
TAB = os.path.join(RES, 'tables')
FIG = os.environ.get('BENCH_FIGURES', os.path.join(ROOT, 'figures'))
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
sys.path.insert(0, os.path.dirname(__file__))
from power_monitor import COLUMNS  # noqa: E402

RNG = np.random.default_rng(2026)
B = 2000
TASKS = ['arc', 'boolq', 'gsm8k']
TASK_LABEL = {'arc': 'ARC-Challenge', 'boolq': 'BoolQ', 'gsm8k': 'GSM8K'}
MODES = ['MODE_15W', 'MODE_30W', 'MODE_40W', 'MAXN']
MODE_LABEL = {'MODE_15W': '15 W', 'MODE_30W': '30 W', 'MODE_40W': '40 W', 'MAXN': 'MAXN'}
# validated palette (dataviz reference, light mode): families = first three categorical slots
FAM_COLOR = {'Qwen2.5': '#2a78d6', 'Llama3.2': '#eb6834', 'Gemma2': '#1baf7a', 'Qwen3': '#4a3aa7'}
MODE_COLOR = {'MODE_15W': '#86b6ef', 'MODE_30W': '#3987e5', 'MODE_40W': '#1c5cab', 'MAXN': '#0d366b'}
QUANT_MARKER = {'Q3_K_M': 'v', 'Q4_K_M': 'o', 'Q5_K_M': 'D', 'Q6_K': 'P', 'Q8_0': 's', 'FP16': '^'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 9.5, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.edgecolor': INK2,
    'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK, 'axes.grid': True,
    'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
    'lines.linewidth': 2, 'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'figure.facecolor': 'white', 'axes.facecolor': 'white', 'legend.frameon': False,
})


def logx(a):
    a.set_xscale('log')
    a.xaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1, 2, 5)))
    a.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
    a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def short(label):
    return label.replace('Qwen2.5-', 'Q').replace('Llama3.2-', 'L').replace('Gemma2-', 'G').replace('_K_M', '').replace('_0', '')


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + '.png'))
    fig.savefig(os.path.join(FIG, name + '.pdf'))
    plt.close(fig)


# ---------------------------------------------------------------- metadata
def model_meta(models):
    cache = os.path.join(ROOT, 'code', 'model_meta.json')
    meta = json.load(open(cache)) if os.path.exists(cache) else {}
    for m in models:
        if m in meta:
            continue
        info = {}
        for host in ('http://127.0.0.1:11500', 'http://127.0.0.1:11434'):
            try:
                s = requests.post(host + '/api/show', json={'model': m}, timeout=30).json()
                tags = requests.get(host + '/api/tags', timeout=30).json()['models']
                size = next(t['size'] for t in tags if t['name'] == m)
                info = dict(params=s['model_info']['general.parameter_count'], size_bytes=size,
                            quant=s['details']['quantization_level'], family_raw=s['details']['family'])
                break
            except Exception:
                continue
        meta[m] = info
    json.dump(meta, open(cache, 'w'), indent=1)
    rows = []
    for m in models:
        i = meta.get(m, {})
        fam = ('Qwen3' if m.startswith('qwen3') else 'Qwen2.5' if m.startswith('qwen2.5')
               else 'Llama3.2' if m.startswith('llama') else 'Gemma2')
        q = i.get('quant', '')
        q = 'FP16' if q.upper() in ('F16', 'FP16') else q.upper().replace('Q4_K_M', 'Q4_K_M')
        size_tag = re.search(r':(\d+\.?\d*b)', m).group(1).upper()
        bpw = i['size_bytes'] * 8 / i['params'] if i else np.nan
        rows.append(dict(model=m, family=fam, size_tag=size_tag, quant=q, params_B=i.get('params', np.nan) / 1e9,
                         size_GB=i.get('size_bytes', np.nan) / 1e9, bpw=bpw,
                         label=f'{fam}-{size_tag} {q}' + (' Instruct-2507' if 'instruct-2507' in m else ' Thinking-2507' if m == 'qwen3:4b' else '')))
    return pd.DataFrame(rows).set_index('model')


# ---------------------------------------------------------------- loading
def load_runs():
    rows = []
    for f in glob.glob(os.path.join(RUNS, '*', '*', '*.jsonl')):
        tag = f.split(os.sep)[-3]
        for line in open(f):
            r = json.loads(line)
            r['tag'] = tag
            rows.append(r)
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    d['think'] = d['think'].fillna('na')
    d['energy_dyn_J'] = d.energy_J - d.p_idle_W * d.wall_s
    d['ttft_s'] = d.load_s + d.prompt_eval_s
    d['parse_fail'] = d.pred.isna()
    d['truncated'] = d.done_reason == 'length'
    return d


def load_idle():
    f = os.path.join(RES, 'idle_baseline.jsonl')
    return pd.DataFrame([json.loads(l) for l in open(f)]) if os.path.exists(f) else pd.DataFrame()


# ---------------------------------------------------------------- statistics
def boot_ci(x, fn=np.mean, b=B):
    x = np.asarray(x)
    idx = RNG.integers(0, len(x), (b, len(x)))
    v = np.array([fn(x[i]) for i in idx])
    return np.percentile(v, [2.5, 97.5])


def jpca_ci(e, c, b=B):
    e, c = np.asarray(e, float), np.asarray(c, float)
    idx = RNG.integers(0, len(e), (b, len(e)))
    se, sc = e[idx].sum(1), c[idx].sum(1)
    v = np.where(sc > 0, se / np.maximum(sc, 1), np.nan)
    return np.nanpercentile(v, [2.5, 97.5])


def summarize(d, keys):
    out = []
    for k, g in d.groupby(keys):
        acc_lo, acc_hi = boot_ci(g.correct.values.astype(float))
        j_lo, j_hi = jpca_ci(g.energy_J, g.correct)
        out.append(dict(zip(keys, k if isinstance(k, tuple) else (k,)),
                        n=len(g), acc=g.correct.mean(), acc_lo=acc_lo, acc_hi=acc_hi,
                        E_query_J=g.energy_J.mean(), E_query_med_J=g.energy_J.median(),
                        E_dyn_query_J=g.energy_dyn_J.mean(),
                        JPCA=g.energy_J.sum() / max(g.correct.sum(), 1), JPCA_lo=j_lo, JPCA_hi=j_hi,
                        latency_s=g.wall_s.mean(), latency_p95_s=g.wall_s.quantile(0.95), ttft_s=g.ttft_s.mean(),
                        power_W=(g.energy_J.sum() / g.wall_s.sum()), p_idle_W=g.p_idle_W.mean(),
                        decode_tps=g.output_tokens.sum() / g.eval_s.sum(),
                        prefill_tps=g.prompt_tokens.sum() / g.prompt_eval_s.clip(lower=1e-6).sum(),
                        J_per_out_token=g.energy_decode_J.sum() / g.output_tokens.sum(),
                        J_per_prompt_token=g.energy_prefill_J.sum() / g.prompt_tokens.sum(),
                        out_tokens=g.output_tokens.mean(), prompt_tokens=g.prompt_tokens.mean(),
                        parse_fail=g.parse_fail.mean(), truncated=g.truncated.mean(),
                        T_tj_max=g.T_tj_max.max(), gpu_freq=g.gpu_freq_mean.mean(),
                        EDP=(g.energy_J * g.wall_s).mean()))
    return pd.DataFrame(out)


def pareto(df, x, y):
    """Indices of points minimizing x and maximizing y."""
    s = df.sort_values([x, y], ascending=[True, False])
    best, keep = -np.inf, []
    for i, r in s.iterrows():
        if r[y] > best:
            keep.append(i)
            best = r[y]
    return keep


# ---------------------------------------------------------------- EDA figures
def fig_trace_example():
    fs = sorted(glob.glob(os.path.join(RUNS, 'E1_main', 'MAXN', '*qwen2.5_3b-instruct-q4_K_M__gsm8k__trace*.npz')))
    if not fs:
        return
    z = np.load(fs[0])
    tr = z['trace']
    t = tr[:, 0] - tr[0, 0]
    m = (t > 30) & (t < 90)
    fig, ax = plt.subplots(2, 1, figsize=(7.2, 3.6), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
    rails = ['P_VDD_GPU_SOC', 'P_VDD_CPU_CV', 'P_VIN_SYS_5V0', 'P_VDDQ_VDD2_1V8AO']
    cols = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
    ys = [tr[m, COLUMNS.index(r)] for r in rails]
    ax[0].stackplot(t[m] - 30, ys, labels=[r[2:] for r in rails], colors=cols, linewidth=0.3, edgecolor='white')
    ax[0].set_ylabel('Power (W)')
    ax[0].legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.22))
    ax[1].plot(t[m] - 30, tr[m, COLUMNS.index('gpu_load')], color=INK2, lw=1)
    ax[1].set_ylabel('GPU load (%)')
    ax[1].set_xlabel('Time (s)')
    save(fig, 'fig02_power_trace_example')


def fig_idle(idle):
    if idle.empty:
        return
    g = idle.groupby('mode')[[c for c in idle.columns if c.startswith('P_')]].median().reindex(
        [m for m in MODES if m in set(idle['mode'])])
    rails = ['P_VDD_GPU_SOC', 'P_VDD_CPU_CV', 'P_VIN_SYS_5V0', 'P_VDDQ_VDD2_1V8AO']
    cols = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    bottom = np.zeros(len(g))
    for r, c in zip(rails, cols):
        ax.bar([MODE_LABEL[m] for m in g.index], g[r], bottom=bottom, color=c, label=r[2:], width=0.6,
               edgecolor='white', linewidth=1)
        bottom += g[r].values
    for i, v in enumerate(bottom):
        ax.text(i, v + 0.2, f'{v:.1f} W', ha='center', fontsize=7, color=INK)
    ax.set_ylabel('Idle module power (W)')
    ax.legend(fontsize=6, loc='upper left', bbox_to_anchor=(1, 1))
    save(fig, 'fig03_idle_power_by_mode')


def fig_token_eda(d):
    e1 = d[(d.tag == 'E1_main')]
    if e1.empty:
        return
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5))
    first = e1[e1.model == e1.model.iloc[0]]
    for t, c in zip(TASKS, FAM_COLOR.values()):
        x = first[first.task == t].prompt_tokens
        if len(x):
            ax[0].hist(x, bins=30, alpha=0.75, color=c, label=TASK_LABEL[t], edgecolor='white', linewidth=0.4)
    ax[0].set_xlabel('Prompt length (tokens, Qwen tokenizer)')
    ax[0].set_ylabel('Questions')
    ax[0].legend()
    g = e1[e1.task == 'gsm8k']
    order = g.groupby('model').output_tokens.median().sort_values().index
    meta = model_meta(list(order))
    ax[1].boxplot([g[g.model == m].output_tokens for m in order], vert=False, widths=0.6, showfliers=False,
                  medianprops=dict(color=INK, lw=1.2), boxprops=dict(color=INK2, lw=0.8),
                  whiskerprops=dict(color=INK2, lw=0.8), capprops=dict(color=INK2, lw=0.8))
    ax[1].set_yticks(range(1, len(order) + 1), [meta.loc[m, 'label'] for m in order], fontsize=5.5)
    ax[1].set_xlabel('GSM8K output length (tokens)')
    save(fig, 'fig04_token_distributions')


def fig_energy_ecdf(d, meta):
    e1 = d[d.tag == 'E1_main']
    if e1.empty:
        return
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3), sharey=True)
    pick = [m for m in ['qwen2.5:0.5b-instruct-q4_K_M', 'qwen2.5:3b-instruct-q4_K_M', 'qwen2.5:7b-instruct-q4_K_M',
                        'qwen2.5:3b-instruct-fp16'] if m in set(e1.model)]
    seq = ['#86b6ef', '#3987e5', '#1c5cab', '#0d366b']
    for a, t in zip(ax, TASKS):
        for m, c in zip(pick, seq):
            x = np.sort(e1[(e1.model == m) & (e1.task == t)].energy_J.values)
            if len(x):
                a.plot(x, np.arange(1, len(x) + 1) / len(x), color=c, lw=1.6, label=meta.loc[m, 'label'])
        logx(a)
        a.set_title(TASK_LABEL[t])
        a.set_xlabel('Energy per query (J)')
    ax[0].set_ylabel('ECDF')
    ax[2].legend(fontsize=6, loc='lower right')
    save(fig, 'fig05_energy_ecdf')


# ---------------------------------------------------------------- results figures
def fig_pareto(s1, meta):
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 3.1), gridspec_kw=dict(wspace=0.3))
    for a, t in zip(ax, TASKS):
        g = s1[s1.task == t].copy()
        if g.empty:
            continue
        for _, r in g.iterrows():
            mm = meta.loc[r.model]
            a.scatter(r.E_query_J, r.acc * 100, s=10 + 9 * mm.params_B, color=FAM_COLOR[mm.family],
                      marker=QUANT_MARKER.get(mm.quant, 'o'), edgecolor='white', linewidth=0.8, zorder=3)
        pf = g.loc[pareto(g, 'E_query_J', 'acc')]
        a.step(pf.E_query_J, pf.acc * 100, where='post', color=INK2, lw=1, ls='--', zorder=2)
        logx(a)
        # label every point; adjustText pushes labels apart so they do not overlap each other or the markers
        texts = [a.text(r.E_query_J, r.acc * 100, short(meta.loc[r.model, 'label']), fontsize=6, color=INK, zorder=4)
                 for _, r in g.iterrows()]
        try:
            from adjustText import adjust_text
            adjust_text(texts, ax=a, expand=(1.3, 1.6), arrowprops=dict(arrowstyle='-', color=INK2, lw=0.4),
                        min_arrow_len=3, only_move={'points': 'xy', 'text': 'xy'})
        except ImportError:
            pass
        a.set_title(TASK_LABEL[t])
        a.set_xlabel('Mean energy per query (J, log)')
    ax[0].set_ylabel('Accuracy (%)')
    h = [plt.Line2D([], [], marker='o', ls='', color=c, label=f) for f, c in FAM_COLOR.items() if f != 'Qwen3']
    h += [plt.Line2D([], [], marker=mk, ls='', color=INK2, label=q) for q, mk in QUANT_MARKER.items()]
    fig.legend(handles=h, ncol=9, loc='upper center', bbox_to_anchor=(0.5, 1.07), fontsize=6, markerscale=0.7)
    save(fig, 'fig06_pareto_accuracy_energy')


def fig_quant_ladder(s1, meta):
    g = s1[s1.model.str.startswith('qwen2.5:3b')].copy()
    if g.empty:
        return
    g['bpw'] = g.model.map(meta.bpw)
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4), gridspec_kw=dict(wspace=0.45))
    cols = dict(zip(TASKS, ['#2a78d6', '#eb6834', '#1baf7a']))
    for t in TASKS:
        h = g[g.task == t].sort_values('bpw')
        if h.empty:
            continue
        ax[0].errorbar(h.bpw, h.acc * 100, yerr=[(h.acc - h.acc_lo) * 100, (h.acc_hi - h.acc) * 100], color=cols[t],
                       marker='o', ms=4, lw=1.5, capsize=2, label=TASK_LABEL[t])
        ax[1].errorbar(h.bpw, h.JPCA, yerr=[h.JPCA - h.JPCA_lo, h.JPCA_hi - h.JPCA], color=cols[t], marker='o',
                       ms=4, lw=1.5, capsize=2)
    h = g[g.task == 'gsm8k'].sort_values('bpw')
    ax[2].plot(h.bpw, h.decode_tps, marker='o', ms=4, color=INK)
    ax[0].set_ylabel('Accuracy (%)')
    ax[1].set_ylabel('Joules per correct answer')
    ax[1].set_yscale('log')
    ax[1].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax[1].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[2].set_ylabel('Decode throughput (tok/s)')
    for a in ax:
        a.set_xlabel('Bits per weight')
    ax[0].legend(fontsize=6)
    save(fig, 'fig07_quantization_ladder')


def energy_law(s1, meta):
    g = s1.copy()
    g['size_GB'] = g.model.map(meta.size_GB)
    g['params_B'] = g.model.map(meta.params_B)
    dec = g[g.task == 'gsm8k']
    if len(dec) < 3:
        return {}
    lr = stats.linregress(dec.size_GB, dec.J_per_out_token)
    lr2 = stats.linregress(dec.size_GB, 1 / dec.decode_tps)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5))
    for _, r in dec.iterrows():
        mm = meta.loc[r.model]
        for a, y in ((ax[0], r.J_per_out_token), (ax[1], 1000 / r.decode_tps)):
            a.scatter(r.size_GB, y, color=FAM_COLOR[mm.family], marker=QUANT_MARKER.get(mm.quant, 'o'), s=28,
                      edgecolor='white', linewidth=0.8, zorder=3)
    xx = np.linspace(0, dec.size_GB.max() * 1.05, 50)
    ax[0].plot(xx, lr.intercept + lr.slope * xx, color=INK2, lw=1, ls='--')
    ax[0].text(0.03, 0.9, f'E/token = {lr.intercept:.3f} + {lr.slope:.3f}·GB\n$R^2$ = {lr.rvalue**2:.3f}',
               transform=ax[0].transAxes, fontsize=6.5, va='top')
    ax[1].plot(xx, 1000 * (lr2.intercept + lr2.slope * xx), color=INK2, lw=1, ls='--')
    ax[1].text(0.03, 0.9, f'$R^2$ = {lr2.rvalue**2:.3f}', transform=ax[1].transAxes, fontsize=6.5, va='top')
    ax[0].set_ylabel('Decode energy per output token (J)')
    ax[1].set_ylabel('Time per output token (ms)')
    for a in ax:
        a.set_xlabel('Model weight file size (GB)')
    h = [plt.Line2D([], [], marker='o', ls='', color=c, label=f) for f, c in FAM_COLOR.items() if f != 'Qwen3']
    h += [plt.Line2D([], [], marker=mk, ls='', color=INK2, label=q) for q, mk in QUANT_MARKER.items()]
    fig.legend(handles=h, ncol=9, loc='upper center', bbox_to_anchor=(0.5, 1.06), fontsize=6, markerscale=0.7)
    save(fig, 'fig08_energy_per_token_vs_size')
    # effective memory bandwidth implied by slope of time/token vs bytes
    bw = 1 / lr2.slope if lr2.slope > 0 else np.nan
    return dict(J_per_token_slope_per_GB=lr.slope, J_per_token_intercept=lr.intercept, J_per_token_R2=lr.rvalue ** 2,
                J_per_token_slope_p=lr.pvalue, s_per_token_slope_per_GB=lr2.slope, s_per_token_R2=lr2.rvalue ** 2,
                implied_effective_bandwidth_GBps=bw, n_configs=len(dec))


def regression(d1, meta):
    g = d1.copy()
    g['size_GB'] = g.model.map(meta.size_GB)
    g['family'] = g.model.map(meta.family)
    g['logE'] = np.log(g.energy_J)
    g['log_out'] = np.log1p(g.output_tokens)
    g['log_in'] = np.log(g.prompt_tokens)
    g['log_size'] = np.log(g.size_GB)
    f = 'logE ~ log_size + log_out + log_in' + (' + C(family)' if g.family.nunique() > 1 else '')
    fit = smf.ols(f, data=g).fit(cov_type='HC3')
    with open(os.path.join(TAB, 'regression_logE.txt'), 'w') as f:
        f.write(fit.summary().as_text())
    return dict(R2=fit.rsquared, n=int(fit.nobs), coef={k: float(v) for k, v in fit.params.items()},
                p={k: float(v) for k, v in fit.pvalues.items()})


def fig_modes(s2, meta):
    if s2.empty or s2['mode'].nunique() < 2:
        return {}
    base = s2[s2['mode'] == 'MAXN'].set_index(['model', 'task'])
    rel = s2.join(base[['E_query_J', 'latency_s', 'EDP']], on=['model', 'task'], rsuffix='_maxn')
    for c in ['E_query_J', 'latency_s', 'EDP']:
        rel[c + '_rel'] = rel[c] / rel[c + '_maxn']
    rel.to_csv(os.path.join(TAB, 'T4_power_modes_relative.csv'), index=False)
    fig, ax = plt.subplots(2, 3, figsize=(7.2, 4.2), sharex=True)
    models = sorted(s2.model.unique(), key=lambda m: meta.loc[m, 'size_GB'])
    x = np.arange(len(models))
    w = 0.2
    modes = [m for m in MODES if m in set(s2['mode'])]
    for j, t in enumerate(TASKS):
        for k, md in enumerate(modes):
            h = rel[(rel.task == t) & (rel['mode'] == md)].set_index('model').reindex(models)
            ax[0, j].bar(x + (k - 1.5) * w, h.E_query_J_rel, w, color=MODE_COLOR[md], label=MODE_LABEL[md],
                         edgecolor='white', linewidth=0.5)
            ax[1, j].bar(x + (k - 1.5) * w, h.latency_s_rel, w, color=MODE_COLOR[md], edgecolor='white', linewidth=0.5)
        ax[0, j].axhline(1, color=INK2, lw=0.8)
        ax[1, j].axhline(1, color=INK2, lw=0.8)
        ax[0, j].set_title(TASK_LABEL[t])
        ax[1, j].set_xticks(x, [short(meta.loc[m, 'label']) for m in models], rotation=70, fontsize=7)
    ax[0, 0].set_ylabel('Energy / query\n(relative to MAXN)')
    ax[1, 0].set_ylabel('Latency / query\n(relative to MAXN)')
    ax[0, 2].legend(fontsize=6, loc='upper left', bbox_to_anchor=(1, 1))
    save(fig, 'fig09_power_modes')
    summ = rel.groupby(['mode', 'task'])[['E_query_J_rel', 'latency_s_rel', 'EDP_rel']].agg(['mean', 'min', 'max'])
    summ.to_csv(os.path.join(TAB, 'T4b_power_modes_summary.csv'))
    return {f'{m}|{t}': dict(E_rel=float(r[('E_query_J_rel', 'mean')]), lat_rel=float(r[('latency_s_rel', 'mean')]),
                             EDP_rel=float(r[('EDP_rel', 'mean')])) for (m, t), r in summ.iterrows()}


def mode_consistency(d):
    """Are greedy outputs identical across power modes? (determinism check)"""
    e2 = d[d.tag == 'E2_modes']
    if e2['mode'].nunique() < 2:
        return {}
    p = e2.pivot_table(index=['model', 'task', 'qid'], columns='mode', values='response', aggfunc='first')
    same = p.apply(lambda r: r.dropna().nunique() == 1, axis=1)
    pc = e2.pivot_table(index=['model', 'task', 'qid'], columns='mode', values='correct', aggfunc='first')
    same_c = pc.apply(lambda r: r.dropna().nunique() == 1, axis=1)
    return dict(identical_response_frac=float(same.mean()), identical_correctness_frac=float(same_c.mean()),
                n=int(len(p)))


def repeatability(d):
    """E1 vs E2 at MAXN on overlapping questions: energy agreement (test-retest)."""
    a = d[(d.tag == 'E1_main') & (d['mode'] == 'MAXN')].set_index(['model', 'task', 'qid'])
    b = d[(d.tag == 'E2_modes') & (d['mode'] == 'MAXN')].set_index(['model', 'task', 'qid'])
    j = a[['energy_J', 'wall_s']].join(b[['energy_J', 'wall_s']], rsuffix='_2', how='inner')
    if len(j) < 10:
        return {}
    rel_diff = (j.energy_J_2 - j.energy_J) / ((j.energy_J_2 + j.energy_J) / 2)
    agg = j.groupby(level=[0, 1]).sum()
    agg_diff = (agg.energy_J_2 - agg.energy_J) / agg.energy_J
    r = stats.pearsonr(np.log(j.energy_J), np.log(j.energy_J_2))
    return dict(n=len(j), per_query_median_abs_reldiff=float(rel_diff.abs().median()),
                per_query_log_pearson_r=float(r.statistic), config_total_reldiff_mean=float(agg_diff.mean()),
                config_total_reldiff_absmax=float(agg_diff.abs().max()))


def fig_think(s3, d, meta):
    if s3.empty:
        return {}
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    out = {}
    for a, t in zip(ax, ['arc', 'gsm8k']):
        for m, c in zip(['qwen3:1.7b', 'qwen3:4b', 'qwen3:8b'], ['#86b6ef', '#3987e5', '#0d366b']):
            off = s3[(s3.model == m) & (s3.task == t) & (s3.think == 'off')]
            on = s3[(s3.model == m) & (s3.task == t) & (s3.think == 'on')]
            if off.empty or on.empty:
                continue
            o, n = off.iloc[0], on.iloc[0]
            a.annotate('', (n.E_query_J, n.acc * 100), (o.E_query_J, o.acc * 100),
                       arrowprops=dict(arrowstyle='->', color=c, lw=1.5))
            a.scatter([o.E_query_J], [o.acc * 100], color=c, marker='o', s=30, zorder=3, edgecolor='white')
            a.scatter([n.E_query_J], [n.acc * 100], color=c, marker='*', s=70, zorder=3, edgecolor='white')
            a.annotate(meta.loc[m, 'size_TAG'] if 'size_TAG' in meta else meta.loc[m, 'size_tag'],
                       (n.E_query_J, n.acc * 100), xytext=(4, 2), textcoords='offset points', fontsize=6)
            # paired test on same questions
            g = d[(d.tag == 'E3_think') & (d.model == m) & (d.task == t)]
            p = g.pivot_table(index='qid', columns='think', values='correct', aggfunc='first').dropna()
            tb = pd.crosstab(p['off'].astype(bool), p['on'].astype(bool)).reindex(index=[False, True], columns=[False, True],
                                                                                  fill_value=0)
            mt = mcnemar(tb.values, exact=True)
            out[f'{m}|{t}'] = dict(acc_off=o.acc, acc_on=n.acc, E_off=o.E_query_J, E_on=n.E_query_J,
                                   E_mult=n.E_query_J / o.E_query_J, JPCA_off=o.JPCA, JPCA_on=n.JPCA,
                                   marginal_J_per_extra_correct=(n.E_query_J - o.E_query_J) / (n.acc - o.acc)
                                   if n.acc > o.acc else np.inf,
                                   mcnemar_p=float(mt.pvalue), out_tok_off=o.out_tokens, out_tok_on=n.out_tokens,
                                   trunc_on=n.truncated)
        logx(a)
        a.set_title(TASK_LABEL[t])
        a.set_xlabel('Mean energy per query (J, log)')
    ax[0].set_ylabel('Accuracy (%)')
    h = [plt.Line2D([], [], marker='o', ls='', color=INK2, label='thinking off'),
         plt.Line2D([], [], marker='*', ls='', color=INK2, ms=9, label='thinking on')]
    ax[1].legend(handles=h, loc='lower right')
    save(fig, 'fig10_thinking_tradeoff')
    return out


# ---------------------------------------------------------------- cascades
def cascades(d1, meta):
    """Agreement cascade: run small models s1,s2; accept if they agree on a parsable answer, else escalate to L.
    Policy selected on a random half of questions, evaluated on the held-out half."""
    res, frontier_rows = {}, []
    models = sorted(d1.model.unique())
    for t in TASKS:
        g = d1[d1.task == t]
        P = g.pivot_table(index='qid', columns='model', values='pred', aggfunc='first')
        C = g.pivot_table(index='qid', columns='model', values='correct', aggfunc='first').astype(float)
        E = g.pivot_table(index='qid', columns='model', values='energy_J', aggfunc='first')
        L = g.pivot_table(index='qid', columns='model', values='wall_s', aggfunc='first')
        common = P.dropna(axis=1, thresh=int(0.99 * len(P))).columns
        P, C, E, L = (x[common].dropna() for x in (P, C, E, L))
        q = np.array(P.index)
        perm = RNG.permutation(len(q))
        sel, test = q[perm[: len(q) // 2]], q[perm[len(q) // 2:]]

        def evaluate(qs, s1, s2, big):
            agree = (P.loc[qs, s1] == P.loc[qs, s2]) & P.loc[qs, s1].notna()
            corr = np.where(agree, C.loc[qs, s1], C.loc[qs, big])
            en = E.loc[qs, s1] + E.loc[qs, s2] + np.where(agree, 0, E.loc[qs, big])
            lat = L.loc[qs, s1] + L.loc[qs, s2] + np.where(agree, 0, L.loc[qs, big])
            return corr.mean(), en.mean(), lat.mean(), 1 - agree.mean()

        cands = []
        size = meta.size_GB
        for s1, s2 in itertools.combinations(common, 2):
            for big in common:
                if big in (s1, s2) or size[big] <= max(size[s1], size[s2]):
                    continue
                a, e, l, esc = evaluate(sel, s1, s2, big)
                cands.append((s1, s2, big, a, e))
        cdf = pd.DataFrame(cands, columns=['s1', 's2', 'big', 'acc', 'E'])
        singles = pd.DataFrame([(m, C.loc[sel, m].mean(), E.loc[sel, m].mean()) for m in common],
                               columns=['m', 'acc', 'E'])
        best_single = singles.sort_values('acc').iloc[-1]
        # cascades on the selection set's Pareto frontier
        pf = cdf.loc[pareto(cdf, 'E', 'acc')]
        # headline: cheapest cascade whose selection accuracy >= best single model accuracy - 1pp
        ok = pf[pf.acc >= best_single.acc - 0.01]
        pick = ok.sort_values('E').iloc[0] if len(ok) else pf.sort_values('acc').iloc[-1]
        a, e, l, esc = evaluate(test, pick.s1, pick.s2, pick.big)
        bm = best_single.m
        ref_a, ref_e, ref_l = C.loc[test, bm].mean(), E.loc[test, bm].mean(), L.loc[test, bm].mean()
        # paired test on test split
        agree = (P.loc[test, pick.s1] == P.loc[test, pick.s2]) & P.loc[test, pick.s1].notna()
        corr_c = np.where(agree, C.loc[test, pick.s1], C.loc[test, pick.big]).astype(bool)
        tb = pd.crosstab(pd.Series(C.loc[test, bm].astype(bool).values), pd.Series(corr_c)).reindex(
            index=[False, True], columns=[False, True], fill_value=0)
        # oracle router: cheapest correct model per question (lower bound)
        Ec = E.loc[test].where(C.loc[test] == 1)
        oracle_e = np.where(Ec.notna().any(axis=1), Ec.min(axis=1), E.loc[test].min(axis=1)).mean()
        oracle_a = (C.loc[test].max(axis=1)).mean()
        res[t] = dict(policy=f'{pick.s1} + {pick.s2} -> {pick.big}', s1=pick.s1, s2=pick.s2, big=pick.big,
                      test_acc=a, test_E=e, test_latency=l, escalation_rate=esc, ref_model=bm, ref_acc=ref_a,
                      ref_E=ref_e, ref_latency=ref_l, energy_saving=1 - e / ref_e,
                      mcnemar_p=float(mcnemar(tb.values, exact=True).pvalue), n_test=len(test), n_sel=len(sel),
                      oracle_acc=oracle_a, oracle_E=oracle_e, n_candidates=len(cdf))
        # test-split frontier for figure
        for _, r in pf.iterrows():
            a2, e2, _, _ = evaluate(test, r.s1, r.s2, r.big)
            frontier_rows.append(dict(task=t, kind='cascade', acc=a2, E=e2, name=f'{r.s1}+{r.s2}->{r.big}'))
        for m in common:
            frontier_rows.append(dict(task=t, kind='single', acc=C.loc[test, m].mean(), E=E.loc[test, m].mean(), name=m))
    fr = pd.DataFrame(frontier_rows)
    fr.to_csv(os.path.join(TAB, 'T6_cascade_frontier_test.csv'), index=False)
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for a, t in zip(ax, TASKS):
        h = fr[fr.task == t]
        if h.empty:
            continue
        s = h[h.kind == 'single']
        c = h[h.kind == 'cascade'].sort_values('E')
        a.scatter(s.E, s.acc * 100, color='#2a78d6', s=14, label='single model', zorder=3, edgecolor='white', lw=0.6)
        a.scatter(c.E, c.acc * 100, color='#eb6834', marker='D', s=12, label='agreement cascade', zorder=3,
                  edgecolor='white', lw=0.6)
        r = res[t]
        a.scatter([r['test_E']], [r['test_acc'] * 100], s=90, facecolor='none', edgecolor=INK, lw=1.2, zorder=4)
        logx(a)
        a.set_title(TASK_LABEL[t])
        a.set_xlabel('Mean energy per query (J, log)')
    ax[0].set_ylabel('Accuracy (%) — held-out split')
    ax[2].legend(loc='upper left', fontsize=6)
    save(fig, 'fig11_cascade')
    return res


def fig_thermal(meta):
    fs = sorted(glob.glob(os.path.join(RUNS, 'E1_main', 'MAXN', '*qwen2.5_7b-instruct-q8_0__gsm8k__trace*.npz')))
    if not fs:
        return {}
    tr = np.load(fs[0])['trace']
    t = (tr[:, 0] - tr[0, 0]) / 60
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.2))
    k = 250  # 5 s rolling mean
    sm = lambda x: pd.Series(x).rolling(k, min_periods=1).mean().values  # noqa: E731
    for name, c in [('T_tj', '#e34948'), ('T_gpu', '#2a78d6'), ('T_cpu', '#eb6834')]:
        ax[0].plot(t, sm(tr[:, COLUMNS.index(name)]), color=c, lw=1.2, label=name[2:].upper())
    ax[0].set_ylabel('Temperature (°C)')
    ax[0].legend(fontsize=7, loc='upper left', ncol=3)
    ax[0].set_ylim(top=ax[0].get_ylim()[1] + 1.2)
    ax[1].plot(t, sm(tr[:, COLUMNS.index('gpu_freq_mhz')]), color=INK, lw=1.2)
    ax[1].set_ylabel('GPU clock (MHz)')
    for a in ax:
        a.set_xlabel('Time (min)')
    save(fig, 'fig12_thermal_sustained')
    return dict(minutes=float(t[-1]), tj_start=float(np.nanmean(tr[:250, COLUMNS.index('T_tj')])),
                tj_end=float(np.nanmean(tr[-250:, COLUMNS.index('T_tj')])),
                tj_max=float(np.nanmax(tr[:, COLUMNS.index('T_tj')])),
                gpu_freq_min=float(np.min(tr[:, COLUMNS.index('gpu_freq_mhz')])),
                gpu_freq_median=float(np.median(tr[:, COLUMNS.index('gpu_freq_mhz')])))


def key_tests(d1, meta):
    """McNemar tests on accuracy for prespecified comparisons (Holm-corrected)."""
    pairs = [('qwen2.5:7b-instruct-q4_K_M', 'qwen2.5:3b-instruct-fp16'),
             ('qwen2.5:3b-instruct-q4_K_M', 'qwen2.5:3b-instruct-fp16'),
             ('qwen2.5:3b-instruct-q3_K_M', 'qwen2.5:3b-instruct-q4_K_M'),
             ('qwen2.5:3b-instruct-q4_K_M', 'qwen2.5:3b-instruct-q8_0'),
             ('qwen2.5:7b-instruct-q4_K_M', 'qwen2.5:7b-instruct-q8_0'),
             ('llama3.2:3b-instruct-q4_K_M', 'llama3.2:3b-instruct-fp16'),
             ('qwen2.5:1.5b-instruct-q4_K_M', 'qwen2.5:1.5b-instruct-fp16'),
             ('gemma2:2b-instruct-q4_K_M', 'gemma2:2b-instruct-fp16')]
    rows = []
    for a, b in pairs:
        for t in TASKS:
            g = d1[(d1.task == t) & d1.model.isin([a, b])]
            p = g.pivot_table(index='qid', columns='model', values='correct', aggfunc='first').dropna()
            ep = g.pivot_table(index='qid', columns='model', values='energy_J', aggfunc='first').reindex(p.index)
            if a not in p or b not in p or len(p) < 10:
                continue
            tb = pd.crosstab(p[a].astype(bool), p[b].astype(bool)).reindex(index=[False, True], columns=[False, True],
                                                                         fill_value=0)
            # paired bootstrap of JPCA ratio
            idx = RNG.integers(0, len(p), (B, len(p)))
            ea, eb, ca, cb = ep[a].values, ep[b].values, p[a].values.astype(float), p[b].values.astype(float)
            ratio = (ea[idx].sum(1) / np.maximum(ca[idx].sum(1), 1)) / (eb[idx].sum(1) / np.maximum(cb[idx].sum(1), 1))
            rows.append(dict(A=meta.loc[a, 'label'], B=meta.loc[b, 'label'], task=t, n=len(p), acc_A=ca.mean(),
                             acc_B=cb.mean(), diff_pp=(ca.mean() - cb.mean()) * 100,
                             p_mcnemar=mcnemar(tb.values, exact=True).pvalue,
                             JPCA_ratio=(ea.sum() / max(ca.sum(), 1)) / (eb.sum() / max(cb.sum(), 1)),
                             JPCA_ratio_lo=np.percentile(ratio, 2.5), JPCA_ratio_hi=np.percentile(ratio, 97.5)))
    df = pd.DataFrame(rows)
    if len(df):
        df['p_holm'] = multipletests(df.p_mcnemar, method='holm')[1]
    df.to_csv(os.path.join(TAB, 'T7_pairwise_tests.csv'), index=False)
    return df


def main():
    d = load_runs()
    if d.empty:
        print('no data yet')
        return
    idle = load_idle()
    meta = model_meta(sorted(d.model.unique()))
    meta.to_csv(os.path.join(TAB, 'T2_models.csv'))
    summary = dict(n_queries_total=int(len(d)), by_tag=d.groupby('tag').size().to_dict(),
                   total_energy_kJ=float(d.energy_J.sum() / 1000), total_hours=float(d.wall_s.sum() / 3600))
    if not idle.empty:
        summary['idle_W'] = idle.groupby('mode').P_total.median().to_dict()
    fig_idle(idle)
    fig_trace_example()
    fig_token_eda(d)
    fig_energy_ecdf(d, meta)

    d1 = d[(d.tag == 'E1_main') & (d['mode'] == 'MAXN')]
    if len(d1):
        s1 = summarize(d1, ['model', 'task'])
        s1 = s1.join(meta, on='model')
        s1.to_csv(os.path.join(TAB, 'T3_main_results.csv'), index=False)
        fig_pareto(s1, meta)
        fig_quant_ladder(s1, meta)
        summary['energy_law'] = energy_law(s1, meta)
        summary['regression'] = regression(d1, meta)
        summary['pareto'] = {t: [meta.loc[m, 'label'] for m in s1.loc[pareto(s1[s1.task == t], 'E_query_J', 'acc'),
                                                                        'model']] for t in TASKS if (s1.task == t).any()}
        summary['tests'] = key_tests(d1, meta).to_dict('records')
        summary['thermal'] = fig_thermal(meta)
        if d1.model.nunique() > 5:
            summary['cascade'] = cascades(d1, meta)
        summary['parse_fail_rate'] = float(d1.parse_fail.mean())
    d2 = d[d.tag == 'E2_modes']
    if len(d2):
        s2 = summarize(d2, ['mode', 'model', 'task']).join(meta, on='model')
        s2.to_csv(os.path.join(TAB, 'T4_power_modes.csv'), index=False)
        summary['modes'] = fig_modes(s2, meta)
        summary['mode_consistency'] = mode_consistency(d)
        summary['repeatability'] = repeatability(d)
        jp = s2.groupby(['mode', 'task']).JPCA.median().unstack()
        summary['modes_JPCA_median'] = jp.to_dict()
    bad = d[(d.tag == 'E3_think') & (d.model == 'qwen3:4b') & (d.think == 'off')]
    if len(bad):
        summary['thinking_4b_switch_ineffective'] = {t: dict(acc=float(g.correct.mean()), truncated=float(g.truncated.mean()),
                                                              out_tokens=float(g.output_tokens.mean()), n=int(len(g)))
                                                     for t, g in bad.groupby('task')}
        d = d.drop(bad.index)
    d.loc[(d.tag == 'E3_think') & (d.model == 'qwen3:4b-instruct-2507-q4_K_M'), 'model'] = 'qwen3:4b'
    d3 = d[d.tag == 'E3_think']
    if len(d3):
        s3 = summarize(d3, ['model', 'task', 'think']).join(meta, on='model')
        s3.to_csv(os.path.join(TAB, 'T5_thinking.csv'), index=False)
        summary['thinking'] = fig_think(s3, d, meta)
    json.dump(summary, open(os.path.join(RES, 'summary.json'), 'w'), indent=1, default=float)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('tests',)}, indent=1, default=float)[:6000])


if __name__ == '__main__':
    main()
