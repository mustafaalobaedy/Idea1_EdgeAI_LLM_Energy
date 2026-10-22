"""Turn the CSV outputs of code/analyze.py into the compact Markdown tables used by build_paper.py.

python make_tables.py      -> paper/tables/T2_models.md, T3_main.md, T4_modes.md, T5_thinking.md, T6_cascade.md
Missing inputs are skipped with a note, so the script can be run on partial data.
"""
import os
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TAB_IN = ROOT / 'results' / 'tables'
TAB_OUT = HERE / 'tables'
TASKS = [('arc', 'ARC'), ('boolq', 'BoolQ'), ('gsm8k', 'GSM8K')]
MODE_LABEL = {'MODE_15W': '15 W', 'MODE_30W': '30 W', 'MODE_40W': '40 W', 'MAXN': 'MAXN'}


def md(rows, header):
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    for r in rows:
        lines.append('| ' + ' | '.join(str(x) for x in r) + ' |')
    return '\n'.join(lines) + '\n'


def write(name, text):
    (TAB_OUT / name).write_text(text)
    print('wrote', name)


def f(x, nd=1):
    try:
        return f'{float(x):,.{nd}f}'
    except (TypeError, ValueError):
        return '–'


def pct(x):
    return f(100 * float(x), 1)


def model_order(meta):
    fam_rank = {'Qwen2.5': 0, 'Llama3.2': 1, 'Gemma2': 2, 'Qwen3': 3}
    q_rank = {'Q3_K_M': 0, 'Q4_K_M': 1, 'Q5_K_M': 2, 'Q6_K': 3, 'Q8_0': 4, 'FP16': 5, '': 6}
    return sorted(meta.index, key=lambda m: (fam_rank.get(meta.loc[m, 'family'], 9), meta.loc[m, 'params_B'],
                                             q_rank.get(str(meta.loc[m, 'quant']), 6)))


def main():
    TAB_OUT.mkdir(exist_ok=True)
    if not (TAB_IN / 'T2_models.csv').exists():
        print('no T2_models.csv yet'); return
    meta = pd.read_csv(TAB_IN / 'T2_models.csv').set_index('model')
    order = model_order(meta)
    rows = [[m, meta.loc[m, 'family'], f(meta.loc[m, 'params_B'], 2), meta.loc[m, 'quant'],
             f(meta.loc[m, 'size_GB'], 2), f(meta.loc[m, 'bpw'], 1)] for m in order]
    write('T2_models.md', md(rows, ['Ollama tag', 'Family', 'Params (B)', 'Quant.', 'File size (GB)', 'Bits per weight']))

    # main grid: one row per model, accuracy / energy per query / JPCA for each task
    if (TAB_IN / 'T3_main_results.csv').exists():
        s = pd.read_csv(TAB_IN / 'T3_main_results.csv')
        rows = []
        for m in order:
            g = s[s.model == m]
            if g.empty:
                continue
            r = [meta.loc[m, 'label']]
            for t, _ in TASKS:
                x = g[g.task == t]
                r += [pct(x.acc.iloc[0]), f(x.E_query_J.iloc[0], 1), f(x.JPCA.iloc[0], 1)] if len(x) else ['–'] * 3
            rows.append(r)
        head = ['Model'] + [f'{lab} {k}' for _, lab in TASKS for k in ('acc. (%)', 'E (J)', 'JPCA (J)')]
        write('T3_main.md', md(rows, head))
        # latency and throughput companion table
        rows = []
        for m in order:
            g = s[s.model == m]
            if g.empty:
                continue
            r = [meta.loc[m, 'label']]
            for t, _ in TASKS:
                x = g[g.task == t]
                r += [f(x.latency_s.iloc[0], 2), f(x.decode_tps.iloc[0], 1)] if len(x) else ['–'] * 2
            x = g[g.task == 'arc']
            r += [f(x.ttft_s.iloc[0], 2) if len(x) else '–', pct(g.truncated.mean()), pct(g.parse_fail.mean())]
            rows.append(r)
        head = ['Model'] + [f'{lab} {k}' for _, lab in TASKS for k in ('lat. (s)', 'tok/s')] + \
               ['TTFT ARC (s)', 'Trunc. (%)', 'Parse fail (%)']
        write('T3b_latency.md', md(rows, head))

    # power modes: energy, latency and JPCA relative to MAXN, per task (median over the 8 models)
    if (TAB_IN / 'T4_power_modes.csv').exists():
        s = pd.read_csv(TAB_IN / 'T4_power_modes.csv')
        base = s[s['mode'] == 'MAXN'].set_index(['model', 'task'])
        rows = []
        for mode in ['MODE_15W', 'MODE_30W', 'MODE_40W', 'MAXN']:
            g = s[s['mode'] == mode].set_index(['model', 'task'])
            if g.empty:
                continue
            for t, lab in TASKS:
                gt = g[g.index.get_level_values('task') == t]
                if gt.empty:
                    continue
                b = base.loc[gt.index]
                rows.append([MODE_LABEL[mode], lab, f(gt.power_W.median(), 1), f(gt.E_query_J.median(), 1),
                             f((gt.E_query_J / b.E_query_J).median(), 2), f((gt.latency_s / b.latency_s).median(), 2),
                             f((gt.JPCA / b.JPCA).median(), 2), pct(gt.acc.mean())])
        write('T4_modes.md', md(rows, ['Mode', 'Task', 'Power (W)', 'E/query (J)', 'E rel. MAXN', 'Latency rel. MAXN',
                                       'JPCA rel. MAXN', 'Acc. (%)']))

    if (TAB_IN / 'T5_thinking.csv').exists():
        s = pd.read_csv(TAB_IN / 'T5_thinking.csv')
        rows = []
        for m in [x for x in order if x.startswith('qwen3')]:
            for t, lab in TASKS:
                for th in ['off', 'on']:
                    x = s[(s.model == m) & (s.task == t) & (s.think.astype(str) == th)]
                    if len(x):
                        lbl = ('Qwen3-4B Q4_K_M ' + ('Instruct-2507' if th == 'off' else 'Thinking-2507')) if m == 'qwen3:4b' else meta.loc[m, 'label']
                        rows.append([lbl, lab, th, pct(x.acc.iloc[0]), f(x.out_tokens.iloc[0], 0),
                                     f(x.latency_s.iloc[0], 1), f(x.E_query_J.iloc[0], 1), f(x.JPCA.iloc[0], 1),
                                     pct(x.truncated.iloc[0])])
        write('T5_thinking.md', md(rows, ['Model', 'Task', 'Think', 'Acc. (%)', 'Out. tokens', 'Lat. (s)',
                                          'E (J)', 'JPCA (J)', 'Trunc. (%)']))

    if (TAB_IN / 'T6_cascade_frontier_test.csv').exists():
        s = pd.read_csv(TAB_IN / 'T6_cascade_frontier_test.csv')
        cols = list(s.columns)
        rows = [[f(v, 2) if isinstance(v, float) else v for v in r] for r in s.itertuples(index=False)]
        write('T6_cascade.md', md(rows, cols))

    # paired tests (pre-specified comparisons)
    if (TAB_IN / 'T7_pairwise_tests.csv').exists():
        s = pd.read_csv(TAB_IN / 'T7_pairwise_tests.csv')
        rows = [[r.A, r.B, dict(TASKS)[r.task], f(r.diff_pp, 1), f(r.p_mcnemar, 3) if r.p_mcnemar >= 0.001 else '<0.001',
                 f(r.p_holm, 3) if r.p_holm >= 0.001 else '<0.001', f'{f(r.JPCA_ratio, 2)} [{f(r.JPCA_ratio_lo, 2)}, {f(r.JPCA_ratio_hi, 2)}]']
                for r in s.itertuples()]
        write('T7_tests.md', md(rows, ['A', 'B', 'Task', 'Acc. A − B (pp)', 'McNemar p', 'Holm p', 'JPCA A / B [95% CI]']))

    # cascade summary from summary.json
    sj = ROOT / 'results' / 'summary.json'
    if sj.exists():
        import json
        c = json.load(open(sj)).get('cascade', {})
        lab = lambda m: meta.loc[m, 'label'] if m in meta.index else m
        rows = []
        for t, l in TASKS:
            if t not in c:
                continue
            x = c[t]
            rows.append([l, f"{lab(x['s1'])} + {lab(x['s2'])} → {lab(x['big'])}", pct(x['escalation_rate']), pct(x['test_acc']),
                         f(x['test_E'], 1), lab(x['ref_model']), pct(x['ref_acc']), f(x['ref_E'], 1), pct(x['oracle_acc']), f(x['oracle_E'], 1)])
        write('T9_cascade.md', md(rows, ['Task', 'Cascade (s1 + s2 → L)', 'Esc. (%)', 'Casc. acc. (%)', 'Casc. E (J)',
                                        'Best single', 'Acc. (%)', 'E (J)', 'Oracle acc. (%)', 'Oracle E (J)']))

    # regression coefficients from the statsmodels summary text
    rt = TAB_IN / 'regression_logE.txt'
    if rt.exists():
        import re
        names = {'Intercept': 'Intercept', 'C(family)[T.Llama3.2]': 'Family: Llama 3.2', 'C(family)[T.Qwen2.5]': 'Family: Qwen2.5',
                 'log_size': 'log weight-file size (GB)', 'log_out': 'log output tokens', 'log_in': 'log prompt tokens'}
        rows = []
        for line in rt.read_text().splitlines():
            m = re.match(r'^(\S+)\s+(-?\d+\.\d+)\s+(\d+\.\d+)\s+(-?\d+\.\d+)\s+(\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$', line)
            if m and m.group(1) in names:
                coef, se, z, p = float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5))
                mult = f'{100 * (2 ** coef - 1):+.0f}% per doubling' if m.group(1).startswith('log') else (
                    f'{100 * (2.718281828 ** coef - 1):+.0f}% vs Gemma 2' if 'family' in m.group(1) else '–')
                rows.append([names[m.group(1)], f(coef, 3), f(se, 3), f(z, 1), '<0.001' if p < 0.001 else f(p, 3), mult])
        write('T8_regression.md', md(rows, ['Term', 'Coefficient', 'Robust SE', 'z', 'p', 'Effect on energy']))


if __name__ == '__main__':
    main()
