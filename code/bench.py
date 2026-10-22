"""Energy-instrumented LLM benchmark runner for Jetson AGX Orin.

Usage:
  python bench.py --mode MAXN --models qwen2.5:3b-instruct-q4_K_M ... --tasks arc boolq gsm8k \
                  --n arc=300 boolq=300 gsm8k=200 [--think on|off] [--tag main]

Writes one JSONL line per question to results/runs/<tag>/<mode>/<model>__<task>[__think].jsonl,
a compressed power trace per (model, task) and a meta JSON with idle power. Resumable.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np
import requests

sys.path.insert(0, os.path.dirname(__file__))
from power_monitor import COLUMNS, PowerMonitor, integrate  # noqa: E402
from tasks import MAX_TOKENS, load_task, score  # noqa: E402

HOST = os.environ.get('BENCH_OLLAMA', 'http://127.0.0.1:11500')
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
SYS_OLLAMA = 'http://127.0.0.1:11434'
WARMUP = 3
IDLE_S = 20


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def safe(m):
    return m.replace(':', '_').replace('/', '_')


def loadavg():
    return float(open('/proc/loadavg').read().split()[0])


def system_ollama_busy():
    try:
        return [m['name'] for m in requests.get(SYS_OLLAMA + '/api/ps', timeout=5).json().get('models', [])]
    except Exception:
        return []


def heavy_procs(thresh=50.0):
    """Foreign processes above `thresh` %CPU over a 5 s window (excluding ollama and this runner)."""
    out = subprocess.run(['top', '-b', '-n', '2', '-d', '5', '-w', '200'], capture_output=True, text=True).stdout
    block = out.split('PID USER')[-1].splitlines()[1:]
    me = str(os.getpid())
    hv = []
    for line in block:
        f = line.split()
        if len(f) < 12:
            continue
        try:
            cpu = float(f[8].replace(',', '.'))
        except ValueError:
            continue
        if cpu > thresh and f[0] != me and f[11] not in ('ollama', 'top'):
            hv.append(f'{f[11]}({f[0]}):{cpu:.0f}%')
    return hv


def wait_idle(max_wait_s=12 * 3600):
    """Block until no foreign CPU-heavy process runs and the shared Ollama has no model loaded."""
    if os.environ.get('BENCH_SKIP_IDLE'):
        return loadavg()
    t0 = time.time()
    while True:
        hv, busy = heavy_procs(), system_ollama_busy()
        if not hv and not busy:
            return loadavg()
        if time.time() - t0 > max_wait_s:
            log(f'WARNING: proceeding despite heavy={hv} sys_ollama={busy}')
            return loadavg()
        log(f'waiting for idle machine: heavy={hv} sys_ollama={busy}')
        time.sleep(120)


def chat(model, prompt, max_tokens, think, num_ctx):
    body = dict(model=model, stream=False, keep_alive='10m',
                messages=[{'role': 'user', 'content': prompt}],
                options=dict(temperature=0, top_k=1, seed=42, num_predict=max_tokens, num_ctx=num_ctx))
    if think is not None:
        body['think'] = think
    r = requests.post(HOST + '/api/chat', json=body, timeout=3600)
    r.raise_for_status()
    return r.json()


def unload(model):
    try:
        requests.post(HOST + '/api/generate', json=dict(model=model, keep_alive=0), timeout=120)
    except Exception:
        pass


def run_model(args, model, mon, pidx, think, tasks, outdir, suffix):
    """Measure one model on all pending tasks. Returns (ran, contaminated_by)."""
    todo = {}
    foreign = []
    last_check = time.time()
    for t in args.tasks:
        fn = os.path.join(outdir, f'{safe(model)}__{t}{suffix}.jsonl')
        done = set()
        if os.path.exists(fn):
            done = {json.loads(l)['qid'] for l in open(fn)}
        rest = [it for it in tasks[t] if it['qid'] not in done]
        if rest:
            todo[t] = (fn, rest)
    if not todo:
        log(f'skip {model}: complete')
        return False, []

    la = wait_idle()
    log(f'== {model} mode={args.mode} think={args.think} loadavg={la:.2f}')
    unload_others = requests.get(HOST + '/api/ps', timeout=10).json().get('models', [])
    for m in unload_others:
        unload(m['name'])
    # load + warm-up (excluded from measurements)
    t_load0 = time.time()
    first_item = next(iter(todo.values()))[1][0]
    r = chat(model, first_item['prompt'], 4, think, args.num_ctx)
    load_s = time.time() - t_load0
    for i in range(WARMUP):
        chat(model, f'Say the number {i}.', 8, think, args.num_ctx)
    # idle power with the model resident
    mon.drain()
    time.sleep(IDLE_S)
    tr = mon.drain()
    p_idle = float(np.median(tr[:, pidx]))
    meta = dict(model=model, mode=args.mode, think=args.think, load_s=load_s, p_idle_loaded_W=p_idle,
                idle_trace_temp_tj=float(np.nanmean(tr[:, COLUMNS.index('T_tj')])),
                loadavg_before=la, num_ctx=args.num_ctx, time=time.time(),
                nvpmodel=subprocess.run(['nvpmodel', '-q'], capture_output=True, text=True).stdout.split('\n')[0])
    with open(os.path.join(outdir, f'{safe(model)}{suffix}__meta.json'), 'a') as f:
        f.write(json.dumps(meta) + '\n')
    log(f'   load={load_s:.1f}s idle={p_idle:.2f}W')

    for t, (fn, items) in todo.items():
        traces = []
        mon.drain()
        with open(fn, 'a') as fo:
            for k, it in enumerate(items):
                t0 = time.time()
                res = chat(model, it['prompt'], args.think_tokens if think else MAX_TOKENS[t], think, args.num_ctx)
                t1 = time.time()
                tr = mon.snapshot()
                e, pw, ns = integrate(tr, t0, t1, pidx)
                pre_s = (res.get('load_duration', 0) + res.get('prompt_eval_duration', 0)) / 1e9
                e_pre, _, _ = integrate(tr, t0, min(t1, t0 + pre_s), pidx)
                msg = res['message']
                text = msg.get('content', '')
                pred, ok = score(t, text, it)
                rec = dict(qid=it['qid'], task=t, model=model, mode=args.mode, think=args.think,
                           correct=bool(ok), pred=None if pred is None else str(pred), gold=str(it['gold']),
                           t0=t0, t1=t1, wall_s=t1 - t0, energy_J=e, energy_prefill_J=e_pre,
                           energy_decode_J=e - e_pre, mean_power_W=pw, p_idle_W=p_idle, n_power_samples=ns,
                           prompt_tokens=res.get('prompt_eval_count'), output_tokens=res.get('eval_count'),
                           prompt_eval_s=res.get('prompt_eval_duration', 0) / 1e9,
                           eval_s=res.get('eval_duration', 0) / 1e9, load_s=res.get('load_duration', 0) / 1e9,
                           done_reason=res.get('done_reason'),
                           T_tj_max=float(np.nanmax(tr[tr[:, 0] >= t0][:, COLUMNS.index('T_tj')])) if ns else None,
                           gpu_freq_mean=float(np.mean(tr[tr[:, 0] >= t0][:, COLUMNS.index('gpu_freq_mhz')])) if ns else None,
                           response=text, thinking=msg.get('thinking'))
                fo.write(json.dumps(rec) + '\n')
                fo.flush()
                if k % 50 == 0:
                    log(f'   {t} {k}/{len(items)} E={e:.1f}J P={pw:.1f}W out={rec["output_tokens"]} ok={ok}')
                if len(mon._buf) > 200000:
                    traces.append(mon.drain())
                if not os.environ.get('BENCH_SKIP_IDLE') and time.time() - last_check > 180:
                    hv = heavy_procs()  # between queries, outside any measured window
                    last_check = time.time()
                    if hv:
                        foreign = hv
                        log(f'   foreign load detected: {hv}')
                        break
        if foreign:
            break
        traces.append(mon.drain())
        np.savez_compressed(fn.replace('.jsonl', f'__trace_{int(time.time())}.npz'),
                            trace=np.concatenate([x for x in traces if len(x)]), columns=np.array(COLUMNS))
    unload(model)
    return True, foreign


def run(args):
    mon = PowerMonitor(0.02).start()
    pidx = COLUMNS.index('P_total')
    think = None if args.think is None else (args.think == 'on')
    nmap = dict(kv.split('=') for kv in args.n)
    outdir = os.path.join(os.environ.get('BENCH_RESULTS', os.path.join(ROOT, 'results')), 'runs', args.tag, args.mode)
    os.makedirs(outdir, exist_ok=True)
    tasks = {t: load_task(t, int(nmap[t])) for t in args.tasks}
    suffix = '' if think is None else f'__think{args.think}'

    for model in args.models:
        for attempt in range(4):
            ran, foreign = run_model(args, model, mon, pidx, think, tasks, outdir, suffix)
            if not foreign:
                break
            # foreign load during the run: quarantine this model's files for this setting and repeat
            qdir = os.path.join(outdir, 'contaminated')
            os.makedirs(qdir, exist_ok=True)
            for f in glob.glob(os.path.join(outdir, f'{safe(model)}__*{suffix}*')):
                if suffix or '__think' not in f:
                    os.rename(f, os.path.join(qdir, f'{int(time.time())}_' + os.path.basename(f)))
            log(f'   CONTAMINATED by {foreign}: quarantined, retrying (attempt {attempt + 1})')
    mon.stop()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True)
    ap.add_argument('--models', nargs='+', required=True)
    ap.add_argument('--tasks', nargs='+', default=['arc', 'boolq', 'gsm8k'])
    ap.add_argument('--n', nargs='+', default=['arc=300', 'boolq=300', 'gsm8k=200'])
    ap.add_argument('--think', choices=['on', 'off'], default=None)
    ap.add_argument('--think_tokens', type=int, default=4096)
    ap.add_argument('--num_ctx', type=int, default=2048)
    ap.add_argument('--tag', default='main')
    ap.add_argument('--max_load', type=float, default=2.0)
    run(ap.parse_args())
