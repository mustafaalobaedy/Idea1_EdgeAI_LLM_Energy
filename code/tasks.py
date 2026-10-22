"""Benchmark tasks: prompt construction, answer extraction and fixed-seed subsampling."""
import os
import re

import pandas as pd

RAW = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
SEED = 2026


def _arc_prompt(r):
    opts = '\n'.join(f'{l}. {t}' for l, t in zip(r['choices']['label'], r['choices']['text']))
    return ('Answer the following multiple-choice science question. '
            'Respond with only the letter of the correct option.\n\n'
            f"Question: {r['question']}\n{opts}\nAnswer:")


def _boolq_prompt(r):
    q = r['question'].strip()
    q = q[0].upper() + q[1:] + ('' if q.endswith('?') else '?')
    return ('Read the passage and answer the question with only "yes" or "no".\n\n'
            f"Passage: {r['passage']}\n\nQuestion: {q}\nAnswer:")


def _gsm_prompt(r):
    return ('Solve the following math problem step by step. '
            'Put the final numerical answer inside \\boxed{}.\n\n'
            f"Problem: {r['question']}")


def parse_arc(text, labels):
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()
    m = re.search(r'\b(?:answer|option)\s*(?:is|:)?\s*\(?([A-E1-5])\b', text, re.I)
    if not m:
        m = re.search(r'^\W*\(?([A-E1-5])\b', text)
    if not m:
        m = re.search(r'\b([A-E])\b', text)
    pred = m.group(1).upper() if m else None
    return pred if pred in labels else None


def parse_boolq(text):
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip().lower()
    m = re.search(r'\b(yes|no|true|false)\b', text)
    if not m:
        return None
    return m.group(1) in ('yes', 'true')


def _num(s):
    s = s.replace(',', '').replace('$', '').rstrip('.')
    try:
        v = float(s)
        return int(v) if v == int(v) else v
    except ValueError:
        return None


def parse_gsm(text):
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S)
    m = re.findall(r'\\boxed\{\s*\\?\$?\s*(-?[\d,]*\.?\d+)', text)
    if not m:
        m = re.findall(r'answer is[:\s]*\$?\s*(-?[\d,]*\.?\d+)', text, re.I)
    if not m:
        m = re.findall(r'-?[\d,]*\.?\d+', text)
    return _num(m[-1]) if m else None


def load_task(name, n):
    """Return list of dicts: qid, prompt, gold, labels(optional). Deterministic subsample of size n."""
    if name == 'arc':
        d = pd.read_parquet(os.path.join(RAW, 'arc_challenge_test.parquet'))
        d = d.sample(n=min(n, len(d)), random_state=SEED)
        return [dict(qid=r['id'], prompt=_arc_prompt(r), gold=r['answerKey'],
                     labels=list(r['choices']['label'])) for _, r in d.iterrows()]
    if name == 'boolq':
        d = pd.read_parquet(os.path.join(RAW, 'boolq_validation.parquet')).reset_index()
        d = d.sample(n=min(n, len(d)), random_state=SEED)
        return [dict(qid=f"boolq_{r['index']}", prompt=_boolq_prompt(r), gold=bool(r['answer']))
                for _, r in d.iterrows()]
    if name == 'gsm8k':
        d = pd.read_parquet(os.path.join(RAW, 'gsm8k_test.parquet')).reset_index()
        d = d.sample(n=min(n, len(d)), random_state=SEED)
        return [dict(qid=f"gsm_{r['index']}", prompt=_gsm_prompt(r),
                     gold=_num(r['answer'].split('####')[-1].strip())) for _, r in d.iterrows()]
    raise ValueError(name)


def score(task, text, item):
    if task == 'arc':
        pred = parse_arc(text, item['labels'])
    elif task == 'boolq':
        pred = parse_boolq(text)
    else:
        pred = parse_gsm(text)
        if pred is not None and item['gold'] is not None:
            return pred, abs(float(pred) - float(item['gold'])) < 1e-6
    return pred, pred == item['gold']


MAX_TOKENS = {'arc': 8, 'boolq': 4, 'gsm8k': 512}
