# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A journal paper plus the benchmark that produced it: 27 small-LLM configurations (Qwen2.5, Llama 3.2, Gemma 2, Qwen3; GGUF via Ollama) measured on this NVIDIA Jetson AGX Orin 32 GB for accuracy, latency, and on-module energy on ARC-Challenge, BoolQ, and GSM8K. The main metric is joules per correct answer (JPCA). Target venue: PeerJ Computer Science, single author. The campaign is complete (23,960 queries); the remaining work is manuscript revision.

The folder lives in Dropbox and is also a git repository pushed to `https://github.com/mustafaalobaedy/Idea1_EdgeAI_LLM_Energy` (private). Commit author must be "Mustafa Muwafak Theab" only; the repo-local git config is already set. Do not add other contributors or co-author trailers to commits.

## Commands

`PY=/mnt/data/miniforge3/envs/edgeai/bin/python`, run from the project root. Dropbox strips execute bits, so run shell scripts with `bash`.

```bash
bash code/start_server.sh                      # dedicated Ollama 0.33 on :11500, model store /mnt/data/private_ai/ollama
bash code/resume_campaign.sh                   # start server if needed, re-pull missing models, then run_campaign.sh (resumable)
$PY code/bench.py --mode MAXN --tag E1_main --models qwen2.5:3b-instruct-q4_K_M --n arc=300 boolq=300 gsm8k=200
$PY code/analyze.py                            # results/tables/*.csv, results/summary.json, figures/*  (~1.5 min)
$PY paper/make_tables.py                       # results/tables CSV + summary.json -> paper/tables/*.md
$PY paper/build_paper.py --version V07         # paper/paper.docx + "Joules per Correct Answer Energy V07.docx" + PDF (reading copy, figures embedded)
$PY paper/build_paper.py --version V07 --submission   # same without embedded figures/tables + paper/submission/ (Figure*.png, Table*.docx, legends)
```

Smoke test of the harness: `BENCH_RESULTS=<scratch> BENCH_SKIP_IDLE=1 $PY code/bench.py ... --n arc=2`.

Never run `analyze.py` (or anything CPU-heavy) while `bench.py` is measuring: the idle-machine guard quarantines the running configuration if a foreign process exceeds 50% of a core.

## Pipeline

```
data/raw/*.parquet --tasks.py (seed 2026 subsample)--> items
items + Ollama:11500 --bench.py + power_monitor.py--> results/runs/<tag>/<mode>/<model>__<task>[__think{on,off}].jsonl
                                                      + <model>__meta.json + <model>__<task>__trace_*.npz
results/runs --analyze.py--> results/summary.json, results/tables/*.csv, figures/*.png|pdf
summary.json + tables CSV --make_tables.py--> paper/tables/*.md
paper/sections/*.md + paper/tables + figures + references.bib --build_paper.py--> paper/*.docx (+ submission/)
```

- **Runs resume by qid**: `bench.py` skips items already in a run file and models whose files are complete. `run_campaign.sh` is the full experiment; `resume_campaign.sh` wraps it. A quarantined configuration (`contaminated/`) is retried up to four times, then skipped; relaunching `resume_campaign.sh` later picks it up.
- **Tags**: `E1_main` (23 configs, 300/300/200 items), `E2_modes` (8 configs at MAXN and MODE_40W, 100/100/60 items), `E3_think` (Qwen3 1.7B/4B/8B, thinking on/off, 100 ARC + 100 GSM8K). `analyze.py` reads only `runs/<tag>/<mode>/*.jsonl`; subfolders like `contaminated/` and `partial_stopped/` are ignored.
- **`summary.json` is the only source of numbers for the paper text.** `make_tables.py` renders the tables from it and the CSVs; figures come from `analyze.py`. Do not type numbers into the draft by hand.
- **Power modes**: MODE_30W and MODE_15W change the GPU TPC mask and need a reboot on JetPack 6; they are out of scope by the author's decision (`code/run_mode_after_reboot.sh` documents the procedure but is not part of the study).
- **Qwen3-4B quirk**: the Ollama tag `qwen3:4b` is Qwen3-4B-Thinking-2507 and ignores `think=off`. `analyze.py` drops that run, reports it as `thinking_4b_switch_ineffective`, and substitutes `qwen3:4b-instruct-2507-q4_K_M` (think off) as the 4B non-thinking arm under the alias `qwen3:4b`.

## Paper build (`paper/build_paper.py`)

Template: `PeerJ-research-manuscript-template.docx` (also at the research root; `PAPER_TEMPLATE=<path>` overrides). The build discards the template body and appends `paper/sections/*.md` in file-name order (files containing "outline" are skipped). Draft syntax is documented in the script docstring: `TITLE:`, `AUTHOR: name | affiliation | address | email`, multi-line `ABSTRACT:` (a leading `Word.` becomes a bold structured-abstract heading), `# / ## / ### / ####` headings, `- BULLET:`, `HIGHLIGHT:` (yellow, used for the AI statement), `{{FIG:path|caption|label}}`, `{{DIAGRAM:name|caption|label}}` (editable shapes from `diagrams.py`), `{{TABLE:path.md|caption|label}}`, `{{EQ:name}}` (OMML from `equations.py`), `{{ALGO:caption|label}}…{{/ALGO}}`, cross-references `[fig:x] [tab:x] [eq:x] [alg:x]`, citations `[@key; @key2]` rendered PeerJ author-year (`Lastname et al. [@key]` becomes narrative).

The build fails on: unknown citation keys, labels defined but never cited or cited but undefined, duplicate labels, and figures/tables whose first citation is not in ascending numerical order (PeerJ rule). It warns on `[TO WRITE]` placeholders and sentences over 30 words.

## Author's rules (binding)

From `../00 Instructions/Instructions.txt`: passive voice, good American English, short sentences; every reference real and cited, every citation listed; every table/figure/equation/algorithm cited; equations as Word equation objects; diagrams editable and black-and-white; pseudocode and full EDA explained; no-funding statement, highlighted AI statement; code link lists only the author. Output is Word, named from the first five title words plus a version (`Joules per Correct Answer Energy Vxx.docx`); **never overwrite an earlier version**, bump the number. V06 is the latest at the time of writing. A cover letter (`paper/Cover Letter Vxx.docx`) accompanies the manuscript.

PeerJ template guidance (in its yellow text boxes): no figures or tables embedded in the submitted text (use `--submission`), structured abstract ≤ 500 words, title ≤ 250 characters, US Letter, line numbers on, left-aligned, full statistical reporting (test, statistic or discordant counts, exact p-value, effect size with CI), distributions rather than bare bars where possible.

## Things that matter

- The Jetson is shared with other projects (other Claude sessions, training jobs). Check `ps` before any energy run; the guard will otherwise pause or quarantine.
- The system Ollama on :11434 uses a different model store; always pull benchmark models with `OLLAMA_HOST=127.0.0.1:11500`.
- Submission figures must be at least 900 × 900 px (PeerJ); figures are saved at 600 dpi for that reason. Do not lower `savefig.dpi`.
- `logs/ollama_bench_server*.log` are gigabytes and git-ignored; `logs/campaign.log` is the run history.
- Sentences in the draft are in passive voice on purpose; keep that voice when editing.
