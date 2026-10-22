# Joules per Correct Answer

Energy, accuracy, and latency trade-offs of quantized small language models across power modes on an NVIDIA Jetson AGX Orin 32 GB.

This repository holds the measurement harness, the raw per-query logs with model outputs, the full-resolution power traces, the analysis scripts, and the manuscript sources of the study.

## Layout

| Folder | Content |
|---|---|
| `code/` | `bench.py` (energy-instrumented benchmark runner), `power_monitor.py` (50 Hz INA3221 sampler), `tasks.py` (prompts, parsing, fixed-seed subsampling), `analyze.py` (statistics, tables, figures), campaign scripts |
| `data/raw/` | The evaluated splits of ARC-Challenge (test), BoolQ (validation), and GSM8K (test) as Parquet files |
| `results/runs/<tag>/<mode>/` | One JSON line per query (`*__<task>.jsonl`), a per-configuration `*__meta.json`, and compressed power traces (`*__trace_*.npz`) for `E1_main` (main grid), `E2_modes` (power modes), and `E3_think` (thinking on/off); `contaminated/` holds runs quarantined by the idle-machine guard |
| `results/tables/`, `results/summary.json` | Outputs of `analyze.py` |
| `figures/` | Figures of the paper (PNG and PDF) |
| `paper/` | Manuscript sections (Markdown), bibliography, table generator, and the Word build pipeline |
| `logs/` | Campaign and model-pull logs |

## Reproducing the campaign

```bash
bash code/start_server.sh            # dedicated Ollama 0.33 instance on port 11500
bash code/resume_campaign.sh         # pulls the models, then runs E1, E2 (MAXN, 40 W), E3; resumable
python code/analyze.py               # results/tables, results/summary.json, figures/
python paper/make_tables.py && python paper/build_paper.py --version V01
```

The 30 W and 15 W modes change the GPU power-gating mask and require a reboot on JetPack 6. They are not part of the study; `code/run_mode_after_reboot.sh` documents how they could be run.

## Environment

Jetson AGX Orin 32 GB Developer Kit, JetPack 6 (L4T R36.4.7), CUDA 12.6, Ollama 0.33.0, Python 3.11 with numpy, pandas, scipy, statsmodels, matplotlib, adjustText, python-docx.

## Record format

Each line of a run file contains the question id, the task, the model and power mode, the parsed prediction and gold answer, correctness, wall-clock timestamps, total, prefill, and decode energy in joules, mean power, prompt and output token counts, Ollama phase durations, the stop reason, peak junction temperature, mean GPU clock, and the full model response.

## License

Code: MIT. Data and figures: CC BY 4.0. The benchmark datasets keep their original licenses.
