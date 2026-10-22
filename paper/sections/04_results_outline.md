# 4. Results (outline — fill in after the campaign completes)

Each subsection maps to one research question from Section 1 and to the outputs of `code/analyze.py` (tables in `results/tables/`, figures in `figures/`, headline numbers in `results/summary.json`). Numbers below are to be taken from those files; do not type them by hand.

## 4.0 Campaign overview and data quality
- Totals: queries, kJ, hours (`summary.json: n_queries_total, total_energy_kJ, total_hours`).
- Idle power per power mode (Table from `idle_baseline.jsonl`; Fig. 3 `fig03_idle_power_by_mode`).
- Example power trace of one GSM8K query showing prefill and decode phases (Fig. 2 `fig02_power_trace_example`).
- Token-length EDA per task and model (Fig. 4 `fig04_token_distributions`) and per-query energy ECDFs (Fig. 5 `fig05_energy_ecdf`).
- Parse-failure and truncation rates (T3 columns `parse_fail`, `truncated`). Note from the partial data: ARC outputs hit the 8-token cap in 50–73% of items for Qwen2.5 Q4_K_M, yet parse failure stays ≤3%, because the letter is emitted within the first tokens. State this explicitly.
- Contaminated and re-run configurations (list from `results/runs/*/*/contaminated/`).

## 4.1 RQ1 — Size and precision (main grid, MAXN)
- Table 4 = `T3_main_results.csv` (accuracy with CI, E/query, JPCA with CI, latency, TTFT, throughput, J/token).
- Fig. 6 `fig06_pareto_accuracy_energy`: Pareto fronts per task (`summary.json: pareto`).
- Paired McNemar tests with Holm correction (`T7_pairwise_tests.csv`, `summary.json: tests`).
- Log-energy regression (`regression_logE.txt`, `summary.json: regression`): coefficients for log size, log output tokens, log prompt tokens.
- Energy-per-token versus weight-file size and implied effective bandwidth (Fig. 8 `fig08_energy_per_token_vs_size`, `summary.json: energy_law`).
- Thermal: sustained T_j and GPU clock over long runs (Fig. 12 `fig12_thermal_sustained`, `summary.json: thermal`).

## 4.2 RQ2 — Power modes (E2_modes)
- Table 5 = `T4_power_modes.csv`, relative-to-MAXN view `T4_power_modes_relative.csv`, summary `T4b_power_modes_summary.csv`.
- Fig. 9 `fig09_power_modes`: energy, latency and JPCA per mode and task.
- Output consistency across modes under greedy decoding (`summary.json: mode_consistency`) — justifies reusing MAXN accuracy.
- Test–retest repeatability of MAXN energies between E1 and E2 on overlapping items (`summary.json: repeatability`).
- Answer to the race-to-idle question: does 15 W save joules per correct answer or not.

## 4.3 RQ3 — Quantization ladder (Qwen2.5-3B, Q3_K_M…FP16)
- Fig. 7 `fig07_quantization_ladder`: accuracy, energy and JPCA versus effective bits per weight.
- Where accuracy saturates and where energy keeps rising; JPCA-optimal level per task.

## 4.4 RQ4 — Thinking on/off (E3_think, Qwen3 1.7B/4B/8B)
- Table 6 = `T5_thinking.csv`; Fig. 10 `fig10_thinking_tradeoff` (`summary.json: thinking`).
- Marginal joules per additional correct answer; token-length blow-up on ARC versus GSM8K; truncation at the 4,096-token budget.

## 4.5 RQ5 — Agreement cascade
- Table 7 = `T6_cascade_frontier_test.csv`; Fig. 11 `fig11_cascade` (`summary.json: cascade`).
- Selected triple on the selection half, performance on the held-out half, oracle-router bound.

## Checklist before writing prose
- [ ] Every figure and table number above is referenced in the text.
- [ ] Figure labels do not overlap; fonts are dark (user instruction).
- [ ] All numbers copied from `results/tables/*.csv` or `summary.json`.
