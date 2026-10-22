# Materials & Methods

## Hardware platform and power modes

All experiments were run on a single NVIDIA Jetson AGX Orin 32 GB Developer Kit [@karumbunathan2022orin]. The module carries an 8-core Arm Cortex-A78AE CPU and an Ampere-architecture integrated GPU with 7 texture-processing clusters (TPCs). This corresponds to 14 streaming multiprocessors and 1,792 CUDA cores. Its 32 GB of LPDDR5 memory is shared by the CPU and the GPU. The software stack was Ubuntu 22.04.5 LTS with NVIDIA Jetson Linux R36.4.7 (JetPack 6), CUDA runtime 12.6, and Ollama 0.33.0 [@ollama2023ollama], by which the llama.cpp inference engine is wrapped [@gerganov2023llamacpp]. Dynamic voltage and frequency scaling (DVFS) was left at its default governor. The clocks were deliberately not locked with `jetson_clocks`, so that the measurements reflect how a deployed device behaves.

Pre-defined power budgets are exposed by Jetson modules through the `nvpmodel` tool. The four budgets that are shipped for the AGX Orin 32 GB are listed in [tab:pmodes]. Two of them, MAXN and MODE_40W, share the same GPU power-gating mask and can be switched at run time. The 30 W and 15 W modes change the GPU TPC mask and require a reboot on JetPack 6. The device is shared by several research projects, so reboots could not be scheduled inside the measurement campaign. The power-mode sweep therefore covers MAXN and MODE_40W, and the two reboot-only modes are left for future work (Discussion section). The budgets differ in the number of online CPU cores and in the number of power-gated GPU TPCs. They also differ in the frequency caps of the CPU, the GPU, and the external memory controller (EMC). The values were read directly from the `nvpmodel` configuration of the device.

{{TABLE:paper/tables/T1_power_modes.md|Stock power modes of the AGX Orin 32 GB (values read from the device's nvpmodel configuration). MAXN and MODE_40W were evaluated; MODE_30W and MODE_15W require a reboot|pmodes}}

## Power, thermal, and utilization measurement

The measurement setup is shown in [fig:setup]. Two Texas Instruments INA3221 current and voltage monitors are carried by the AGX Orin. Four on-module supply rails are reported by them through the Linux `hwmon` interface: VDD_GPU_SOC, VDD_CPU_CV, VIN_SYS_5V0, and VDDQ_VDD2_1V8AO [@nvidia2024jetsonpower]. The bus voltage and current of each rail were read by a dedicated sampling thread at 50 Hz. The module power was computed as the sum of the rail powers, as given in [eq:pmodule].

{{EQ:pmodule}}

In [eq:pmodule], *V_r*(*t*) and *I_r*(*t*) are the voltage and current of rail *r* at time *t*. The CPU, GPU, SoC, and junction (*T_j*) temperatures, the GPU utilization, and the GPU clock frequency were logged by the same thread. The INA3221 averages 512 internal conversions per reported value, so the effective refresh period is about 70 ms. The sensor is therefore oversampled at a 20 ms period, and no aliasing is introduced. The module is measured by these rails, not the carrier board, the fan, or the peripherals, so the reported energies are *module-level* energies. This is the standard scope in Jetson studies, and board-specific overheads are excluded by it. Internal Jetson power sensors can be systematically biased against external reference meters, and calibration is beneficial [@shalavi2023accurate]. No calibrated external meter was available, so the absolute joule values should be read with this caveat. The central conclusions, however, are ratios and rankings between configurations that were measured with the same sensors on the same device. Examples are relative JPCA, relative energy across power modes, and Pareto membership. A consistent multiplicative sensor bias is canceled in such comparisons.

{{DIAGRAM:setup|Measurement setup. The benchmark harness sends one request at a time to a dedicated Ollama server and reads the four INA3221 power rails at 50 Hz in a background thread. Energy is integrated between the request timestamps t0 and t1|setup}}

For each query *q*, wall-clock timestamps *t*0 and *t*1 were recorded immediately before the HTTP request was sent and after the full response was received. The energy was obtained by trapezoidal integration of the power trace over the interval [*t*0, *t*1], with linear interpolation at the boundaries, as given in [eq:energy].

{{EQ:energy}}

The duration of the prompt-processing (prefill) phase and of the token-generation (decode) phase is reported by Ollama for every request. The query energy was split into prefill and decode energy by integration over the corresponding sub-intervals, as given in [eq:split].

{{EQ:split}}

The *dynamic* energy, in which the idle power of the device is removed, is also reported, as given in [eq:dyn].

{{EQ:dyn}}

In [eq:dyn], the idle power is the median module power during a 20 s window with the model resident in memory but no request in flight. A 60 s idle baseline with no model loaded was also recorded for each evaluated power mode.

## Models and quantization

Twenty-seven model configurations from four open-weight families were evaluated. All are instruction-tuned and distributed in the GGUF format ([tab:models]):

- BULLET: **Qwen2.5-Instruct** [@qwen2024qwen25] at 0.5B, 1.5B, 3B, and 7B parameters;
- BULLET: **Llama 3.2-Instruct** [@grattafiori2024llama3; @meta2024llama32] at 1B and 3B;
- BULLET: **Gemma 2-IT** [@gemma2024gemma2] at 2B;
- BULLET: **Qwen3** [@yang2025qwen3] at 1.7B and 8B, hybrid models that can be run with or without an explicit reasoning ("thinking") phase. At 4B, the Thinking-2507 and Instruct-2507 variants were used, which reason always and never, respectively.

Quantization was varied with the k-quant and round-to-nearest formats of llama.cpp. All sub-7B models were tested at Q4_K_M, Q8_0, and FP16. The Qwen2.5-7B model was tested at Q4_K_M and Q8_0, because the shared CPU–GPU memory would have been contended by FP16 (15.2 GB). For Qwen2.5-3B, a full *quantization ladder* (Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0, FP16) was run, so that the effect of numerical precision was isolated while architecture and training were held fixed. The effective bits per weight were computed from the actual weight-file size and the parameter count reported in the GGUF metadata.

{{TABLE:paper/tables/T2_models.md|Evaluated model configurations (Ollama tags; size is the GGUF weight-file size; bits per weight are computed from file size and parameter count)|models}}

## Tasks and evaluation protocol

Three tasks were chosen to cover three different inference regimes:

- BULLET: **ARC-Challenge** [@clark2018think]: grade-school science multiple-choice questions. The prompts are short and the outputs are a single letter, so the work is dominated by prefill and fixed overheads.
- BULLET: **BoolQ** [@clark2019boolq]: yes/no questions about a Wikipedia passage. The prompts are long (about 150–400 tokens) and the outputs are a single token, so prefill dominates.
- BULLET: **GSM8K** [@cobbe2021training]: multi-step arithmetic word problems answered with zero-shot chain-of-thought [@wei2022chain]. The prompts are short and the outputs are long (about 150–500 tokens), so decode dominates.

A fixed random subsample was drawn with seed 2026 from each test split (ARC, GSM8K) or validation split (BoolQ). The main grid used 300 ARC, 300 BoolQ, and 200 GSM8K items. The power-mode sweep used 100, 100, and 60 items on eight configurations. Identical items were seen by all configurations. The prompts were zero-shot and identical across models. Each prompt was sent as a single user turn through the native chat template of the model. In the ARC prompts, "only the letter of the correct option" was requested. In the BoolQ prompts, "only yes or no" was requested. In the GSM8K prompts, the model was asked to "solve the problem step by step" and to put the final numerical answer inside `\boxed{}`. Decoding was greedy (temperature 0, top-k = 1, fixed seed). The output budgets were 8, 4, and 512 tokens respectively, with a 2,048-token context window. Answers were extracted with deterministic regular expressions. The first option letter was taken for ARC, and the first yes/no/true/false for BoolQ. For GSM8K, the last `\boxed{}` number was taken, with a fallback to "the answer is" and then to the last number in the text. An unparsable answer was scored as incorrect. The parse-failure and truncation rates are reported for every configuration.

For Qwen3, both modes were run on the same 100 ARC and 100 GSM8K items. In non-thinking mode, the output limits were as above. In thinking mode, the budget was 4,096 tokens with an 8,192-token context. Reasoning tokens were included in all latency and energy figures.

## Measurement protocol and threats to validity

A dedicated Ollama server instance was started for benchmarking (one request at a time, at most one resident model, flash attention enabled). It was isolated from the shared system server, which was required to have no model loaded during measurement. The protocol that was followed for each configuration is given in [alg:protocol]. Any resident model was first unloaded. The target model was then loaded, and three warm-up requests were run and excluded from analysis. Twenty seconds of loaded-idle power were recorded. Finally, the items of every task were run back-to-back.

{{ALGO:Energy-instrumented benchmark protocol for one model configuration and one power mode|protocol}}
**Input:** model *m*, power mode *p*, task item sets *Q_τ* for τ in {ARC, BoolQ, GSM8K}
**Output:** one record per item with answer, correctness, energy, and timing
Set power mode *p* with nvpmodel; wait 30 s; record a 60 s idle baseline
Start the power sampler (50 Hz, four INA3221 rails, temperatures, GPU clock)
Wait until no foreign process exceeds 50% of one CPU core (idle-machine guard)
Unload any resident model; load *m*; run 3 warm-up requests (discarded)
Record 20 s of loaded-idle power; *P_idle* ← median module power
**for each** task τ **do**
  **for each** item *q* in *Q_τ* **do**
    *t*0 ← now; send the prompt of *q* with greedy decoding; *t*1 ← now
    *E_q* ← trapezoidal integral of *P_module* over [*t*0, *t*1]
    Split *E_q* into prefill and decode parts from the Ollama phase durations
    Parse the answer; score it against the gold label; append the record
    **every** 180 s: if foreign CPU load is detected **then** quarantine all records of (*m*, *p*) and restart from line 3
  **end for**
  Save the full-resolution power trace of task τ
**end for**
Unload *m*; restore MAXN at the end of the campaign
{{/ALGO}}

Because the device is a shared research machine, an **idle-machine guard** was enforced by the harness. Before a configuration was started, and every 180 s during it, it was checked that no foreign process exceeded 50% of a CPU core over a 5 s window. The periodic checks were made between requests, outside any measured window. If foreign load appeared, the data of the configuration were quarantined, and the configuration was re-run from scratch. In a pilot run that was made *without* this guard, run-to-run discrepancies of up to 58% in total energy were observed. This shows why such controls are needed for edge energy measurements.

The power mode was switched with `nvpmodel` and allowed to settle for 30 s before a fresh idle baseline was taken. Model outputs should be made independent of the power mode by greedy decoding. This was checked empirically (Results section), and the MAXN accuracy estimates were reused on that basis. The test–retest reliability of the energy measurements was assessed in two ways. First, the eight power-sweep configurations were repeated at MAXN and compared with the main grid on the overlapping items. Second, one configuration (Qwen2.5-1.5B FP16) was measured twice, 17 days apart, after an interruption of the campaign, and the two runs were compared on 600 overlapping items.

## Metrics

For each configuration *c* and task τ with item set *Q*, the following quantities are reported. The accuracy is the fraction of correctly answered items, as given in [eq:acc].

{{EQ:acc}}

The mean energy per query is the average of *E_q* over the items. The primary efficiency metric is the **joules per correct answer** (JPCA), as given in [eq:jpca].

{{EQ:jpca}}

JPCA is the expected energy that must be spent to obtain one correct answer. Cheap-but-wrong and expensive-but-accurate models are penalized by it in the same unit. In addition, the mean and 95th-percentile end-to-end latency and the time-to-first-token (load plus prefill) are reported. The decode and prefill throughput (tokens per second), the decode energy per output token, and the prefill energy per prompt token are also given. Finally, the energy–delay product (EDP) is reported. The EDP is given in [eq:edp] [@gonzalez1996energy].

{{EQ:edp}}

## Statistical analysis

Confidence intervals (95%) for accuracy and JPCA were obtained by non-parametric bootstrap over items (2,000 resamples) [@efron1979bootstrap]. For JPCA, the numerator and denominator were resampled jointly. Paired accuracy differences between configurations evaluated on the same items were tested with the exact McNemar test [@mcnemar1947note; @dietterich1998approximate]. The family-wise error rate across the pre-specified comparisons was controlled with Holm's procedure [@holm1979simple]. Paired-bootstrap confidence intervals were given to JPCA ratios between paired configurations. To relate energy to model and workload characteristics, an ordinary-least-squares model was fitted with heteroskedasticity-robust (HC3) standard errors. Log per-query energy was regressed on log weight-file size, log output length, log prompt length, and model family. A simple linear relation between decode energy per token and weight-file size was also fitted, motivated by the memory-bandwidth-bound nature of autoregressive decoding.

## Energy-aware cascade

Whether any single model can be beaten on the energy–accuracy plane by a combination of models was tested with an *agreement cascade*, which is given in [alg:cascade]. Every query is answered by two small models *s*1 and *s*2. If the same parsable answer is produced by both, that answer is returned. Otherwise, the query is escalated to a larger model *L*. Agreement between independent models is a training-free confidence signal, related to self-consistency [@wang2023selfconsistency] and to cascade routing [@chen2024frugalgpt]. The cascade energy is given in [eq:cascade].

{{EQ:cascade}}

{{ALGO:Agreement cascade with two small models and one large model|cascade}}
**Input:** query *q*; small models *s*1, *s*2; large model *L*; measured per-item energies *E*
**Output:** answer *a* and cascade energy *E_casc*
*a*1 ← answer of *s*1 to *q*; *a*2 ← answer of *s*2 to *q*
*E_casc* ← *E*(*s*1, *q*) + *E*(*s*2, *q*)
**if** *a*1 is parsable **and** *a*1 = *a*2 **then**
  *a* ← *a*1
**else**
  *a* ← answer of *L* to *q*; *E_casc* ← *E_casc* + *E*(*L*, *q*)
**end if**
**return** *a*, *E_casc*
{{/ALGO}}

The cascade energy was computed per item from the measured per-query energies of the main grid. The cascades were therefore evaluated on real per-item measurements, not on simulated costs. To avoid selection bias, the items of each task were randomly split in half. The cascade policy was chosen on the selection half and reported only on the held-out half. The policy was defined as the cheapest triple on the Pareto front of the selection half whose accuracy came within 1 percentage point of the best single model. As an upper bound on the achievable savings, an *oracle router*, by which each item is sent to the cheapest model that answers it correctly, is also reported.

## Reproducibility

All code (measurement harness, task definitions, and analysis), the raw per-query logs including model outputs, and the full-resolution power traces are released with the paper. The experiment can be re-run end to end with one script. The repository is available at https://github.com/mustafaalobaedy/Idea1_EdgeAI_LLM_Energy.
