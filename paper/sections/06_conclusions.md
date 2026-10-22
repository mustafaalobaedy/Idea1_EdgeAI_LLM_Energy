# Conclusions

The energy of a correct answer was measured for 26 small-language-model configurations on an NVIDIA Jetson AGX Orin, across quantization levels, two power modes, a reasoning switch, and three tasks. Five conclusions are drawn.

First, joules per correct answer separates configurations that tokens per joule cannot. The cheapest model per token (Qwen2.5-0.5B) was also the cheapest per correct answer, but only at accuracies below 65%. Once an accuracy floor of 80% is required, the best choices were Qwen2.5-3B Q4_K_M for ARC and GSM8K and Gemma 2-2B Q4_K_M for BoolQ. A 7B model at Q4_K_M was more accurate than a 3B model at FP16 from a smaller weight file, and it was cheaper per correct answer on the long-output task.

Second, the 4-bit level is the knee of the quantization ladder. Below it, accuracy collapsed by 8 to 31 points. Above it, accuracy did not change significantly, while energy per correct answer rose by up to 79%. FP16 never appeared on the Pareto front of the decode-bound task.

Third, decode energy per token is a linear function of the weight-file size with an effective bandwidth close to the memory peak and a fixed cost of about 0.35 J per token. Output length is the dominant driver of per-query energy, with an elasticity near one.

Fourth, the 40 W power mode cut energy by only 1 to 4% while raising latency by 10%, because the idle floor of about 10 W is paid for the whole duration. The model outputs were unchanged by the power mode under greedy decoding, and the energy measurements were repeatable to within a few percent across passes and across 17 days.

Fifth, reasoning and cascading did not pay on this device. Thinking multiplied the energy per query by 3 to 225 and brought no significant accuracy gain for the 1.7B and 8B hybrid models on 100 items per task. An agreement cascade of two small models and one large model was dominated by the best single model on every task, although an oracle router showed a three- to five-fold headroom.

The practical recommendation for an edge deployment of this class is therefore a 3B to 7B model at Q4_K_M. It should be run at the default power mode, with thinking disabled for tasks of this kind. The 30 W and 15 W modes, which need a reboot on this platform, and the search for a better routing signal than agreement are left for future work. All logs, traces, and code are released so that the campaign can be reproduced and extended.
