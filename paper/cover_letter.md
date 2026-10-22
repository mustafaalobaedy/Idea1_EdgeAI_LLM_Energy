22 October 2026

Editorial Office
PeerJ Computer Science

Dear Editors,

The manuscript "Joules per Correct Answer: Energy, Accuracy, and Latency Trade-offs of Quantized Small Language Models across Power Modes on an Edge GPU" is submitted for consideration as a research article in PeerJ Computer Science.

Small language models are now run on edge devices such as the NVIDIA Jetson AGX Orin, where every generated token is paid for from a limited power budget. Most measurement studies on such devices report tokens per second or tokens per joule. These metrics describe how fast a model writes, not whether the answer is right. In this work, the energy of a *correct answer* is measured. Twenty-seven model configurations from four open-weight families were served through Ollama at quantization levels from 3-bit to 16-bit weights and under the two stock power modes that can be switched at run time. Module power was sampled at 50 Hz and integrated over every one of 23,960 requests on ARC-Challenge, BoolQ, and GSM8K.

The main findings are as follows. Joules per correct answer ranged from 2.6 J to 12 J on the short-answer tasks and from 145 J to 791 J on GSM8K, and the cheapest correct answers above an 80% accuracy floor came from 2B to 3B models at 4-bit precision. The 4-bit level was the knee of a six-level quantization ladder: 3-bit weights lost 8 to 31 accuracy points, while higher precision gained nothing significant and cost up to 79% more energy per correct answer. Decode energy per token grew linearly with the weight-file size at an effective bandwidth close to the memory peak, with a fixed cost of 0.35 J per token. The 40 W power mode left every output unchanged, saved only 1% to 4% of energy, and raised latency by 10%. Reasoning ("thinking") multiplied the energy per query by 3 to 225 without a significant accuracy gain, and an agreement cascade of small models was dominated by the best single model, although an oracle router showed a three- to five-fold headroom.

The work fits the scope of PeerJ Computer Science in the areas of artificial intelligence, embedded computing, and green computing. All per-query logs, power traces, and code are released at https://github.com/mustafaalobaedy/Idea1_EdgeAI_LLM_Energy, so the campaign can be reproduced end to end with one script.

The manuscript has not been published previously and is not under consideration elsewhere. The author has approved the submission. No funding was received for this work, and the author has no competing interests. Generative AI tools were used for code assistance and language editing, as stated in the manuscript.

Thank you for considering this submission.

Sincerely,

Mustafa Muwafak Alobaedy
Centre for Image and Vision Computing, Multimedia University
63100 Cyberjaya, Malaysia
mustafa.alobaedy@mmu.edu.my
