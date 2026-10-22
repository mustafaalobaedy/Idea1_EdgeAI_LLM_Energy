#!/bin/bash
# Full experimental campaign. Restores MAXN on exit no matter what.
cd "$(dirname "$0")/.."
PY=/mnt/data/miniforge3/envs/edgeai/bin/python
trap 'echo no | sudo nvpmodel -m 0 >/dev/null 2>&1; echo "$(date) restored MAXN"' EXIT
set_mode () {  # $1 id $2 name
  echo no | sudo nvpmodel -m $1 >/dev/null 2>&1
  sleep 30
  cur=$(nvpmodel -q | head -1)
  echo "$(date) power mode -> $cur"
  [[ "$cur" == *"$2"* ]] || { echo "FAILED to set $2"; exit 1; }
  $PY code/idle_baseline.py $2 60
}
E1_MODELS=$(grep -v '^qwen3' code/models.txt | tr '\n' ' ')
E2_MODELS="qwen2.5:0.5b-instruct-q4_K_M qwen2.5:1.5b-instruct-q4_K_M qwen2.5:3b-instruct-q4_K_M qwen2.5:3b-instruct-q8_0 qwen2.5:3b-instruct-fp16 qwen2.5:7b-instruct-q4_K_M llama3.2:3b-instruct-q4_K_M gemma2:2b-instruct-q4_K_M"
E2_N="arc=100 boolq=100 gsm8k=60"
Q3="qwen3:1.7b qwen3:4b qwen3:8b"

until grep -q ALLDONE logs/pull.log; do sleep 60; done; echo "$(date) all models downloaded"
# wait until the machine is free before touching power modes
$PY -c "import sys;sys.path.insert(0,'code');import bench;bench.wait_idle()"

set_mode 0 MAXN
$PY code/bench.py --mode MAXN --tag E1_main --models $E1_MODELS --n arc=300 boolq=300 gsm8k=200
$PY code/bench.py --mode MAXN --tag E2_modes --models $E2_MODELS --n $E2_N
$PY code/bench.py --mode MAXN --tag E3_think --models $Q3 --think off --tasks arc gsm8k --n arc=100 gsm8k=100
$PY code/bench.py --mode MAXN --tag E3_think --models $Q3 --think on  --tasks arc gsm8k --n arc=100 gsm8k=100 --num_ctx 8192 --think_tokens 4096
# Power-mode sweep: only MODE_40W can be switched at run time; MODE_30W and MODE_15W change the GPU
# power-gating mask and need a reboot on JetPack 6, so they are outside this campaign (see run_mode_after_reboot.sh).
set_mode 3 MODE_40W
$PY code/bench.py --mode MODE_40W --tag E2_modes --models $E2_MODELS --n $E2_N
echo "$(date) CAMPAIGN COMPLETE"
