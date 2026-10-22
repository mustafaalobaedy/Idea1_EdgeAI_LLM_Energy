#!/bin/bash
# Power-mode sweep for a mode that needs a reboot on JetPack 6 (MODE_30W, MODE_15W change the GPU TPC mask).
#
# Procedure (each mode costs two reboots of the shared machine):
#   1. sudo nvpmodel -m 2      # 2 = MODE_30W, 1 = MODE_15W; answer "yes" to the reboot prompt
#   2. after the reboot:  cd <project>; nohup bash code/run_mode_after_reboot.sh MODE_30W >> logs/campaign.log 2>&1 &
#   3. when it prints "MODE DONE", switch back:  sudo nvpmodel -m 0   (answer "yes" to reboot into MAXN)
# bench.py is resumable, so an interrupted pass can be relaunched with the same command.
cd "$(dirname "$0")/.."
MODE=${1:?usage: run_mode_after_reboot.sh MODE_30W|MODE_15W}
PY=/mnt/data/miniforge3/envs/edgeai/bin/python
cur=$(nvpmodel -q | head -1)
[[ "$cur" == *"$MODE"* ]] || { echo "$(date) device is in '$cur', not $MODE; run: sudo nvpmodel -m <id> and reboot first"; exit 1; }
export OLLAMA_HOST=127.0.0.1:11500
if ! curl -s -m 3 http://127.0.0.1:11500/api/ps >/dev/null; then
  setsid nohup bash code/start_server.sh >> logs/ollama_bench_server.log 2>&1 < /dev/null &
  for i in $(seq 1 30); do curl -s -m 3 http://127.0.0.1:11500/api/ps >/dev/null && break; sleep 2; done
fi
echo "===== $(date) power-mode pass $MODE ====="
$PY -c "import sys;sys.path.insert(0,'code');import bench;bench.wait_idle()"
sleep 30
$PY code/idle_baseline.py $MODE 60
E2_MODELS="qwen2.5:0.5b-instruct-q4_K_M qwen2.5:1.5b-instruct-q4_K_M qwen2.5:3b-instruct-q4_K_M qwen2.5:3b-instruct-q8_0 qwen2.5:3b-instruct-fp16 qwen2.5:7b-instruct-q4_K_M llama3.2:3b-instruct-q4_K_M gemma2:2b-instruct-q4_K_M"
$PY code/bench.py --mode $MODE --tag E2_modes --models $E2_MODELS --n arc=100 boolq=100 gsm8k=60
echo "$(date) MODE DONE: $MODE  (now: sudo nvpmodel -m 0 and reboot to return to MAXN)"
