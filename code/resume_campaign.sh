#!/bin/bash
# Resume the campaign after an interruption: start the bench server if needed,
# re-pull any model missing from the bench store, then run the normal campaign
# (bench.py is resumable, so completed configurations are skipped).
cd "$(dirname "$0")/.."
export OLLAMA_HOST=127.0.0.1:11500
if ! curl -s -m 3 http://127.0.0.1:11500/api/ps >/dev/null; then
  setsid nohup bash code/start_server.sh >> logs/ollama_bench_server.log 2>&1 < /dev/null &
  for i in $(seq 1 30); do curl -s -m 3 http://127.0.0.1:11500/api/ps >/dev/null && break; sleep 2; done
fi
echo "$(date) bench server up: $(curl -s -m 3 http://127.0.0.1:11500/api/version)"
STORE=/mnt/data/private_ai/ollama/manifests/registry.ollama.ai/library
while read m; do
  n=${m%%:*}; t=${m##*:}
  if [ ! -f "$STORE/$n/$t" ]; then
    echo "== pull $m" >> logs/pull.log
    until ollama pull "$m" >> logs/pull.log 2>&1; do echo "retry $m" >> logs/pull.log; sleep 60; done
  fi
done < code/models.txt
missing=0
while read m; do n=${m%%:*}; t=${m##*:}; [ -f "$STORE/$n/$t" ] || { echo "STILL MISSING $m"; missing=1; }; done < code/models.txt
[ $missing = 0 ] || exit 1
echo "$(date) all models present, starting campaign"
exec bash code/run_campaign.sh
