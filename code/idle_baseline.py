"""Record module idle power (no model resident) for the current power mode."""
import json, os, subprocess, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from power_monitor import COLUMNS, PowerMonitor
mode, secs = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 60
m = PowerMonitor(0.02).start(); time.sleep(secs); tr = m.drain(); m.stop()
out = os.path.join(os.path.dirname(__file__), '..', 'results', 'idle_baseline.jsonl')
rec = dict(mode=mode, time=time.time(), secs=secs, **{c: float(np.nanmedian(tr[:, i])) for i, c in enumerate(COLUMNS) if c != 't'},
           nvpmodel=subprocess.run(['nvpmodel', '-q'], capture_output=True, text=True).stdout.split('\n')[0])
open(out, 'a').write(json.dumps(rec) + '\n'); print(rec)
np.savez_compressed(out.replace('.jsonl', f'_{mode}_{int(time.time())}.npz'), trace=tr, columns=np.array(COLUMNS))
