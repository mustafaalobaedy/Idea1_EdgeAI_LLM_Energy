"""Background sampler for Jetson AGX Orin on-module power rails, temperatures and GPU state.

Reads the two INA3221 monitors exposed through hwmon. Module power is the sum of the
four monitored rails (VDD_GPU_SOC, VDD_CPU_CV, VIN_SYS_5V0, VDDQ_VDD2_1V8AO).
"""
import threading
import time

import numpy as np

_B = '/sys/bus/i2c/drivers/ina3221/'
RAILS = {
    'VDD_GPU_SOC': (_B + '1-0040/hwmon/hwmon1/in1_input', _B + '1-0040/hwmon/hwmon1/curr1_input'),
    'VDD_CPU_CV': (_B + '1-0040/hwmon/hwmon1/in2_input', _B + '1-0040/hwmon/hwmon1/curr2_input'),
    'VIN_SYS_5V0': (_B + '1-0040/hwmon/hwmon1/in3_input', _B + '1-0040/hwmon/hwmon1/curr3_input'),
    'VDDQ_VDD2_1V8AO': (_B + '1-0041/hwmon/hwmon2/in2_input', _B + '1-0041/hwmon/hwmon2/curr2_input'),
}
TEMPS = {'cpu': 0, 'gpu': 1, 'soc0': 5, 'tj': 8}
GPU_LOAD = '/sys/devices/platform/bus@0/17000000.gpu/load'
GPU_FREQ = '/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/cur_freq'
COLUMNS = ['t'] + ['P_' + r for r in RAILS] + ['P_total'] + ['T_' + k for k in TEMPS] + ['gpu_load', 'gpu_freq_mhz']


def _rd(f):
    f.seek(0)
    return int(f.read())


class PowerMonitor:
    def __init__(self, period_s=0.02):
        self.period = period_s
        self._rails = [(open(v), open(c)) for v, c in RAILS.values()]
        self._temps = [open(f'/sys/class/thermal/thermal_zone{z}/temp') for z in TEMPS.values()]
        self._load = open(GPU_LOAD)
        self._freq = open(GPU_FREQ)
        self._buf = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._th = None

    def _sample(self):
        p = [_rd(v) * _rd(c) / 1e6 for v, c in self._rails]  # mV*mA -> W
        temps = []
        for f in self._temps:
            try:
                temps.append(_rd(f) / 1000.0)
            except OSError:
                temps.append(np.nan)
        return [time.time()] + p + [sum(p)] + temps + [_rd(self._load) / 10.0, _rd(self._freq) / 1e6]

    def _run(self):
        nxt = time.time()
        while not self._stop.is_set():
            s = self._sample()
            with self._lock:
                self._buf.append(s)
            nxt += self.period
            time.sleep(max(0.0, nxt - time.time()))

    def start(self):
        self._stop.clear()
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()
        return self

    def stop(self):
        self._stop.set()
        self._th.join()

    def snapshot(self):
        with self._lock:
            return np.array(self._buf, dtype=float)

    def drain(self):
        """Return and clear all samples collected so far."""
        with self._lock:
            arr, self._buf = np.array(self._buf, dtype=float), []
        return arr


def integrate(trace, t0, t1, col=None):
    """Energy (J) of trace column between t0 and t1 using trapezoidal integration.
    Returns (energy_J, mean_power_W, n_samples)."""
    col = COLUMNS.index('P_total') if col is None else col
    t, p = trace[:, 0], trace[:, col]
    m = (t >= t0) & (t <= t1)
    if m.sum() < 2:  # interval shorter than the sampling period: use nearest samples
        i = np.clip(np.searchsorted(t, [t0, t1]), 0, len(t) - 1)
        pw = p[i[0]:i[1] + 1].mean() if i[1] >= i[0] else p[i[0]]
        return pw * (t1 - t0), pw, int(m.sum())
    tt = np.concatenate([[t0], t[m], [t1]])
    pp = np.concatenate([[np.interp(t0, t, p)], p[m], [np.interp(t1, t, p)]])
    e = np.trapezoid(pp, tt)
    return e, e / (t1 - t0), int(m.sum())
