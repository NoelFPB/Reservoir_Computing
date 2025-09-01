"""
Minimal photonic reservoir controller (Rigol HDO1074 + serial heaters)

What it does (end-to-end):
1) Randomize internal heaters once (fixed mesh "reservoir").
2) Drive scalar input sequence over time onto selected input heaters.
3) Read 4 PD channels from the scope, K times per symbol (virtual nodes).
4) Train a ridge regression readout for a simple task (MG/NARMA10).
5) Report NMSE.

Edit the CONFIG section only.
"""

# ==========================
# CONFIG (EDIT THESE)
# ==========================
SERIAL_PORT = 'COM3'
BAUD_RATE = 115200

# Scope: Rigol HDO1074 channels wired to your PDs (choose 4 or 7, etc.)
SCOPE_CHANNELS = [1, 2, 3, 4]

# Heaters you modulate with the input stream
INPUT_HEATERS = [32, 33, 34, 35, 36, 37]

# All heaters addressable in your firmware (0..39 common in your code)
ALL_HEATERS = list(range(40))

# Safe heater voltages
V_MIN, V_MAX = 0.20, 4.90
V_BIAS_INTERNAL = 2.50   # internal mesh baseline
V_BIAS_INPUT = 2.50      # input heaters baseline

# Reservoir timing
T_SYMBOL = 0.10          # s per symbol (pick ~2× thermal tau as a start)
K_VIRTUAL = 4            # virtual nodes per symbol (time-multiplexing)
SETTLE = 0.020           # s to wait after each heater update
READ_AVG = 3             # scope reads to average each sample

# Input drive scaling (keep modest; too big will clip heaters)
MASK_GAIN = 0.30
MICRO_MASK_GAIN = 0.10

# Dataset sizes
TRAIN_STEPS = 2000
TEST_STEPS = 800
WASHOUT = 200

# Task: 'mackey_glass' or 'narma10'
TASK = 'mackey_glass'

# Dry-run without hardware (simulated PD readings)
SIMULATION_MODE = False

# ==========================
# IMPORTS
# ==========================
import time, serial, pyvisa
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# ==========================
# SMALL HELPERS
# ==========================
def rand_mask(n):
    """Balanced ±1 mask of length n."""
    m = np.ones(n)
    m[np.random.choice(n, size=n//2, replace=False)] = -1
    return m

def mackey_glass(T, tau=17, beta=0.2, gamma=0.1, n=10, x0=1.2, dt=1.0):
    x = np.zeros(T + tau + 1)
    x[:tau+1] = x0
    for t in range(tau+1, T + tau + 1):
        x[t] = x[t-1] + dt * (beta * x[t-tau] / (1 + x[t-tau]**n) - gamma * x[t-1])
    seq = x[tau+1:]
    seq = (seq - seq.min()) / (seq.max() - seq.min() + 1e-12)
    return seq

def narma10(T, u=None):
    if u is None:
        u = 0.5 * np.random.rand(T)
    y = np.zeros(T)
    for t in range(10, T):
        y[t] = 0.3*y[t-1] + 0.05*y[t-1]*np.sum(y[t-10:t]) + 1.5*u[t-10]*u[t-1] + 0.1
    return u, y

def build_features(Z, quadratic=True):
    """Features = [1, Z, Z^2] (or [1, Z])."""
    if quadratic:
        return np.hstack([np.ones((len(Z),1)), Z, Z**2])
    return np.hstack([np.ones((len(Z),1)), Z])

def train_ridge(X, y, alphas=(1e-4,1e-3,1e-2,1e-1,1.0)):
    """Tiny CV to choose alpha, then fit."""
    best_a, best_mse = None, float('inf')
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for a in alphas:
        errs = []
        for tr, va in kf.split(X):
            mdl = Ridge(alpha=a, fit_intercept=False)
            mdl.fit(X[tr], y[tr])
            errs.append(mean_squared_error(y[va], mdl.predict(X[va])))
        m = float(np.mean(errs))
        if m < best_mse:
            best_mse, best_a = m, a
    mdl = Ridge(alpha=best_a, fit_intercept=False)
    mdl.fit(X, y)
    return mdl, best_a

def nmse(y, yhat):
    return float(np.sum((y - yhat)**2) / (np.sum((y - np.mean(y))**2) + 1e-12))

# ==========================
# HARDWARE WRAPPERS (MINIMAL)
# ==========================
class RigolScope:
    """Very small Rigol HDO reader: VAVG per channel, with a tiny fallback."""
    def __init__(self, channels):
        self.channels = channels
        if SIMULATION_MODE:
            self.rm = self.scope = None
            return
        self.rm = pyvisa.ResourceManager()
        addr = self.rm.list_resources()[0]
        self.scope = self.rm.open_resource(addr)
        self.scope.timeout = 5000
        self.scope.read_termination = '\n'
        self.scope.write_termination = '\n'
        _ = self.scope.query('*IDN?')
        self.scope.write('*CLS')
        self.scope.write(':RUN')
        for ch in channels:
            self.scope.write(f':CHANnel{ch}:DISPlay ON')
            self.scope.write(f':CHANnel{ch}:SCALe 2')
            self.scope.write(f':CHANnel{ch}:OFFSet 0')
        try:
            self.scope.write(':MEASure:STATe ON')
        except Exception:
            pass

    def read_channel(self, ch):
        if SIMULATION_MODE:
            return float(2.0 + 0.05*np.random.randn())
        s = self.scope
        try:
            s.write(':RUN')
            return float(s.query(f':MEASure:ITEM? VAVG,CHANnel{ch}'))
        except Exception:
            # tiny generic fallback
            try:
                return float(s.query(f':MEASure:VAVerage? CHANnel{ch}'))
            except Exception:
                return np.nan

    def read_many(self, avg=1):
        vals = []
        for ch in self.channels:
            samples = []
            for _ in range(max(1, avg)):
                v = self.read_channel(ch)
                if np.isfinite(v): samples.append(v)
                time.sleep(0.002)
            vals.append(float(np.mean(samples)) if samples else np.nan)
        return np.array(vals, float)

    def close(self):
        if SIMULATION_MODE: return
        try: self.scope.close()
        except: pass
        try: self.rm.close()
        except: pass

class HeaterBus:
    """Serial sender for 'heater,value;...\\n' strings."""
    def __init__(self):
        if SIMULATION_MODE:
            self.ser = None
            return
        self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
        time.sleep(0.4)
        self.ser.reset_input_buffer(); self.ser.reset_output_buffer()

    def send(self, config: dict):
        if SIMULATION_MODE: return
        msg = "".join(f"{h},{float(v):.3f};" for h,v in config.items()) + "\n"
        self.ser.write(msg.encode()); self.ser.flush()

    def close(self):
        if SIMULATION_MODE: return
        try: self.ser.close()
        except: pass

# ==========================
# RESERVOIR CORE (TINY)
# ==========================
class PhotonicReservoir:
    def __init__(self, input_heaters, all_heaters, scope_channels):
        self.input_heaters = list(input_heaters)
        self.internal_heaters = [h for h in all_heaters if h not in self.input_heaters]
        self.scope = RigolScope(scope_channels)
        self.bus = HeaterBus()

        # Fixed random mesh bias (to create mixing)
        rng = np.random.default_rng(42)
        self.mesh_bias = {
            h: float(np.clip(V_BIAS_INTERNAL + rng.normal(0, 0.8), V_MIN, V_MAX))
            for h in self.internal_heaters
        }
        # Input heater baseline
        self.input_bias = {h: V_BIAS_INPUT for h in self.input_heaters}

        # Input masks (fixed)
        self.main_mask = rand_mask(len(self.input_heaters))
        self.micro_masks = [rand_mask(len(self.input_heaters)) for _ in range(K_VIRTUAL)]

        # Apply initial baseline so the chip is in a defined state
        self.bus.send({**self.mesh_bias, **self.input_bias})
        time.sleep(0.1)

    def _compose_config(self, s_scalar, k_subslot):
        cfg = dict(self.mesh_bias)
        mm = self.main_mask
        rr = self.micro_masks[k_subslot]
        for i, h in enumerate(self.input_heaters):
            v = self.input_bias[h] + MASK_GAIN*mm[i]*s_scalar + MICRO_MASK_GAIN*rr[i]
            cfg[h] = float(np.clip(v, V_MIN, V_MAX))
        return cfg

    def drive(self, s):
        """Return Z with shape (T, len(scope_channels)*K_VIRTUAL)."""
        T = len(s)
        D = len(SCOPE_CHANNELS) * K_VIRTUAL
        Z = np.zeros((T, D), float)

        for t in range(T):
            for k in range(K_VIRTUAL):
                self.bus.send(self._compose_config(float(s[t]), k))
                time.sleep(SETTLE)
                y = self.scope.read_many(avg=READ_AVG)
                Z[t, k*len(SCOPE_CHANNELS):(k+1)*len(SCOPE_CHANNELS)] = y

            # pad to total T_SYMBOL if needed
            used = K_VIRTUAL * SETTLE
            if T_SYMBOL > used:
                time.sleep(T_SYMBOL - used)
        return Z

    def close(self):
        self.scope.close()
        self.bus.close()

# ==========================
# MAIN
# ==========================
def main():
    # Build task sequences
    if TASK == 'mackey_glass':
        s_all = mackey_glass(TRAIN_STEPS + TEST_STEPS + WASHOUT + 10)
        s_drive, target = s_all[:-1], s_all[1:]
    elif TASK == 'narma10':
        u, y = narma10(TRAIN_STEPS + TEST_STEPS + WASHOUT + 10)
        s_drive, target = u, y
    else:
        raise ValueError("TASK must be 'mackey_glass' or 'narma10'")

    # Remove washout
    s_drive = s_drive[WASHOUT:]
    target  = target [WASHOUT:]

    # Split
    s_tr = s_drive[:TRAIN_STEPS]
    y_tr = target [:TRAIN_STEPS]
    s_te = s_drive[TRAIN_STEPS:TRAIN_STEPS+TEST_STEPS]
    y_te = target [TRAIN_STEPS:TRAIN_STEPS+TEST_STEPS]

    print(f"Reservoir run: TRAIN {len(s_tr)} | TEST {len(s_te)} | outs={len(SCOPE_CHANNELS)} | K={K_VIRTUAL}")

    res = PhotonicReservoir(INPUT_HEATERS, ALL_HEATERS, SCOPE_CHANNELS)

    try:
        print("Collecting training features...")
        Z_tr = res.drive(s_tr)
        X_tr = build_features(Z_tr, quadratic=True)

        print("Collecting test features...")
        Z_te = res.drive(s_te)
        X_te = build_features(Z_te, quadratic=True)

        print("Training ridge...")
        mdl, alpha = train_ridge(X_tr, y_tr)
        print(f"alpha = {alpha}")

        yhat_tr = mdl.predict(X_tr)
        yhat_te = mdl.predict(X_te)
        print(f"NMSE train = {nmse(y_tr, yhat_tr):.4f}")
        print(f"NMSE test  = {nmse(y_te, yhat_te):.4f}")

    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        res.close()

if __name__ == "__main__":
    main()
