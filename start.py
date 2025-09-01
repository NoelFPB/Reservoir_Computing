import serial
import time
import pyvisa
import numpy as np
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# ==========================
# USER CONFIG (EDIT THESE)
# ==========================
SERIAL_PORT = 'COM4'
BAUD_RATE = 115200

# Photodiode channels on your scope (1..4 or 1..7)
SCOPE_CHANNELS = [1, 2, 3, 4]   # change to [1,2,3,4,5,6,7] if you have 7 outputs available
NUM_OUTPUTS = len(SCOPE_CHANNELS)

# Heaters you will modulate with the scalar input stream (1–7 heaters is fine)
INPUT_HEATERS = [36, 37]        # you can add more, e.g., [34,35,36,37,38,39,40]

# All other tunable heaters used as fixed mesh bias (random offsets)
ALL_HEATERS = list(range(40))   # edit if your address space is different
INTERNAL_HEATERS = [h for h in ALL_HEATERS if h not in INPUT_HEATERS]

# Safe voltage limits for heaters
V_MIN = 0.10
V_MAX = 4.90
V_BIAS = 2.50                  # mid-bias for internal heaters
V_INPUT_BIAS = 2.50            # mid-bias for input heaters

# Reservoir timing
T_SYMBOL = 0.10                # seconds per symbol (≈ 2× heater tau is a good start)
K_VIRTUAL = 4                  # virtual nodes per symbol (time-multiplexing factor)
SETTLE_PER_SUBSLOT = 0.020     # wait after updating heaters before reading (s)
READ_AVG_SAMPLES = 3           # average N reads for noise reduction

# Input drive scaling
MASK_GAIN = 0.30               # scales s_t onto heaters (fraction of full-scale)
MICRO_MASK_GAIN = 0.10         # tiny offset per sub-slot (jitter mask)

# Data lengths
TRAIN_STEPS = 3000
TEST_STEPS = 1000
WASHOUT = 200                  # discard initial transients

# Task: 'mackey_glass' or 'narma10'
TASK = 'mackey_glass'

# Optional: run without hardware for dry-run testing (simulated readings)
SIMULATION_MODE = False

# ==========================
# HELPER DATA STRUCTURES
# ==========================
@dataclass
class ReservoirConfig:
    input_heaters: List[int]
    internal_heaters: List[int]
    scope_channels: List[int]
    v_min: float
    v_max: float
    v_bias: float
    v_input_bias: float
    mask_gain: float
    micro_mask_gain: float
    t_symbol: float
    k_virtual: int
    settle_per_subslot: float
    read_avg_samples: int

# ==========================
# HARDWARE INTERFACE
# ==========================
class PhotonicReservoir:
    def __init__(self, cfg: ReservoirConfig):
        self.cfg = cfg
        self.serial = None
        self.scope = None
        self.rm = None

        # Precompute per-input random mask (fixed over the whole experiment)
        self.main_mask = self._make_mask(len(cfg.input_heaters))
        # Per-subslot micro-masks (K of them)
        self.micro_masks = [self._make_mask(len(cfg.input_heaters)) for _ in range(cfg.k_virtual)]

        # Fixed random biases for the internal heaters (mesh mixing)
        rng = np.random.default_rng(42)
        self.mesh_bias = {
            h: float(np.clip(cfg.v_bias + rng.normal(0, 0.25), cfg.v_min, cfg.v_max))
            for h in cfg.internal_heaters
        }

        # Input heater biases
        self.input_bias = {h: cfg.v_input_bias for h in cfg.input_heaters}

        if not SIMULATION_MODE:
            self._init_serial()
            self._init_scope()
            self._prime_scope()

    def _make_mask(self, n: int) -> np.ndarray:
        # entries in {-1, +1} with ~balanced signs
        m = np.ones(n)
        idx = np.random.choice(n, size=n//2, replace=False)
        m[idx] = -1
        return m

    def _init_serial(self):
        self.serial = serial.Serial(SERIAL_PORT, BAUD_RATE)
        time.sleep(0.8)
        self.serial.reset_input_buffer()
        self.serial.reset_output_buffer()

    def _init_scope(self):
        self.rm = pyvisa.ResourceManager()
        resources = self.rm.list_resources()
        if not resources:
            raise RuntimeError("No VISA resources found for scope")
        # Open the first resource by default (edit if you need a specific VISA address)
        self.scope = self.rm.open_resource(resources[0])
        self.scope.timeout = 5000

    def _prime_scope(self):
        # Turn on and set up the channels we will read
        for ch in self.cfg.scope_channels:
            self.scope.write(f':CHANnel{ch}:DISPlay ON')
            self.scope.write(f':CHANnel{ch}:SCALe 2')
            self.scope.write(f':CHANnel{ch}:OFFSet 0')

        # Optional: set acquisition mode to average if your scope supports it (comment if not)
        # self.scope.write(':ACQuire:TYPE NORMal')
        # self.scope.write(':MEASure:CLEar')

    def cleanup(self):
        try:
            if self.serial:
                self.serial.close()
            if self.scope:
                self.scope.close()
            if self.rm:
                self.rm.close()
        except Exception:
            pass

    # ---------- SERIAL COMM ----------
    def send_heater_values(self, config: Dict[int, float]):
        """
        Same wire format you already use: 'heater,value;...\\n'
        """
        msg = "".join(f"{h},{float(v):.3f};" for h, v in config.items()) + "\n"
        if SIMULATION_MODE:
            return
        self.serial.write(msg.encode())
        self.serial.flush()

    # ---------- SCOPE READ ----------
    def _read_channel_once(self, ch: int) -> Optional[float]:
        """
        Try a few common SCPI queries. Replace the first working line with your scope's preferred command.
        Returns a single float voltage sample (V).
        """
        if SIMULATION_MODE:
            # Simulated PD reading in volts (noisy sigmoid-ish response)
            return float(np.clip(2.0 + 0.05*np.random.randn(), 0.0, 5.0))

        try:
            # Try a simple DC measurement (common on many scopes)
            # 1) Tek/Keysight style:
            try:
                v = float(self.scope.query(f':MEASure:VAVerage? CHANnel{ch}'))
                return v
            except Exception:
                pass

            # 2) Fallback: statistic current mean
            try:
                v = float(self.scope.query(f':MEASure:STATistic:ITEM? CURRent,MEAN,CHANnel{ch}'))
                return v
            except Exception:
                pass

            # 3) Last resort: peak (not ideal, but better than nothing)
            v = float(self.scope.query(f':MEASure:STATistic:ITEM? CURRent,VMAX,CHANnel{ch}'))
            return v
        except Exception:
            return None

    def read_outputs(self, avg_samples: int = 1) -> np.ndarray:
        vals = []
        for ch in self.cfg.scope_channels:
            samples = []
            for _ in range(avg_samples):
                v = self._read_channel_once(ch)
                if v is not None:
                    samples.append(v)
                # tiny pause between repeated reads
                time.sleep(0.002)
            if len(samples) == 0:
                vals.append(np.nan)
            else:
                vals.append(float(np.mean(samples)))
        return np.array(vals, dtype=float)

    # ---------- DRIVE & RECORD ----------
    def _compose_config(self, s_scalar: float, subslot_idx: int) -> Dict[int, float]:
        """
        Build heater vector for this (symbol, subslot):
          V_input = bias + MASK_GAIN * main_mask * s + MICRO_MASK_GAIN * micro_mask[subslot]
          V_internal = fixed mesh_bias
        """
        cfg = {}
        # internal biases
        cfg.update(self.mesh_bias)

        # inputs
        mm = self.main_mask
        rr = self.micro_masks[subslot_idx]
        for i, h in enumerate(self.cfg.input_heaters):
            v = self.input_bias[h] + self.cfg.mask_gain * mm[i] * s_scalar + self.cfg.micro_mask_gain * rr[i]
            v = float(np.clip(v, self.cfg.v_min, self.cfg.v_max))
            cfg[h] = v

        return cfg

    def drive_sequence(self, s: np.ndarray) -> np.ndarray:
        """
        Drive the reservoir with scalar input sequence s[t] of length T.
        Returns features matrix Z of shape (T, NUM_OUTPUTS * K_VIRTUAL)
        """
        T = len(s)
        feats = np.zeros((T, NUM_OUTPUTS * self.cfg.k_virtual), dtype=float)

        for t in range(T):
            for k in range(self.cfg.k_virtual):
                # Apply heater values for this sub-slot
                cfg = self._compose_config(float(s[t]), k)
                self.send_heater_values(cfg)

                # Let things settle (crucial vs heater tau)
                time.sleep(self.cfg.settle_per_subslot)

                # Read PDs (average a few samples)
                y = self.read_outputs(avg_samples=self.cfg.read_avg_samples)
                feats[t, k*NUM_OUTPUTS:(k+1)*NUM_OUTPUTS] = y

            # Optional: ensure one full symbol duration passes
            total_used = self.cfg.k_virtual * (self.cfg.settle_per_subslot)
            if self.cfg.t_symbol > total_used:
                time.sleep(self.cfg.t_symbol - total_used)

        return feats

# ==========================
# TASK GENERATORS
# ==========================
def gen_mackey_glass(T: int, tau: int = 17, beta=0.2, gamma=0.1, n=10, x0=1.2, dt=1.0) -> np.ndarray:
    """
    Simple discrete Mackey-Glass generator (normalized to ~[0,1]).
    """
    x = np.zeros(T + tau + 1)
    x[:tau+1] = x0
    for t in range(tau+1, T + tau + 1):
        x[t] = x[t-1] + dt * (beta * x[t-tau] / (1 + x[t-tau]**n) - gamma * x[t-1])
    seq = x[tau+1:]
    # Normalize to 0..1
    seq = (seq - np.min(seq)) / (np.max(seq) - np.min(seq) + 1e-9)
    return seq

def gen_narma10(T: int, u: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    NARMA10 benchmark. Returns (u, y) with u in [0,0.5].
    """
    if u is None:
        u = 0.5 * np.random.rand(T)
    y = np.zeros(T)
    for t in range(10, T):
        y[t] = 0.3*y[t-1] + 0.05*y[t-1]*np.sum(y[t-10:t]) + 1.5*u[t-10]*u[t-1] + 0.1
    return u, y

# ==========================
# TRAIN / EVAL
# ==========================
def build_features(Z_raw: np.ndarray, quadratic: bool = True) -> np.ndarray:
    """
    Expand features with bias and (optional) per-channel quadratic terms.
    """
    T, D = Z_raw.shape
    if quadratic:
        Zq = Z_raw**2
        X = np.hstack([np.ones((T,1)), Z_raw, Zq])
    else:
        X = np.hstack([np.ones((T,1)), Z_raw])
    return X

def train_ridge(X: np.ndarray, y: np.ndarray, alphas=(1e-4, 1e-3, 1e-2, 1e-1, 1.0)) -> Tuple[Ridge, float]:
    """
    Simple K-fold CV to pick alpha, then fit Ridge.
    """
    best_alpha = None
    best_score = float('inf')
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for a in alphas:
        errs = []
        for tr, va in kf.split(X):
            model = Ridge(alpha=a, fit_intercept=False)  # intercept already in features
            model.fit(X[tr], y[tr])
            yhat = model.predict(X[va])
            errs.append(mean_squared_error(y[va], yhat))
        m = float(np.mean(errs))
        if m < best_score:
            best_score = m
            best_alpha = a
    model = Ridge(alpha=best_alpha, fit_intercept=False)
    model.fit(X, y)
    return model, best_alpha

def nmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sum((y_true - y_pred)**2) / (np.sum((y_true - np.mean(y_true))**2) + 1e-12))

# ==========================
# MAIN PIPELINE
# ==========================
def main():
    cfg = ReservoirConfig(
        input_heaters=INPUT_HEATERS,
        internal_heaters=INTERNAL_HEATERS,
        scope_channels=SCOPE_CHANNELS,
        v_min=V_MIN, v_max=V_MAX,
        v_bias=V_BIAS, v_input_bias=V_INPUT_BIAS,
        mask_gain=MASK_GAIN, micro_mask_gain=MICRO_MASK_GAIN,
        t_symbol=T_SYMBOL, k_virtual=K_VIRTUAL,
        settle_per_subslot=SETTLE_PER_SUBSLOT,
        read_avg_samples=READ_AVG_SAMPLES
    )

    res = PhotonicReservoir(cfg)

    try:
        # --------- Build task sequences ---------
        if TASK == 'mackey_glass':
            s_all = gen_mackey_glass(TRAIN_STEPS + TEST_STEPS + WASHOUT + 10)
            # Input is s(t); target is s(t+1)
            s_drive = s_all[:-1]
            target = s_all[1:]
        elif TASK == 'narma10':
            u_all, y_all = gen_narma10(TRAIN_STEPS + TEST_STEPS + WASHOUT + 10)
            s_drive = u_all  # drive is input u(t)
            target = y_all   # target is y(t)
        else:
            raise ValueError("TASK must be 'mackey_glass' or 'narma10'")

        # Trim washout
        s_drive = s_drive[WASHOUT:]
        target = target[WASHOUT:]

        # Split train/test
        s_train = s_drive[:TRAIN_STEPS]
        y_train = target[:TRAIN_STEPS]
        s_test  = s_drive[TRAIN_STEPS:TRAIN_STEPS+TEST_STEPS]
        y_test  = target[TRAIN_STEPS:TRAIN_STEPS+TEST_STEPS]

        print(f"Driving reservoir: TRAIN {len(s_train)} steps, TEST {len(s_test)} steps; outputs={NUM_OUTPUTS}, K={cfg.k_virtual}")

        # --------- Drive & record ---------
        print("Collecting training features...")
        Z_train = res.drive_sequence(s_train)
        X_train = build_features(Z_train, quadratic=True)

        print("Collecting test features...")
        Z_test = res.drive_sequence(s_test)
        X_test = build_features(Z_test, quadratic=True)

        # --------- Train readout ---------
        print("Training ridge regression (with CV)...")
        model, alpha = train_ridge(X_train, y_train)
        print(f"Chosen alpha = {alpha}")

        # --------- Evaluate ---------
        y_pred_train = model.predict(X_train)
        y_pred_test  = model.predict(X_test)
        nmse_tr = nmse(y_train, y_pred_train)
        nmse_te = nmse(y_test, y_pred_test)
        print(f"Train NMSE: {nmse_tr:.4f}")
        print(f"Test  NMSE: {nmse_te:.4f}")

        # Small summary
        print("\n--- SUMMARY ---")
        print(f"Outputs: {NUM_OUTPUTS} | Virtual nodes: {cfg.k_virtual} | Features: {X_train.shape[1]}")
        print(f"Task: {TASK} | Train NMSE: {nmse_tr:.4f} | Test NMSE: {nmse_te:.4f}")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        res.cleanup()

if __name__ == "__main__":
    main()
