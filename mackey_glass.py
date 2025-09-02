import time, serial, pyvisa
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
"""
Minimal photonic reservoir controller (Rigol HDO1074 + serial heaters)
WITH DEBUG PRINTS to understand what's happening

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
BAUD_RATE = 112500

# Scope: Rigol HDO1074 channels wired to your PDs (choose 4 or 7, etc.)
SCOPE_CHANNELS = [1, 2, 3, 4]

# Heaters you modulate with the input stream
INPUT_HEATERS = [33, 34, 35, 36, 37, 38, 39]

# All heaters addressable in your firmware (0..39 common in your code)
ALL_HEATERS = list(range(40))

# Safe heater voltages
V_MIN, V_MAX = 0.10, 4.90
V_BIAS_INTERNAL = 2.50   # internal mesh baseline
V_BIAS_INPUT = 2.50      # input heaters baseline

# Reservoir timing
T_SYMBOL = 0.25          # s per symbol (pick ~2× thermal tau as a start)
K_VIRTUAL = 4            # virtual nodes per symbol (time-multiplexing)
SETTLE = 0.05           # s to wait after each heater update
READ_AVG = 1             # scope reads to average each sample

# Input drive scaling (keep modest; too big will clip heaters)
MASK_GAIN = 2.5
MICRO_MASK_GAIN = 1.25

# Dataset sizes
TRAIN_STEPS = 200
TEST_STEPS = 100
WASHOUT = 10

# Task: 'mackey_glass' or 'narma10'
TASK = 'mackey_glass'

# Dry-run without hardware (simulated PD readings)
SIMULATION_MODE = False

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
    """Very small Rigol HDO reader using MEASure commands."""
    def __init__(self, channels):
        self.channels = channels
        if SIMULATION_MODE:
            self.rm = self.scope = None
            print("[DEBUG] Scope in SIMULATION_MODE - will return random values")
            return
        
        print("[DEBUG] Connecting to Rigol scope...")
        self.rm = pyvisa.ResourceManager()
        addr = self.rm.list_resources()[0]
        print(f"[DEBUG] Found scope at: {addr}")
        self.scope = self.rm.open_resource(addr)
        self.scope.timeout = 5000
        self.scope.read_termination = '\n'
        self.scope.write_termination = '\n'
        
        # Get scope ID for verification
        try:
            idn = self.scope.query('*IDN?')
            print(f"[DEBUG] Scope ID: {idn.strip()}")
        except Exception as e:
            print(f"[DEBUG] ERROR querying scope ID: {e}")
        
        # Clear and configure scope
        self.scope.write('*CLS')
        self.scope.write(':RUN')  # Make sure scope is running
        
        # Configure channels
        for ch in channels:
            self.scope.write(f':CHANnel{ch}:DISPlay ON')
            self.scope.write(f':CHANnel{ch}:SCALe 2')
            self.scope.write(f':CHANnel{ch}:OFFSet -6')
            print(f"[DEBUG] Configured channel {ch}")
        
        # Enable measurement system
        try:
            self.scope.write(':MEASure:STATe ON')
            print("[DEBUG] Measurement system enabled")
        except Exception as e:
            print(f"[DEBUG] Warning: Could not enable measurement system: {e}")
    
    def read_channel(self, ch):
        """Read a single channel using MEASure command."""
        if SIMULATION_MODE:
            # Return varying simulated values with some input dependence
            base = 2.0 + 0.3 * np.sin(time.time() * 2 + ch)  # Time-varying
            noise = 0.05 * np.random.randn()
            return float(base + noise)
        
        try:
            # Try different measurement types in order of preference
            measurement_types = ['VAVG', 'VMEAN', 'VMAX']  # Average, Mean, or Max
            
            for meas_type in measurement_types:
                try:
                    query = f':MEASure:STATistic:ITEM? CURRent,{meas_type},CHANnel{ch}'
                    value = float(self.scope.query(query))
                    
                    # Check if we got a reasonable voltage reading
                    if not np.isnan(value) and -10 <= value <= 10:  # Reasonable voltage range
                        return round(value, 5)
                    
                except Exception as e:
                    print(f"[DEBUG] {meas_type} measurement failed for CH{ch}: {e}")
                    continue
            
            # If all measurement types failed, try the waveform method as backup
            print(f"[DEBUG] All MEASure commands failed for CH{ch}, trying waveform method...")
            return self._read_waveform_mean_backup(ch)
            
        except Exception as e:
            print(f"[SCPI] Error reading channel {ch}: {e}")
            return np.nan
    
    def _read_waveform_mean_backup(self, ch: int) -> float:
        """Backup waveform reading method if MEASure fails."""
        s = self.scope
        try:
            # Set source & format
            s.write(f':WAVeform:SOURce CHANnel{ch}')
            s.write(':WAVeform:FORMat BYTE')
            s.write(':WAVeform:MODE NORMal')
            
            # Get scaling from preamble
            pre = s.query(':WAVeform:PREamble?').strip()
            print(f"[DEBUG] Ch{ch} preamble: {pre[:100]}...")  # Debug info
            
            fields = pre.split(',')
            if len(fields) < 10:
                print(f"[DEBUG] Incomplete preamble for CH{ch}: {len(fields)} fields")
                return np.nan
                
            y_incr = float(fields[7])  # Voltage increment
            y_orig = float(fields[8])  # Voltage origin
            y_ref = float(fields[9]) if len(fields) > 9 else 0.0  # Reference
            
            print(f"[DEBUG] Ch{ch} scaling: incr={y_incr}, orig={y_orig}, ref={y_ref}")
            
            # Fetch raw bytes
            raw = s.query_binary_values(':WAVeform:DATA?', datatype='B', is_big_endian=False)
            print(f"[DEBUG] Ch{ch} got {len(raw)} data points, first 10: {raw[:10]}")
            
            if len(raw) == 0:
                print(f"[DEBUG] No waveform data for CH{ch}")
                return np.nan
            
            data = np.array(raw, dtype=np.float64)
            
            # Convert to volts and return mean
            volts = (data - y_ref) * y_incr + y_orig
            mean_voltage = float(np.mean(volts))
            print(f"[DEBUG] Ch{ch} waveform mean: {mean_voltage:.4f}V")
            
            return mean_voltage
            
        except Exception as e:
            print(f"[SCPI] Error in backup waveform method for channel {ch}: {e}")
            return np.nan

    def read_many(self, avg=1):
        """Read all channels with averaging."""
        vals = []
        for ch in self.channels:
            samples = []
            for i in range(max(1, avg)):
                v = self.read_channel(ch)
                if np.isfinite(v): 
                    samples.append(v)
                else:
                    print(f"[DEBUG] Got invalid reading from CH{ch} (attempt {i+1}/{avg})")
                time.sleep(0.002)  # Small delay between readings
            
            if samples:
                avg_val = float(np.mean(samples))
                vals.append(avg_val)
                if len(self.channels) <= 4:  # Don't spam if many channels
                    print(f"[DEBUG] CH{ch}: {len(samples)} samples, avg={avg_val:.4f}V")
            else:
                print(f"[DEBUG] No valid samples from CH{ch}")
                vals.append(np.nan)
        
        return np.array(vals, float)

    def test_all_channels(self):
        """Test function to verify all channels are working."""
        print("\n[DEBUG] Testing all channels...")
        for ch in self.channels:
            reading = self.read_channel(ch)
            print(f"[DEBUG] Channel {ch} test: {reading}V")
        print("[DEBUG] Channel test complete\n")

    def close(self):
        if SIMULATION_MODE: 
            return
        try: 
            self.scope.close()
            print("[DEBUG] Scope connection closed")
        except: 
            pass
        try: 
            self.rm.close()
            print("[DEBUG] VISA resource manager closed")
        except: 
            pass

class HeaterBus:
    """Serial sender for 'heater,value;...\\n' strings."""
    def __init__(self):
        if SIMULATION_MODE:
            self.ser = None
            print("[DEBUG] HeaterBus in SIMULATION_MODE - will print commands only")
            return
        
        print(f"[DEBUG] Connecting to serial port {SERIAL_PORT} at {BAUD_RATE} baud...")
        self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
        time.sleep(0.4)
        self.ser.reset_input_buffer(); self.ser.reset_output_buffer()
        print("[DEBUG] Serial connection established")

    def send(self, config: dict):
        msg = "".join(f"{h},{float(v):.3f};" for h,v in config.items()) + "\n"
        
        if SIMULATION_MODE: 
            print(f"[DEBUG] Would send: {msg[:100]}..." if len(msg) > 100 else f"[DEBUG] Would send: {msg}")
            return
            
        self.ser.write(msg.encode()); self.ser.flush()

    def close(self):
        if SIMULATION_MODE: return
        try: self.ser.close()
        except: pass

# ==========================
# RESERVOIR CORE (WITH DEBUG)
# ==========================
class PhotonicReservoir:
    def __init__(self, input_heaters, all_heaters, scope_channels):
        print("\n" + "="*50)
        print("INITIALIZING PHOTONIC RESERVOIR")
        print("="*50)
        
        self.input_heaters = list(input_heaters)
        self.internal_heaters = [h for h in all_heaters if h not in self.input_heaters]
        
        print(f"[DEBUG] Input heaters: {self.input_heaters} ({len(self.input_heaters)} total)")
        print(f"[DEBUG] Internal heaters: {self.internal_heaters[:5]}...{self.internal_heaters[-5:]} ({len(self.internal_heaters)} total)")
        
        self.scope = RigolScope(scope_channels)
        self.bus = HeaterBus()

        # Fixed random mesh bias (to create mixing)
        print(f"[DEBUG] Creating random mesh bias (std=2.0, seed=42)...")
        rng = np.random.default_rng(42)
        self.mesh_bias = {
            h: float(np.clip(V_BIAS_INTERNAL + rng.normal(0, 2), V_MIN, V_MAX))
            for h in self.internal_heaters
        }
        
        # Input heater baseline
        self.input_bias = {h: V_BIAS_INPUT for h in self.input_heaters}
        
        print(f"[DEBUG] Mesh bias voltage range: {min(self.mesh_bias.values()):.2f}V to {max(self.mesh_bias.values()):.2f}V")
        print(f"[DEBUG] Input bias (baseline): {self.input_bias}")
        
        # Input masks (fixed)
        self.main_mask = rand_mask(len(self.input_heaters))
        self.micro_masks = [rand_mask(len(self.input_heaters)) for _ in range(K_VIRTUAL)]
        
        print(f"[DEBUG] Main mask: {self.main_mask}")
        print(f"[DEBUG] Micro masks: {[mm[:3] for mm in self.micro_masks]} (first 3 elements each)")

        # Apply initial baseline so the chip is in a defined state
        print(f"[DEBUG] Sending initial configuration ({len({**self.mesh_bias, **self.input_bias})} heaters)...")
        self.bus.send({**self.mesh_bias, **self.input_bias})
        time.sleep(0.2)
        print("[DEBUG] Reservoir initialization complete!\n")

    def _compose_config(self, s_scalar, k_subslot):
        cfg = dict(self.mesh_bias)
        mm = self.main_mask
        rr = self.micro_masks[k_subslot]
        
        for i, h in enumerate(self.input_heaters):
            base = self.input_bias[h]
            main_drive = MASK_GAIN * mm[i] * s_scalar
            micro_drive = MICRO_MASK_GAIN * rr[i]
            v = base + main_drive + micro_drive
            cfg[h] = float(np.clip(v, V_MIN, V_MAX))
        
        return cfg

    def drive(self, s, phase_name):
        """
        Return Z with shape (T, len(scope_channels)*K_VIRTUAL).
        Added phase_name for progress printing.
        """
        print(f"\n[DEBUG] Starting {phase_name} phase with {len(s)} steps")
        print(f"[DEBUG] Input signal stats: min={s.min():.3f}, max={s.max():.3f}, std={s.std():.3f}")
        print(f"[DEBUG] First 5 input values: {s[:5]}")
        
        T = len(s)
        D = len(SCOPE_CHANNELS) * K_VIRTUAL
        Z = np.zeros((T, D), float)
        
        # Store some stats for analysis
        all_pd_readings = []
        input_voltage_ranges = []

        for t in range(T):
            # Print progress every 100 steps
            if (t + 1) % 100 == 0:
                print(f"[{phase_name}] Step {t+1}/{T}")

            step_pd_readings = []
            step_input_voltages = []

            for k in range(K_VIRTUAL):
                config = self._compose_config(float(s[t]), k)
                
                # Collect input heater voltages for this step
                input_voltages = {h: config[h] for h in self.input_heaters}
                step_input_voltages.append(list(input_voltages.values()))
                
                # Debug print for first few steps
                if t < 3:
                    print(f"[DEBUG] Step {t}, K={k}: input={s[t]:.3f}")
                    print(f"         Input voltages: {input_voltages}")
                
                self.bus.send(config)
                time.sleep(SETTLE)
                y = self.scope.read_many(avg=READ_AVG)
                
                if t < 3:
                    print(f"         PD readings: {y}")
                
                step_pd_readings.extend(y)
                Z[t, k*len(SCOPE_CHANNELS):(k+1)*len(SCOPE_CHANNELS)] = y

            # Store stats
            all_pd_readings.append(step_pd_readings)
            input_voltage_ranges.append([min(voltages) for voltages in step_input_voltages])

            # pad to total T_SYMBOL if needed
            used = K_VIRTUAL * SETTLE
            if T_SYMBOL > used:
                time.sleep(T_SYMBOL - used)
        
        # Print summary stats
        all_pd_readings = np.array(all_pd_readings)
        input_voltage_ranges = np.array(input_voltage_ranges)
        
        print(f"\n[DEBUG] {phase_name} SUMMARY:")
        print(f"         PD readings shape: {all_pd_readings.shape}")
        print(f"         PD voltage range: {all_pd_readings.min():.3f}V to {all_pd_readings.max():.3f}V")
        print(f"         PD std dev: {all_pd_readings.std():.3f}V")
        print(f"         Input voltage range: {input_voltage_ranges.min():.3f}V to {input_voltage_ranges.max():.3f}V")
        print(f"         Feature matrix Z shape: {Z.shape}")
        print(f"         Z stats: min={Z.min():.3f}, max={Z.max():.3f}, std={Z.std():.3f}")
        
        return Z

    def close(self):
        print("[DEBUG] Closing reservoir connections...")
        self.scope.close()
        self.bus.close()

# ==========================
# MAIN
# ==========================
def main():
    print("PHOTONIC RESERVOIR COMPUTING - DEBUG MODE")
    print("="*60)
    
    # Build task sequences
    print("[DEBUG] Generating task data...")
    if TASK == 'mackey_glass':
        s_all = mackey_glass(TRAIN_STEPS + TEST_STEPS + WASHOUT + 10)
        s_drive, target = s_all[:-1], s_all[1:]
        print(f"[DEBUG] Generated Mackey-Glass sequence: {len(s_all)} points")
    elif TASK == 'narma10':
        u, y = narma10(TRAIN_STEPS + TEST_STEPS + WASHOUT + 10)
        s_drive, target = u, y
        print(f"[DEBUG] Generated NARMA10 sequence: {len(u)} points")
    else:
        raise ValueError("TASK must be 'mackey_glass' or 'narma10'")

    print(f"[DEBUG] Task signal stats: min={s_all.min():.3f}, max={s_all.max():.3f}")

    # Remove washout
    s_drive = s_drive[WASHOUT:]
    target  = target [WASHOUT:]

    # Split
    s_tr = s_drive[:TRAIN_STEPS]
    y_tr = target [:TRAIN_STEPS]
    s_te = s_drive[TRAIN_STEPS:TRAIN_STEPS+TEST_STEPS]
    y_te = target [TRAIN_STEPS:TRAIN_STEPS+TEST_STEPS]

    print(f"\n[DEBUG] Dataset split:")
    print(f"         Training: {len(s_tr)} samples")
    print(f"         Testing: {len(s_te)} samples")
    print(f"         Expected features per sample: {len(SCOPE_CHANNELS) * K_VIRTUAL}")

    print(f"\nReservoir run: TRAIN {len(s_tr)} | TEST {len(s_te)} | outs={len(SCOPE_CHANNELS)} | K={K_VIRTUAL}")

    res = PhotonicReservoir(INPUT_HEATERS, ALL_HEATERS, SCOPE_CHANNELS)

    try:
        print("Collecting training features...")
        Z_tr = res.drive(s_tr, "TRAIN")
        X_tr = build_features(Z_tr, quadratic=True)
        print(f"[DEBUG] Training features shape: {X_tr.shape}")

        print("\nCollecting test features...")
        Z_te = res.drive(s_te, "TEST")
        X_te = build_features(Z_te, quadratic=True)
        print(f"[DEBUG] Test features shape: {X_te.shape}")

        print("\nTraining ridge...")
        mdl, alpha = train_ridge(X_tr, y_tr)
        print(f"[DEBUG] Selected alpha = {alpha}")

        yhat_tr = mdl.predict(X_tr)
        yhat_te = mdl.predict(X_te)
        
        print(f"\n[DEBUG] FINAL RESULTS:")
        print(f"         NMSE train = {nmse(y_tr, yhat_tr):.4f}")
        print(f"         NMSE test  = {nmse(y_te, yhat_te):.4f}")
        
        # Interpretation
        train_nmse = nmse(y_tr, yhat_tr)
        test_nmse = nmse(y_te, yhat_te)
        
        print(f"\n[DEBUG] INTERPRETATION:")
        if test_nmse < 0.1:
            print("         EXCELLENT performance!")
        elif test_nmse < 0.5:
            print("         GOOD performance")
        elif test_nmse < 1.0:
            print("         MEDIOCRE performance")
        else:
            print("         POOR performance (worse than predicting average)")

    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        res.close()

if __name__ == "__main__":
    main()